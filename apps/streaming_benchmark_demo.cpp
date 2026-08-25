// Usage:
//   streaming_benchmark_demo <model_package_dir> [core_config_json]

#include <atomic>
#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
  #include <io.h>
  #ifndef STDIN_FILENO
    #define STDIN_FILENO 0
  #endif
  #ifndef ssize_t
    typedef intptr_t ssize_t;
  #endif
  #define read _read
#else
  #include <unistd.h>
#endif

#include "core/config.hpp"
#include "engine/inference_engine.hpp"

namespace {

std::atomic_bool g_cancelled{false};

extern "C" void handle_sigint(int) { g_cancelled.store(true, std::memory_order_relaxed); }

bool cancellation_check(void* user_data) {
    auto* flag = static_cast<std::atomic_bool*>(user_data);
    return flag->load(std::memory_order_relaxed);
}

nlohmann::json load_core_config(const std::string& package_root,
                                const std::string& override_path) {
    std::ifstream in;
    if (!override_path.empty()) {
        in.open(override_path);
    }
    if (!in.is_open()) {
        in.open(std::filesystem::path(package_root) / "core_configs/default.json");
    }
    if (!in.is_open()) {
        in.open("core_configs/default.json");
    }
    nlohmann::json options = nlohmann::json::object();
    if (in.is_open()) {
        in >> options;
    }
    return options;
}

enum class ReadStatus { Ok, Eof, Interrupted };

class LineReader {
public:
    ReadStatus next(std::string& out) {
        out.clear();
        for (;;) {
            if (g_cancelled.load(std::memory_order_relaxed) && pending_.empty()) {
                return ReadStatus::Interrupted;
            }
            const std::string::size_type newline = pending_.find('\n');
            if (newline != std::string::npos) {
                out = pending_.substr(0, newline);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                pending_.erase(0, newline + 1);
                return ReadStatus::Ok;
            }
            char buffer[4096];
            const ssize_t count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count > 0) {
                pending_.append(buffer, static_cast<size_t>(count));
            } else if (count == 0) {
                if (pending_.empty()) return ReadStatus::Eof;
                out = pending_;
                pending_.clear();
                return ReadStatus::Ok;
            } else if (errno == EINTR) {
                if (g_cancelled.load(std::memory_order_relaxed)) return ReadStatus::Interrupted;
            } else {
                return ReadStatus::Eof;
            }
        }
    }

private:
    std::string pending_;
};

std::vector<std::string> split_whitespace(const std::string& line) {
    std::vector<std::string> words;
    std::istringstream stream(line);
    std::string word;
    while (stream >> word) words.push_back(word);
    return words;
}

std::string join(const std::vector<std::string>& words, const std::string& separator = ", ") {
    std::string result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) result += separator;
        result += words[i];
    }
    return result;
}

void print_separator() { std::cout << std::string(72, '-') << '\n'; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model_package_dir> [core_config_json]" << std::endl;
        return 2;
    }

    std::signal(SIGINT, handle_sigint);
    const std::string package_root = argv[1];
    const std::string core_config_path = argc > 2 ? argv[2] : std::string();
    std::cout << "Loading model package from: " << package_root << std::endl;

    phono::engine::InferenceEngine engine(package_root);
    const auto& cfg = engine.config();
    const auto& tokenizer = engine.tokenizer();

    const nlohmann::json core_options = load_core_config(package_root, core_config_path);
    phono::core::CoreConfig core_config;
    const phono::core::CoreConfigError config_error =
        phono::core::parse_core_config(core_options, cfg, core_config);
    if (config_error != phono::core::CoreConfigError::Ok) {
        std::cerr << "invalid core config: "
                  << phono::core::core_config_error_name(config_error) << std::endl;
        return static_cast<int>(config_error);
    }

    phono::context::ContextManager context_manager(cfg, 1, core_config);
    phono::context::Context* slot = context_manager.get_context_auto({});
    phono::engine::InferenceSession session(engine, core_config);

    std::cout << "  context_vocab_size = " << tokenizer.context_vocab_size() << '\n'
              << "  pinyin_vocab_size  = " << tokenizer.pinyin_vocab_size() << '\n'
              << "  chinese_vocab_size = " << tokenizer.chinese_vocab_size() << '\n'
              << "  beam size          = " << session.beam_size() << '\n'
              << "  pre_model.max_seqlen = " << cfg.pre_model.max_seqlen << '\n'
              << "  post_model.max_seqlen = " << cfg.post_model.max_seqlen << '\n'
              << "  max_history_length  = " << core_config.max_history_length << '\n'
              << "  max_pinyin_length   = " << core_config.max_pinyin_length << '\n';
    print_separator();
    std::cout << "Type space-separated pinyin (e.g. 'ni hao'), then pick a candidate."
              << " Press Ctrl-C to stop.\n";
    print_separator();

    LineReader reader;
    std::string committed_text;
    size_t window_count = 0;
    size_t committed_chars = 0;
    std::vector<double> latencies_us;
    int exit_code = 0;

    for (;;) {
        std::cout << "pinyin> " << std::flush;
        std::string line;
        if (reader.next(line) != ReadStatus::Ok) break;
        const std::vector<std::string> pinyin = split_whitespace(line);
        if (pinyin.empty()) continue;

        const std::vector<int32_t> pinyin_ids = tokenizer.encode_pinyin(pinyin);
        const std::vector<int32_t> context_ids = tokenizer.encode_context(committed_text);
        slot = context_manager.get_context_auto(context_ids);
        const auto start = std::chrono::high_resolution_clock::now();
        const phono::engine::GenerateResult generated =
            session.generate(*slot, pinyin_ids, context_ids, cancellation_check, &g_cancelled);
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);

        if (generated.error != phono::engine::InferenceError::Ok) {
            std::cout << "  ! generation failed: "
                      << phono::engine::inference_error_name(generated.error) << '\n';
            if (generated.error != phono::engine::InferenceError::Cancelled) {
                exit_code = static_cast<int>(generated.error);
            }
            break;
        }
        ++window_count;
        std::cout << "  context: \"" << committed_text << "\"\n"
                  << "  pinyin : [" << join(pinyin) << "]\n"
                  << "  candidates (" << std::fixed << std::setprecision(1) << elapsed_us
                  << " us):\n";
        for (size_t i = 0; i < generated.beams.size(); ++i) {
            std::cout << "    [" << i + 1 << "] score=" << std::setprecision(4)
                      << generated.beams[i].score << " \"" << generated.beams[i].decoded
                      << "\"\n";
        }
        if (generated.beams.empty()) continue;

        std::cout << "select (1-" << generated.beams.size() << ", or ctrl-c to stop)> "
                  << std::flush;
        std::string selection;
        if (reader.next(selection) != ReadStatus::Ok) break;
        int choice = 0;
        try {
            choice = std::stoi(selection);
        } catch (...) {
            choice = 0;
        }
        if (choice < 1 || choice > static_cast<int>(generated.beams.size())) {
            std::cout << "  ! invalid choice\n";
            continue;
        }
        const std::string& chosen = generated.beams[static_cast<size_t>(choice - 1)].decoded;
        std::vector<int32_t> new_context_ids;
        new_context_ids.reserve(generated.beams[static_cast<size_t>(choice - 1)].pred_ids.size());
        for (const int32_t chinese_id : generated.beams[static_cast<size_t>(choice - 1)].pred_ids) {
            const int32_t context_id = tokenizer.chinese_id_to_context_id(chinese_id);
            if (context_id < 0) {
                std::cout << "  ! selected candidate cannot be fed to context vocab\n";
                new_context_ids.clear();
                break;
            }
            new_context_ids.push_back(context_id);
        }
        if (new_context_ids.empty()) continue;
        const auto fill_error = session.fill(*slot, new_context_ids);
        if (fill_error != phono::engine::InferenceError::Ok) {
            std::cout << "  ! context fill failed: "
                      << phono::engine::inference_error_name(fill_error) << '\n';
            exit_code = static_cast<int>(fill_error);
            break;
        }
        committed_text += chosen;
        committed_chars += new_context_ids.size();
        std::cout << "  committed: \"" << chosen << "\"\n";
    }

    print_separator();
    std::cout << "Benchmark report\n"
              << "  windows            : " << window_count << '\n'
              << "  committed chars    : " << committed_chars << '\n'
              << "  committed text     : \"" << committed_text << "\"\n"
              << "  history length     : " << slot->context_ids_len() << '\n';
    if (!latencies_us.empty()) {
        const double total = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        std::cout << "  generation latency (us):\n"
                  << "    total            : " << total << '\n'
                  << "    average          : " << total / latencies_us.size() << '\n';
    }
    print_separator();
    return exit_code;
}
