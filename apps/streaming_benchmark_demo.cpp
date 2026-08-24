// Usage:
//   streaming_benchmark_demo <model_package_dir>

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

#include "engine/inference_engine.hpp"

namespace {

std::atomic_bool g_cancelled{false};

extern "C" void handle_sigint(int) { g_cancelled.store(true, std::memory_order_relaxed); }

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
        std::cerr << "usage: " << argv[0] << " <model_package_dir>" << std::endl;
        return 2;
    }

    std::signal(SIGINT, handle_sigint);
    const std::string package_root = argv[1];
    std::cout << "Loading model package from: " << package_root << std::endl;

    phono::engine::InferenceEngine engine(package_root);
    const auto& cfg = engine.config();
    const auto& tokenizer = engine.tokenizer();
    const int32_t model_batch = engine.pre_pass2_batch_size() > 0
                                    ? engine.pre_pass2_batch_size()
                                    : cfg.runtime.batch_size;
    nlohmann::json context_options = nlohmann::json::object();
    std::ifstream context_config(
        std::filesystem::path(package_root) / "core_configs/default.json");
    if (!context_config.is_open()) {
        context_config.open("core_configs/default.json");
    }
    if (context_config.is_open()) {
        context_config >> context_options;
    }
    context_options["beam_size"] = model_batch;
    context_options["max_context_length"] = std::min(
        context_options.value("max_context_length", cfg.pre_model.max_seqlen - 1),
        cfg.pre_model.max_seqlen - 1);
    phono::context::ContextManager context_manager(cfg, 1, context_options);
    phono::context::Context* slot = context_manager.get_context_auto({});
    phono::engine::InferenceSession session(engine, *slot);

    std::cout << "  context_vocab_size = " << tokenizer.context_vocab_size() << '\n'
              << "  pinyin_vocab_size  = " << tokenizer.pinyin_vocab_size() << '\n'
              << "  chinese_vocab_size = " << tokenizer.chinese_vocab_size() << '\n'
              << "  beam size          = " << session.beam_size() << '\n'
              << "  pre_model.max_seqlen = " << cfg.pre_model.max_seqlen << '\n'
              << "  post_model.max_seqlen = " << cfg.post_model.max_seqlen << '\n';
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
            session.generate(pinyin_ids, context_ids, &g_cancelled);
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
        const auto fill_error = session.fill(new_context_ids);
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
              << "  current_seqlen     : " << session.current_seqlen() << '\n'
              << "  history_seqlen     : " << session.history_seqlen() << '\n';
    if (!latencies_us.empty()) {
        const double total = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        std::cout << "  generation latency (us):\n"
                  << "    total            : " << total << '\n'
                  << "    average          : " << total / latencies_us.size() << '\n';
    }
    print_separator();
    return exit_code;
}
