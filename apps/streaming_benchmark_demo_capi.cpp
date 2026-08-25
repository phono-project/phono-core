// Usage:
//   streaming_benchmark_demo_capi <model_package_dir> [core_config_json]
//
// Identical to streaming_benchmark_demo but drives the engine exclusively
// through the stable C ABI (interface/phono_api.h). The core_config is handed
// to the library as a plain JSON string; the library parses and validates it
// internally.

#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

#include "phono_api.h"

namespace {

volatile std::sig_atomic_t g_cancelled = 0;

extern "C" void handle_sigint(int) { g_cancelled = 1; }

int cancellation_check(void* user_data) {
    (void)user_data;
    return g_cancelled != 0;
}

std::string load_core_config(const std::string& package_root,
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
    std::ostringstream buffer;
    if (in.is_open()) {
        buffer << in.rdbuf();
    } else {
        buffer << "{}";
    }
    return buffer.str();
}

enum class ReadStatus { Ok, Eof, Interrupted };

class LineReader {
public:
    ReadStatus next(std::string& out) {
        out.clear();
        for (;;) {
            if (g_cancelled != 0 && pending_.empty()) {
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
                if (g_cancelled != 0) return ReadStatus::Interrupted;
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

int exit_for_status(phono_status status) {
    return status == PHONO_OK ? 0 : static_cast<int>(status);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model_package_dir> [core_config_json]" << std::endl;
        return 2;
    }

    std::signal(SIGINT, handle_sigint);
    const std::string package_root = argv[1];
    const std::string core_config_path = argc > 2 ? argv[2] : std::string();
    const std::string core_config_json = load_core_config(package_root, core_config_path);
    std::cout << "Loading model package from: " << package_root << std::endl;

    phono_engine* engine = nullptr;
    phono_status status = phono_engine_create(package_root.c_str(), &engine);
    if (status != PHONO_OK) {
        std::cerr << "failed to load engine: " << phono_error_name(status) << " ("
                  << phono_last_error_message() << ")" << std::endl;
        return exit_for_status(status);
    }

    phono_context_manager* manager = nullptr;
    status = phono_context_manager_create(engine, core_config_json.c_str(), 1, &manager);
    if (status != PHONO_OK) {
        std::cerr << "failed to create context manager: " << phono_error_name(status) << " ("
                  << phono_last_error_message() << ")" << std::endl;
        phono_engine_destroy(engine);
        return exit_for_status(status);
    }

    phono_session* session = nullptr;
    status = phono_session_create(engine, core_config_json.c_str(), &session);
    if (status != PHONO_OK) {
        std::cerr << "failed to create session: " << phono_error_name(status) << " ("
                  << phono_last_error_message() << ")" << std::endl;
        phono_context_manager_destroy(manager);
        phono_engine_destroy(engine);
        return exit_for_status(status);
    }

    phono_context* slot = phono_context_manager_get_auto(manager, nullptr, 0);
    std::cout << "  beam size          = " << phono_session_beam_size(session) << '\n'
              << "  pre_model.max_seqlen = " << phono_engine_pre_max_seqlen(engine) << '\n'
              << "  post_model.max_seqlen = " << phono_engine_post_max_seqlen(engine) << '\n';
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

        std::vector<const char*> syllable_ptrs;
        syllable_ptrs.reserve(pinyin.size());
        for (const auto& syllable : pinyin) syllable_ptrs.push_back(syllable.c_str());
        int32_t* pinyin_ids = nullptr;
        int32_t pinyin_count = 0;
        status = phono_tokenizer_encode_pinyin(engine, syllable_ptrs.data(),
                                               static_cast<int32_t>(pinyin.size()),
                                               &pinyin_ids, &pinyin_count);
        if (status != PHONO_OK) {
            std::cout << "  ! pinyin encoding failed: " << phono_error_name(status) << '\n';
            exit_code = exit_for_status(status);
            break;
        }
        int32_t* context_ids = nullptr;
        int32_t context_count = 0;
        status = phono_tokenizer_encode_context(engine, committed_text.c_str(),
                                                &context_ids, &context_count);
        if (status != PHONO_OK) {
            std::cout << "  ! context encoding failed: " << phono_error_name(status) << '\n';
            exit_code = exit_for_status(status);
            break;
        }

        slot = phono_context_manager_get_auto(manager, context_ids, context_count);

        phono_generate_result result{};
        const auto start = std::chrono::high_resolution_clock::now();
        status = phono_session_generate(session, slot, pinyin_ids, pinyin_count,
                                        context_ids, context_count, cancellation_check, nullptr,
                                        &result);
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(end - start).count();
        latencies_us.push_back(elapsed_us);
        phono_free(pinyin_ids);
        phono_free(context_ids);

        if (status != PHONO_OK) {
            std::cout << "  ! generation failed: " << phono_error_name(status) << '\n';
            if (status != PHONO_CANCELLED) {
                exit_code = exit_for_status(status);
            }
            phono_generate_result_free(&result);
            break;
        }
        ++window_count;
        std::cout << "  context: \"" << committed_text << "\"\n"
                  << "  pinyin : [" << join(pinyin) << "]\n"
                  << "  candidates (" << std::fixed << std::setprecision(1) << elapsed_us
                  << " us):\n";
        for (int32_t i = 0; i < result.beam_count; ++i) {
            std::cout << "    [" << i + 1 << "] score=" << std::setprecision(4)
                      << result.beams[i].score << " \"" << result.beams[i].decoded << "\"\n";
        }
        if (result.beam_count == 0) {
            phono_generate_result_free(&result);
            continue;
        }

        std::cout << "select (1-" << result.beam_count << ", or ctrl-c to stop)> " << std::flush;
        std::string selection;
        if (reader.next(selection) != ReadStatus::Ok) {
            phono_generate_result_free(&result);
            break;
        }
        int choice = 0;
        try {
            choice = std::stoi(selection);
        } catch (...) {
            choice = 0;
        }
        if (choice < 1 || choice > result.beam_count) {
            std::cout << "  ! invalid choice\n";
            phono_generate_result_free(&result);
            continue;
        }

        const std::string chosen = result.beams[choice - 1].decoded;
        std::vector<int32_t> new_context_ids;
        for (int32_t i = 0; i < result.beams[choice - 1].pred_count; ++i) {
            const int32_t chinese_id = result.beams[choice - 1].pred_ids[i];
            const int32_t context_id = phono_tokenizer_chinese_to_context(engine, chinese_id);
            if (context_id < 0) {
                std::cout << "  ! selected candidate cannot be fed to context vocab\n";
                new_context_ids.clear();
                break;
            }
            new_context_ids.push_back(context_id);
        }
        phono_generate_result_free(&result);
        if (new_context_ids.empty()) continue;

        status = phono_session_fill(session, slot, new_context_ids.data(),
                                    static_cast<int32_t>(new_context_ids.size()));
        if (status != PHONO_OK) {
            std::cout << "  ! context fill failed: " << phono_error_name(status) << '\n';
            exit_code = exit_for_status(status);
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
              << "  history length     : " << phono_context_ids_len(slot) << '\n';
    if (!latencies_us.empty()) {
        const double total = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        std::cout << "  generation latency (us):\n"
                  << "    total            : " << total << '\n'
                  << "    average          : " << total / latencies_us.size() << '\n';
    }
    print_separator();

    phono_session_destroy(session);
    phono_context_manager_destroy(manager);
    phono_engine_destroy(engine);
    return exit_code;
}
