// Usage:
//   ime_demo_capi <model_package_dir> [engine_config_json]
//                 [context_manager_json] [num_contexts]

#include <cctype>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
  #include <conio.h>
  #include <windows.h>
#else
  #include <termios.h>
  #include <unistd.h>
#endif

#include "apps/ime_editor.hpp"
#include "apps/ime_ui.hpp"
#include "phono_api.h"

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

extern "C" void handle_sigint(int) { g_interrupted = 1; }

int cancellation_check(void*) { return g_interrupted != 0; }

std::string load_config(const std::string& filename,
                        const std::string& override_path) {
    std::ifstream in;
    if (!override_path.empty()) in.open(override_path);
    if (!in.is_open()) in.open(std::filesystem::path("core_configs") / filename);

    std::ostringstream buffer;
    if (in.is_open()) {
        buffer << in.rdbuf();
    } else {
        buffer << "";
    }
    return buffer.str();
}

enum class Key { Character, Backspace, Delete, Left, Right, Up, Down, End, Ignore };

struct KeyPress {
    Key key = Key::Ignore;
    char character = '\0';
};

#ifdef _WIN32

class RawTerminal {
public:
    RawTerminal() {
        output_ = GetStdHandle(STD_OUTPUT_HANDLE);
        if (output_ != INVALID_HANDLE_VALUE && GetConsoleMode(output_, &output_mode_)) {
            SetConsoleMode(output_, output_mode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            restore_output_ = true;
        }
    }

    ~RawTerminal() {
        if (restore_output_) SetConsoleMode(output_, output_mode_);
    }

    bool valid() const { return true; }

private:
    HANDLE output_ = INVALID_HANDLE_VALUE;
    DWORD output_mode_ = 0;
    bool restore_output_ = false;
};

KeyPress read_key() {
    const int ch = _getch();
    if (ch == 3) return {Key::End};
    if (ch == 0 || ch == 224) {
        switch (_getch()) {
            case 72: return {Key::Up};
            case 75: return {Key::Left};
            case 77: return {Key::Right};
            case 80: return {Key::Down};
            case 83: return {Key::Delete};
            default: return {Key::Ignore};
        }
    }
    if (ch == 8) return {Key::Backspace};
    return {Key::Character, static_cast<char>(ch)};
}

#else

class RawTerminal {
public:
    RawTerminal() {
        if (tcgetattr(STDIN_FILENO, &original_) != 0) return;
        termios raw = original_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_iflag &= static_cast<tcflag_t>(~(IXON | ICRNL));
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        active_ = tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0;
    }

    ~RawTerminal() {
        if (active_) tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_);
    }

    bool valid() const { return active_; }

private:
    termios original_{};
    bool active_ = false;
};

bool read_byte(char& out) {
    for (;;) {
        const ssize_t count = ::read(STDIN_FILENO, &out, 1);
        if (count == 1) return true;
        if (count == 0) return false;
        if (errno != EINTR || g_interrupted != 0) return false;
    }
}

KeyPress read_key() {
    char ch = '\0';
    if (!read_byte(ch)) return {Key::End};
    if (ch == 127 || ch == 8) return {Key::Backspace};
    if (ch != 27) return {Key::Character, ch};

    char bracket = '\0';
    char code = '\0';
    if (!read_byte(bracket) || bracket != '[' || !read_byte(code)) return {Key::Ignore};
    if (code == 'D') return {Key::Left};
    if (code == 'C') return {Key::Right};
    if (code == 'A') return {Key::Up};
    if (code == 'B') return {Key::Down};
    if (code == '3') {
        char tilde = '\0';
        if (read_byte(tilde) && tilde == '~') return {Key::Delete};
    }
    return {Key::Ignore};
}

#endif

std::string join_syllables(const std::vector<std::string>& syllables) {
    std::string separated;
    for (size_t i = 0; i < syllables.size(); ++i) {
        if (i > 0) separated += '\'';
        separated += syllables[i];
    }
    return separated;
}

std::string with_cursor(const phono::apps::ImeEditor& editor) {
    return editor.text().substr(0, editor.cursor()) + '|' + editor.text().substr(editor.cursor());
}

void render(const phono::apps::ImeEditor& pinyin, const std::string& separated,
            const std::string& engine_status,
            const phono::apps::ImeEditor& history,
            size_t history_index, size_t history_count,
            const std::vector<std::string>& candidates, const std::string& error,
            const std::optional<double>& latency_ms) {
    const bool editing_history = pinyin.text().empty();
    std::cout << "\033[2J\033[H"
              << "pinyin> " << (editing_history ? pinyin.text() : with_cursor(pinyin)) << "\n"
              << "engine: " << engine_status << "\n"
              << "seg: " << separated << "\n"
              << "history " << history_index + 1 << '/' << history_count << ": "
              << (editing_history ? with_cursor(history) : history.text()) << "\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::cout << i + 1 << '.' << candidates[i] << '\n';
    }
    if (!error.empty()) std::cout << "error: " << error << '\n';
    if (latency_ms.has_value()) {
        const char* color = "\033[31m";
        switch (phono::apps::latency_level(*latency_ms)) {
            case phono::apps::LatencyLevel::Green: color = "\033[32m"; break;
            case phono::apps::LatencyLevel::Yellow: color = "\033[33m"; break;
            case phono::apps::LatencyLevel::Red: break;
        }
        std::cout << color << "Latency (ms): " << std::fixed << std::setprecision(1)
                  << *latency_ms << "\033[0m\n";
    } else {
        std::cout << "Latency (ms): --\n";
    }
    std::cout << "\nLeft/Right: move  Up/Down: switch history"
              << "  Backspace/Delete: erase  Ctrl-C: exit" << std::flush;
}

void print_error(const char* operation, phono_status status) {
    std::cerr << operation << ": " << phono_error_name(status);
    const char* detail = phono_last_error_message();
    if (detail != nullptr && *detail != '\0') std::cerr << " (" << detail << ')';
    std::cerr << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "usage: " << argv[0]
                  << " <model_package_dir> [engine_config_json]"
                     " [context_manager_json] [num_contexts]\n";
        return 0;
    }
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " <model_package_dir> [engine_config_json]"
                     " [context_manager_json] [num_contexts]\n";
        return 2;
    }

    std::signal(SIGINT, handle_sigint);
    const std::string package_root = argv[1];
    const std::string engine_config_json = load_config(
        "engine_config.json", argc > 2 ? argv[2] : std::string());
    const std::string context_config_json = load_config(
        "context_manager.json", argc > 3 ? argv[3] : std::string());
    int32_t num_contexts = 2;
    if (argc > 4) {
        try {
            size_t parsed = 0;
            num_contexts = std::stoi(argv[4], &parsed);
            if (parsed != std::string(argv[4]).size() || num_contexts <= 0) {
                throw std::invalid_argument("invalid num_contexts");
            }
        } catch (...) {
            std::cerr << "num_contexts must be a positive integer\n";
            return 2;
        }
    }

    phono_engine* engine = nullptr;
    phono_status status = phono_engine_create(
        package_root.c_str(), engine_config_json.c_str(), &engine);
    if (status != PHONO_OK) {
        print_error("failed to load engine", status);
        return static_cast<int>(status);
    }

    phono_context_manager* manager = nullptr;
    char* info_text = phono_engine_info_json(engine);
    std::string engine_status = "unknown";
    if (info_text != nullptr) {
        const nlohmann::json info = nlohmann::json::parse(info_text);
        engine_status = info["segmenter"]["available"].get<bool>()
            ? "smart scorer + Trie DAG" : "checked FMM fallback";
        if (!info["diagnostics"].empty()) {
            engine_status += " (" + info["diagnostics"].front().get<std::string>() + ")";
        }
        phono_free(info_text);
    }

    status = phono_context_manager_create(
        engine, context_config_json.c_str(), num_contexts, &manager);
    if (status != PHONO_OK) {
        print_error("failed to create context manager", status);
        phono_engine_destroy(engine);
        return static_cast<int>(status);
    }

    phono_session* session = nullptr;
    status = phono_session_create(engine, context_config_json.c_str(), &session);
    if (status != PHONO_OK) {
        print_error("failed to create session", status);
        phono_context_manager_destroy(manager);
        phono_engine_destroy(engine);
        return static_cast<int>(status);
    }
    phono_context* context = nullptr;

    int exit_code = 0;
    {
        RawTerminal terminal;
        if (!terminal.valid()) {
            std::cerr << "ime_demo_capi requires an interactive terminal" << std::endl;
            exit_code = 2;
        } else {
            phono::apps::ImeEditor pinyin;
            std::string separated;
            std::vector<phono::apps::ImeEditor> histories(static_cast<size_t>(num_contexts));
            size_t history_index = 0;
            std::vector<std::string> candidates;
            std::string error;
            std::optional<double> latency_ms;
            render(pinyin, separated, engine_status, histories[history_index], history_index, histories.size(),
                   candidates, error, latency_ms);

            while (g_interrupted == 0) {
                const KeyPress press = read_key();
                if (press.key == Key::End || g_interrupted != 0) break;

                if (pinyin.text().empty() && (press.key == Key::Up || press.key == Key::Down)) {
                    if (press.key == Key::Up) {
                        history_index = phono::apps::previous_history(
                            history_index, histories.size());
                    } else {
                        history_index = phono::apps::next_history(history_index, histories.size());
                    }
                    separated.clear();
                    candidates.clear();
                    error.clear();
                    render(pinyin, separated, engine_status, histories[history_index], history_index,
                           histories.size(), candidates, error, latency_ms);
                    continue;
                }

                phono::apps::ImeEditor& history = histories[history_index];

                bool changed = true;
                switch (press.key) {
                    case Key::Character:
                        if (press.character >= '1' && press.character <= '9' &&
                            static_cast<size_t>(press.character - '1') < candidates.size()) {
                            history.insert(candidates[static_cast<size_t>(press.character - '1')]);
                            pinyin.clear();
                            separated.clear();
                            candidates.clear();
                            error.clear();
                            render(pinyin, separated, engine_status, history, history_index, histories.size(),
                                   candidates, error, latency_ms);
                            continue;
                        } else if ((press.character >= 'a' && press.character <= 'z') ||
                            (press.character >= 'A' && press.character <= 'Z') ||
                            press.character == '\'') {
                            pinyin.insert(static_cast<char>(std::tolower(
                                static_cast<unsigned char>(press.character))));
                        } else {
                            changed = false;
                        }
                        break;
                    case Key::Backspace:
                        if (pinyin.text().empty()) history.backspace();
                        else pinyin.backspace();
                        break;
                    case Key::Delete:
                        if (pinyin.text().empty()) history.erase();
                        else pinyin.erase();
                        break;
                    case Key::Left:
                        if (pinyin.text().empty()) history.move_left();
                        else pinyin.move_left();
                        break;
                    case Key::Right:
                        if (pinyin.text().empty()) history.move_right();
                        else pinyin.move_right();
                        break;
                    case Key::Up:
                    case Key::Down:
                    case Key::End:
                    case Key::Ignore: changed = false; break;
                }
                if (!changed) continue;

                separated.clear();
                candidates.clear();
                error.clear();
                if (!pinyin.text().empty()) {
                    const std::string request = nlohmann::json{
                        {"schema_version", "1.0"}, {"input", pinyin.text()}}.dump();
                    char* segmentation_text = nullptr;
                    status = phono_engine_segment_pinyin(
                        engine, request.c_str(), &segmentation_text);
                    std::vector<int32_t> pinyin_ids;
                    if (segmentation_text != nullptr) {
                        const nlohmann::json segmentation =
                            nlohmann::json::parse(segmentation_text);
                        phono_free(segmentation_text);
                        const auto syllables =
                            segmentation["segments"].get<std::vector<std::string>>();
                        separated = "[" + segmentation["strategy"].get<std::string>() + "] " +
                                    join_syllables(syllables);
                        if (!segmentation["invalid_ranges"].empty()) {
                            error = "invalid_ranges=" +
                                    segmentation["invalid_ranges"].dump();
                        }
                        pinyin_ids =
                            segmentation["pinyin_ids"].get<std::vector<int32_t>>();
                    }

                    int32_t* context_ids = nullptr;
                    int32_t context_count = 0;
                    if (status == PHONO_OK) {
                        const std::string model_history(history.text_before_cursor());
                        status = phono_tokenizer_encode_context(
                            engine, model_history.c_str(), &context_ids, &context_count);
                    }
                    if (status == PHONO_OK) {
                        context = phono_context_manager_get_auto(
                            manager, context_ids, context_count);
                        if (context == nullptr) status = PHONO_MODEL_ERROR;
                    }

                    phono_generate_result result{};
                    if (status == PHONO_OK) {
                        const auto start = std::chrono::steady_clock::now();
                        status = phono_session_generate(
                                                        session, context, pinyin_ids.data(),
                                                        static_cast<int32_t>(pinyin_ids.size()),
                                                        context_ids, context_count,
                                                        cancellation_check, nullptr, &result);
                        latency_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - start)
                                         .count();
                    }
                    if (status == PHONO_OK) {
                        for (int32_t i = 0; i < result.beam_count; ++i) {
                            candidates.emplace_back(result.beams[i].decoded == nullptr
                                                        ? ""
                                                        : result.beams[i].decoded);
                        }
                    } else if (status != PHONO_CANCELLED) {
                        if (error.empty()) error = phono_error_name(status);
                    }
                    phono_generate_result_free(&result);
                    phono_free(context_ids);
                    if (status == PHONO_CANCELLED && g_interrupted != 0) break;
                }
                render(pinyin, separated, engine_status, history, history_index, histories.size(), candidates,
                       error, latency_ms);
            }
        }
    }

    std::cout << "\033[2J\033[H" << std::flush;
    phono_session_destroy(session);
    phono_context_manager_destroy(manager);
    phono_engine_destroy(engine);
    return exit_code;
}
