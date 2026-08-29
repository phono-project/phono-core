// Usage:
//   ime_demo_capi <model_package_dir> [core_config_json]

#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
  #include <conio.h>
  #include <windows.h>
#else
  #include <termios.h>
  #include <unistd.h>
#endif

#include "apps/ime_editor.hpp"
#include "phono_api.h"

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

extern "C" void handle_sigint(int) { g_interrupted = 1; }

int cancellation_check(void*) { return g_interrupted != 0; }

std::string load_core_config(const std::string& package_root,
                             const std::string& override_path) {
    std::ifstream in;
    if (!override_path.empty()) in.open(override_path);
    if (!in.is_open()) {
        in.open(std::filesystem::path(package_root) / "core_configs/default.json");
    }
    if (!in.is_open()) in.open("core_configs/default.json");

    std::ostringstream buffer;
    if (in.is_open()) {
        buffer << in.rdbuf();
    } else {
        buffer << "{}";
    }
    return buffer.str();
}

enum class Key { Character, Backspace, Delete, Left, Right, End, Ignore };

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
            case 75: return {Key::Left};
            case 77: return {Key::Right};
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
    if (code == '3') {
        char tilde = '\0';
        if (read_byte(tilde) && tilde == '~') return {Key::Delete};
    }
    return {Key::Ignore};
}

#endif

std::string join_syllables(char* const* syllables, int32_t count) {
    std::string separated;
    for (int32_t i = 0; i < count; ++i) {
        if (i > 0) separated += '\'';
        separated += syllables[i];
    }
    return separated;
}

void render(const phono::apps::ImeEditor& editor, const std::string& separated,
            const std::vector<std::string>& candidates, const std::string& error) {
    std::cout << "\033[2J\033[H"
              << "pinyin> " << editor.text() << "\n"
              << "[0] " << separated << "\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        std::cout << '[' << i + 1 << "] " << candidates[i] << '\n';
    }
    if (!error.empty()) std::cout << "[!] " << error << '\n';
    std::cout << "\nLeft/Right: move  Backspace/Delete: erase  Ctrl-C: exit"
              << "\033[1;" << editor.cursor() + 9 << 'H' << std::flush;
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
        std::cout << "usage: " << argv[0] << " <model_package_dir> [core_config_json]\n";
        return 0;
    }
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model_package_dir> [core_config_json]\n";
        return 2;
    }

    std::signal(SIGINT, handle_sigint);
    const std::string package_root = argv[1];
    const std::string config_path = argc > 2 ? argv[2] : std::string();
    const std::string config_json = load_core_config(package_root, config_path);

    phono_engine* engine = nullptr;
    phono_status status = phono_engine_create(package_root.c_str(), &engine);
    if (status != PHONO_OK) {
        print_error("failed to load engine", status);
        return static_cast<int>(status);
    }

    phono_context_manager* manager = nullptr;
    status = phono_context_manager_create(engine, config_json.c_str(), 1, &manager);
    if (status != PHONO_OK) {
        print_error("failed to create context manager", status);
        phono_engine_destroy(engine);
        return static_cast<int>(status);
    }

    phono_session* session = nullptr;
    status = phono_session_create(engine, config_json.c_str(), &session);
    if (status != PHONO_OK) {
        print_error("failed to create session", status);
        phono_context_manager_destroy(manager);
        phono_engine_destroy(engine);
        return static_cast<int>(status);
    }
    phono_context* context = phono_context_manager_get_auto(manager, nullptr, 0);

    int exit_code = 0;
    {
        RawTerminal terminal;
        if (!terminal.valid()) {
            std::cerr << "ime_demo_capi requires an interactive terminal" << std::endl;
            exit_code = 2;
        } else {
            phono::apps::ImeEditor editor;
            std::string separated;
            std::vector<std::string> candidates;
            std::string error;
            render(editor, separated, candidates, error);

            while (g_interrupted == 0) {
                const KeyPress press = read_key();
                if (press.key == Key::End || g_interrupted != 0) break;

                bool changed = true;
                switch (press.key) {
                    case Key::Character:
                        if ((press.character >= 'a' && press.character <= 'z') ||
                            (press.character >= 'A' && press.character <= 'Z') ||
                            press.character == '\'') {
                            editor.insert(static_cast<char>(std::tolower(
                                static_cast<unsigned char>(press.character))));
                        } else {
                            changed = false;
                        }
                        break;
                    case Key::Backspace: editor.backspace(); break;
                    case Key::Delete: editor.erase(); break;
                    case Key::Left: editor.move_left(); break;
                    case Key::Right: editor.move_right(); break;
                    case Key::End:
                    case Key::Ignore: changed = false; break;
                }
                if (!changed) continue;

                separated.clear();
                candidates.clear();
                error.clear();
                if (!editor.text().empty()) {
                    char** syllables = nullptr;
                    int32_t syllable_count = 0;
                    status = phono_tokenizer_separate_greedy(
                        engine, editor.text().c_str(), &syllables, &syllable_count);
                    if (status == PHONO_OK) separated = join_syllables(syllables, syllable_count);

                    int32_t* pinyin_ids = nullptr;
                    int32_t pinyin_count = 0;
                    if (status == PHONO_OK) {
                        status = phono_tokenizer_encode_pinyin(
                            engine, syllables, syllable_count, &pinyin_ids, &pinyin_count);
                    }

                    phono_generate_result result{};
                    if (status == PHONO_OK) {
                        status = phono_session_generate(session, context, pinyin_ids, pinyin_count,
                                                        nullptr, 0, cancellation_check, nullptr,
                                                        &result);
                    }
                    if (status == PHONO_OK) {
                        for (int32_t i = 0; i < result.beam_count; ++i) {
                            candidates.emplace_back(result.beams[i].decoded == nullptr
                                                        ? ""
                                                        : result.beams[i].decoded);
                        }
                    } else if (status != PHONO_CANCELLED) {
                        error = phono_error_name(status);
                    }
                    phono_generate_result_free(&result);
                    phono_free(pinyin_ids);
                    phono_free(syllables);
                    if (status == PHONO_CANCELLED && g_interrupted != 0) break;
                }
                render(editor, separated, candidates, error);
            }
        }
    }

    std::cout << "\033[2J\033[H" << std::flush;
    phono_session_destroy(session);
    phono_context_manager_destroy(manager);
    phono_engine_destroy(engine);
    return exit_code;
}
