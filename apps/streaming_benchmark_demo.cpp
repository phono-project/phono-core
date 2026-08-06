// Usage:
//   streaming_benchmark_demo <model_package_dir>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "core/utf8_util.hpp"
#include "engine/inference_engine.hpp"

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

extern "C" void handle_sigint(int) { g_interrupted = 1; }

enum class ReadStatus { Ok, Eof, Interrupted };

// Minimal line reader.
class LineReader {
public:
    ReadStatus next(std::string& out) {
        out.clear();
        for (;;) {
            if (g_interrupted != 0 && pending_.empty()) {
                return ReadStatus::Interrupted;
            }
            const std::string::size_type nl = pending_.find('\n');
            if (nl != std::string::npos) {
                out = pending_.substr(0, nl);
                if (!out.empty() && out.back() == '\r') out.pop_back();
                pending_.erase(0, nl + 1);
                return ReadStatus::Ok;
            }
            char buf[4096];
            const ssize_t n = ::read(STDIN_FILENO, buf, sizeof(buf));
            if (n > 0) {
                pending_.append(buf, static_cast<size_t>(n));
            } else if (n == 0) {
                if (pending_.empty()) return ReadStatus::Eof;
                out = pending_;
                pending_.clear();
                return ReadStatus::Ok;
            } else if (errno == EINTR) {
                if (g_interrupted != 0) return ReadStatus::Interrupted;
                continue;
            } else {
                return ReadStatus::Eof;
            }
        }
    }

private:
    std::string pending_;
};

std::vector<std::string> split_whitespace(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string word;
    while (iss >> word) out.push_back(word);
    return out;
}

std::string join(const std::vector<std::string>& words, const std::string& sep = ", ") {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) out += sep;
        out += words[i];
    }
    return out;
}

size_t utf8_char_count(const std::string& text) { return phono::core::utf8_split_chars(text).size(); }

void print_separator() { std::cout << std::string(72, '-') << '\n'; }

struct BenchmarkStats {
    int window_count = 0;
    size_t pinyin_syllables = 0;
    size_t committed_chars = 0;
    size_t pending_tokens = 0;
    std::vector<double> step_latency_us;
};

enum class EndReason { Manual, ContextLimit };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model_package_dir>" << std::endl;
        return 2;
    }
    const std::string package_root = argv[1];

    std::signal(SIGINT, handle_sigint);

    std::cout << "Loading model package from: " << package_root << std::endl;
    phono::engine::InferenceEngine engine(package_root);
    const auto& cfg = engine.config();
    const auto& decoding_cfg = cfg.decoding;
    const auto& tok = engine.tokenizer();
    const int32_t pre_max = cfg.pre_model.max_seqlen;
    const int32_t post_max = cfg.post_model.max_seqlen;

    std::cout << "  context_vocab_size = " << tok.context_vocab_size() << std::endl;
    std::cout << "  pinyin_vocab_size  = " << tok.pinyin_vocab_size() << std::endl;
    std::cout << "  chinese_vocab_size = " << tok.chinese_vocab_size() << std::endl;
    std::cout << "  pre_model.max_seqlen = " << pre_max << "  (benchmark ends at this context bound)"
              << std::endl;
    std::cout << "  post_model.max_seqlen = " << post_max << "  (max pinyin syllables per window)"
              << std::endl;
    print_separator();
    std::cout << "Type a window of space-separated pinyin (e.g. 'ni hao'), then pick a"
              << "\n  candidate number to commit it to the context. Press Ctrl-C to stop."
              << std::endl;
    print_separator();

    phono::engine::InferenceSession session(engine);

    // One manager, several contexts; this run streams into context 0.
    phono::context::ContextManager context_manager(engine.config(), /*num_contexts=*/4);
    phono::context::Context& ctx = context_manager.get_context_by_id(0);

    LineReader reader;
    std::string committed_text;
    std::string unfed_text;

    BenchmarkStats stats;

    EndReason end_reason = EndReason::Manual;

    bool stop = false;
    while (!stop) {
        std::cout << "pinyin> " << std::flush;
        std::string line;
        const ReadStatus st = reader.next(line);
        if (st != ReadStatus::Ok) {
            end_reason = EndReason::Manual;
            break;
        }
        if (line.empty()) {
            std::cout << "  (empty input ignored)\n";
            continue;
        }

        std::vector<std::string> words = split_whitespace(line);
        if (words.empty()) continue;
        if (static_cast<int>(words.size()) > post_max) {
            std::cout << "  ! window has " << words.size() << " syllables, exceeds post-model "
                      << "max_seqlen " << post_max << "; skipping this window\n";
            continue;
        }

        std::cout << "  context: \"" << committed_text << "\"  (seqlen fed: "
                  << ctx.current_position() << ")\n";
        std::cout << "  pinyin : [" << join(words) << "]\n";

        const auto t0 = std::chrono::high_resolution_clock::now();
        phono::engine::ViterbiStepResult viterbi;
        try {
            // Advance the context with ONLY the text committed since the last
            // window (the delta), then decode the current pinyin window.
            viterbi = session.predict_step_viterbi(
                ctx, unfed_text, words, decoding_cfg.beta_single, decoding_cfg.beta_word,
                decoding_cfg.epsilon, decoding_cfg.n_best);
        } catch (const std::exception& e) {
            // A predict failure almost always means the pre-model context
            // reached the bound the deployed graph was actually exported with
            // (e.g. the post model's pre_max_seqlen) — which may be smaller
            // than config.json's pre_model.max_seqlen. End the benchmark
            // gracefully instead of aborting mid-run.
            std::cout << "  ! predict failed: " << e.what() << "\n";
            end_reason = EndReason::ContextLimit;
            stop = true;
            break;
        }
        unfed_text.clear();  // this delta is now part of the context
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double step_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        ++stats.window_count;
        stats.pinyin_syllables += words.size();
        stats.step_latency_us.push_back(step_us);

        std::cout << "  viterbi N=" << decoding_cfg.n_best << " took " << std::fixed
                  << std::setprecision(1) << step_us << " us\n";
        std::cout << "  candidates:\n";
        for (size_t rank = 0; rank < viterbi.nbest.size(); ++rank) {
            std::cout << "    [" << (rank + 1) << "] score=" << std::setprecision(4)
                      << viterbi.nbest[rank].score << "  text=\"" << viterbi.nbest[rank].text
                      << "\"\n";
        }
        if (viterbi.nbest.empty()) {
            std::cout << "    (no candidates above the epsilon threshold; type a new window)\n";
            continue;
        }

        size_t chosen = std::numeric_limits<size_t>::max();
        for (;;) {
            std::cout << "select (1-" << viterbi.nbest.size() << ", or ctrl-c to stop)> "
                      << std::flush;
            std::string sel;
            const ReadStatus st2 = reader.next(sel);
            if (st2 != ReadStatus::Ok) {
                end_reason = EndReason::Manual;
                stop = true;
                break;
            }
            if (sel.empty()) continue;

            int idx = -1;
            try {
                idx = std::stoi(sel);
            } catch (...) {
                idx = -1;
            }
            if (idx >= 1 && idx <= static_cast<int>(viterbi.nbest.size())) {
                chosen = static_cast<size_t>(idx - 1);
                break;
            }
            std::cout << "  ! invalid choice '" << sel << "', expected a number in 1.."
                      << viterbi.nbest.size() << "\n";
        }
        if (stop) break;

        const auto& chosen_entry = viterbi.nbest[chosen];
        const std::vector<int32_t> pending = tok.encode_context(chosen_entry.text);
        const int32_t projected = ctx.current_position() + static_cast<int32_t>(pending.size());

        if (projected > pre_max) {
            std::cout << "  ! candidate \"" << chosen_entry.text << "\" would push context to "
                      << projected << " > pre-model max_seqlen " << pre_max << "; stopping.\n";
            end_reason = EndReason::ContextLimit;
            break;
        }

        committed_text += chosen_entry.text;
        unfed_text += chosen_entry.text;  // handed to the next predict as the new delta
        stats.committed_chars += utf8_char_count(chosen_entry.text);
        // pending_tokens is the size of the delta still unfed at report time
        // (only the latest commit is unfed). Overwrite, don't accumulate.
        stats.pending_tokens = pending.size();
        std::cout << "  committed: \"" << chosen_entry.text << "\""
                  << "  (context now: \"" << committed_text << "\")\n";

        if (projected >= pre_max) {
            end_reason = EndReason::ContextLimit;
            std::cout << "  reached pre-model context limit (" << projected << "/" << pre_max
                      << "); stopping.\n";
            break;
        }
    }

    double wall_s = 0.0; for (const auto& t : stats.step_latency_us) wall_s += t * 1e-6;

    double total_us = 0.0, min_us = 0.0, max_us = 0.0, avg_us = 0.0;
    if (!stats.step_latency_us.empty()) {
        total_us = std::accumulate(stats.step_latency_us.begin(), stats.step_latency_us.end(), 0.0);
        min_us = *std::min_element(stats.step_latency_us.begin(), stats.step_latency_us.end());
        max_us = *std::max_element(stats.step_latency_us.begin(), stats.step_latency_us.end());
        avg_us = total_us / static_cast<double>(stats.step_latency_us.size());
    }

    const int32_t projected_seqlen =
        ctx.current_position() + static_cast<int32_t>(stats.pending_tokens);

    print_separator();
    std::cout << "Benchmark report\n";
    std::cout << "  end reason         : "
              << (end_reason == EndReason::ContextLimit
                      ? "reached pre-model context limit"
                      : "ctrl-c / manual stop (or stdin EOF)")
              << "\n";
    std::cout << "  wall time          : " << std::setprecision(3) << wall_s << " s\n";
    std::cout << "  windows            : " << stats.window_count << "\n";
    std::cout << "  pinyin syllables   : " << stats.pinyin_syllables << "\n";
    std::cout << "  committed chars    : " << stats.committed_chars << "\n";
    std::cout << "  committed text     : \"" << committed_text << "\"\n";
    std::cout << "  seqlen fed         : " << ctx.current_position() << "\n";
    std::cout << "  seqlen projected   : " << projected_seqlen << " / " << pre_max << "\n";
    std::cout << "  viterbi latency (us):\n";
    std::cout << "    steps            : " << stats.step_latency_us.size() << "\n";
    std::cout << "    total            : " << std::setprecision(1) << total_us << "\n";
    std::cout << "    min              : " << min_us << "\n";
    std::cout << "    avg              : " << avg_us << "\n";
    std::cout << "    max              : " << max_us << "\n";
    if (wall_s > 0.0) {
        std::cout << "  throughput         : " << std::setprecision(1)
                  << static_cast<double>(stats.committed_chars) / wall_s << " chars/s\n";
    }
    print_separator();

    return 0;
}
