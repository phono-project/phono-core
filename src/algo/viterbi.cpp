#include "algo/viterbi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace phono::algo {

namespace {

// Linked-list path nodes.
struct PathNode {
    std::string piece;
    std::shared_ptr<PathNode> prev;
};

struct BeamEntry {
    double score;
    std::shared_ptr<PathNode> node;
};

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

}  // namespace

std::vector<ViterbiResult> viterbi_nbest(const std::vector<std::unordered_map<std::string, double>>& candidates,
                                          const std::vector<std::vector<WordMatch>>& words_at, double beta_single,
                                          double beta_word, int N) {
    const int L = static_cast<int>(candidates.size());
    if (L == 0) {
        return {ViterbiResult{0.0, {}}};
    }

    const double ln_beta_single = beta_single > 0.0 ? std::log(beta_single) : kNegInf;
    const double ln_beta_word = beta_word > 0.0 ? std::log(beta_word) : kNegInf;

    std::vector<std::unordered_map<std::string, double>> candidates_log(L);
    for (int i = 0; i < L; ++i) {
        for (const auto& [ch, prob] : candidates[i]) {
            candidates_log[i][ch] = (prob > 0.0) ? std::log(prob) : kNegInf;
        }
    }

    // paths[idx]: prefix-text -> best (score, path-node) reaching position idx.
    // Using the concatenated text as the dedup key.
    std::vector<std::unordered_map<std::string, BeamEntry>> paths(L + 1);
    paths[0][""] = BeamEntry{0.0, nullptr};

    for (int start = 0; start < L; ++start) {
        if (paths[start].empty()) {
            continue;
        }

        // Prune beam at this position to the top-N unique text prefixes.
        if (static_cast<int>(paths[start].size()) > N) {
            std::vector<std::pair<std::string, BeamEntry>> items(paths[start].begin(), paths[start].end());
            std::partial_sort(items.begin(), items.begin() + N, items.end(),
                               [](const auto& a, const auto& b) { return a.second.score > b.second.score; });
            items.resize(N);
            paths[start].clear();
            for (auto& kv : items) {
                paths[start].emplace(std::move(kv.first), std::move(kv.second));
            }
        }

        const auto& cand_log = candidates_log[start];

        for (const auto& [prefix_str, entry] : paths[start]) {
            // Single-character transitions.
            for (const auto& [ch, log_prob] : cand_log) {
                const double new_score = entry.score + log_prob + ln_beta_single;
                const int next_idx = start + 1;
                std::string new_prefix = prefix_str + ch;

                auto node = std::make_shared<PathNode>(PathNode{ch, entry.node});
                auto& next_map = paths[next_idx];
                auto it = next_map.find(new_prefix);
                if (it == next_map.end() || new_score > it->second.score) {
                    next_map[new_prefix] = BeamEntry{new_score, node};
                }
            }

            // Dictionary-word transitions.
            for (const auto& wm : words_at[start]) {
                const int next_idx = start + wm.length;
                if (next_idx > L) {
                    continue;
                }
                const double new_score = entry.score + wm.log_prob + ln_beta_word;
                std::string new_prefix = prefix_str + wm.word;

                auto node = std::make_shared<PathNode>(PathNode{wm.word, entry.node});
                auto& next_map = paths[next_idx];
                auto it = next_map.find(new_prefix);
                if (it == next_map.end() || new_score > it->second.score) {
                    next_map[new_prefix] = BeamEntry{new_score, node};
                }
            }
        }
    }

    std::vector<ViterbiResult> results;
    if (!paths[L].empty()) {
        std::vector<BeamEntry> finals;
        finals.reserve(paths[L].size());
        for (auto& [prefix, entry] : paths[L]) {
            finals.push_back(entry);
        }
        const int keep = std::min(N, static_cast<int>(finals.size()));
        std::partial_sort(finals.begin(), finals.begin() + keep, finals.end(),
                           [](const BeamEntry& a, const BeamEntry& b) { return a.score > b.score; });
        finals.resize(keep);

        for (const auto& entry : finals) {
            std::vector<std::string> words;
            for (auto node = entry.node; node != nullptr; node = node->prev) {
                words.push_back(node->piece);
            }
            std::reverse(words.begin(), words.end());
            results.push_back(ViterbiResult{entry.score, std::move(words)});
        }
    }

    return results;
}

}  // namespace phono::algo
