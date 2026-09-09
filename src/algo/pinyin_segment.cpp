#include "pinyin_segment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace phono::algo {
namespace {

size_t next_forced_end(size_t begin, size_t input_size,
                       const std::vector<bool>& forced_boundaries) {
    for (size_t gap = begin; gap < forced_boundaries.size(); ++gap) {
        if (forced_boundaries[gap]) return gap + 1;
    }
    return input_size;
}

bool prefer(const SegmentationPath& candidate, const SegmentationPath& current) {
    if (!current.reachable) return true;
    if (candidate.invalid_char_count != current.invalid_char_count) {
        return candidate.invalid_char_count < current.invalid_char_count;
    }
    if (candidate.gap_score != current.gap_score) {
        return candidate.gap_score > current.gap_score;
    }
    if (candidate.edges.size() != current.edges.size()) {
        return candidate.edges.size() < current.edges.size();
    }
    return std::lexicographical_compare(
        current.edges.begin(), current.edges.end(), candidate.edges.begin(), candidate.edges.end(),
        [](const PinyinEdge& lhs, const PinyinEdge& rhs) {
            return (lhs.end - lhs.begin) < (rhs.end - rhs.begin);
        });
}

void append_candidate(std::vector<SegmentationPath>& best, size_t begin, size_t end,
                      bool invalid, const std::vector<float>& gap_logits) {
    if (!best[begin].reachable) return;
    SegmentationPath candidate = best[begin];
    candidate.edges.push_back(PinyinEdge{begin, end, invalid});
    candidate.invalid_char_count += invalid ? static_cast<int32_t>(end - begin) : 0;
    if (end < best.size() - 1) candidate.gap_score += gap_logits[end - 1];
    candidate.reachable = true;
    if (prefer(candidate, best[end])) best[end] = std::move(candidate);
}

}  // namespace

SegmentationPath decode_gap_viterbi(const std::string& input, const Trie& trie,
                                    const std::vector<bool>& forced_boundaries,
                                    const std::vector<float>& gap_logits,
                                    bool allow_invalid) {
    if (forced_boundaries.size() != (input.empty() ? 0 : input.size() - 1)) {
        throw std::invalid_argument("forced_boundaries must have input.size() - 1 entries");
    }
    if (gap_logits.size() != forced_boundaries.size()) {
        throw std::invalid_argument("gap_logits must have input.size() - 1 entries");
    }
    if (input.empty()) return {};

    std::vector<SegmentationPath> best(input.size() + 1);
    best[0].reachable = true;
    for (size_t begin = 0; begin < input.size(); ++begin) {
        if (!best[begin].reachable) continue;
        const size_t end_limit = next_forced_end(begin, input.size(), forced_boundaries);
        for (const size_t length : trie.match_lengths(input, begin, end_limit)) {
            append_candidate(best, begin, begin + length, false, gap_logits);
        }
        if (allow_invalid) append_candidate(best, begin, begin + 1, true, gap_logits);
    }
    return best.back();
}

SegmentationPath separate_fmm_checked(const std::string& input, const Trie& trie,
                                      const std::vector<bool>& forced_boundaries) {
    if (forced_boundaries.size() != (input.empty() ? 0 : input.size() - 1)) {
        throw std::invalid_argument("forced_boundaries must have input.size() - 1 entries");
    }
    // A purely local longest match can enter a dead end even when a complete
    // legal route exists. Zero-score DAG decoding preserves FMM's longest
    // deterministic tie-break while globally minimizing invalid characters.
    return decode_gap_viterbi(
        input, trie, forced_boundaries,
        std::vector<float>(forced_boundaries.size(), 0.0f), true);
}

}  // namespace phono::algo
