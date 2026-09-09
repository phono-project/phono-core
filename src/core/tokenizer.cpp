#include "tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/config.hpp"
#include "core/text_normalizer.hpp"
#include "core/utf8_util.hpp"

namespace phono::core {

namespace {

// one token per line, a line consisting of exactly the two literal 
// characters `\n` (backslash + n) is converted to a real newline token, empty lines are skipped.
std::vector<std::string> read_vocab_tokens(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Tokenizer: cannot open vocab file: " + path);
    }
    std::vector<std::string> tokens;
    std::string line;
    while (std::getline(in, line)) {
        // strip trailing \r (Windows line endings), std::getline already
        // strips \n.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (line.size() == 2 && line[0] == '\\' && line[1] == 'n') {
            line = "\n";
        }
        if (line.empty()) {
            continue;
        }
        tokens.push_back(line);
    }
    return tokens;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& chinese_vocab_path, const std::string& context_vocab_path,
                      const std::string& pinyin_vocab_path,
                      const std::vector<std::string>& special_tokens_def) {
    // chinese_vocab (dedup)
    {
        auto chinese_chars = read_vocab_tokens(chinese_vocab_path);
        int32_t idx = 0;
        for (const auto& ch : chinese_chars) {
            if (chinese_vocab_.find(ch) == chinese_vocab_.end()) {
                chinese_vocab_.emplace(ch, idx);
                id_to_chinese_.emplace(idx, ch);
                ++idx;
            }
        }
    }

    // pinyin_vocab (dedup)
    {
        auto pinyin_tokens = read_vocab_tokens(pinyin_vocab_path);
        int32_t idx = 0;
        for (const auto& py : pinyin_tokens) {
            if (pinyin_vocab_.find(py) == pinyin_vocab_.end()) {
                pinyin_vocab_.emplace(py, idx);
                pinyin_list_.push_back(py);
                pinyin_tree_.insert(py);
                ++idx;
            }
        }
    }

    // context_vocab (own file, dedup) + special tokens
    {
        auto context_tokens = read_vocab_tokens(context_vocab_path);
        for (const auto& ctok : context_tokens) {
            if (context_vocab_.find(ctok) == context_vocab_.end()) {
                context_vocab_.emplace(ctok, static_cast<int32_t>(context_vocab_.size()));
            }
        }
        context_base_size_ = static_cast<int32_t>(context_vocab_.size());
        context_size_ = context_base_size_;

        for (size_t local_idx = 0; local_idx < special_tokens_def.size(); ++local_idx) {
            const std::string& name = special_tokens_def[local_idx];
            const int32_t global_id = context_base_size_ + static_cast<int32_t>(local_idx);
            context_vocab_.emplace(name, global_id);
            special_tokens_.emplace(name, global_id);
        }
        context_size_ += static_cast<int32_t>(special_tokens_def.size());
    }
}

Tokenizer Tokenizer::from_config(const ModelPackageConfig& cfg) {
    return Tokenizer(cfg.resolve(cfg.vocabs.chinese_vocab), cfg.resolve(cfg.vocabs.context_vocab),
                      cfg.resolve(cfg.vocabs.pinyin_vocab), cfg.vocabs.context_special_tokens);
}

int32_t Tokenizer::special_token_id(const std::string& name) const {
    return special_tokens_.at(name);
}

std::vector<int32_t> Tokenizer::encode_context(const std::string& text_utf8) const {
    std::vector<int32_t> ids;
    const std::string normalized = normalize_text_utf8(text_utf8);
    for (const auto& ch : utf8_split_chars(normalized)) {
        auto it = context_vocab_.find(ch);
        if (it != context_vocab_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

int Tokenizer::edit_distance(const std::string& a, const std::string& b) {
    const size_t m = a.size();
    const size_t n = b.size();
    if (m == 0) return static_cast<int>(n);
    if (n == 0) return static_cast<int>(m);

    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (size_t i = 0; i <= m; ++i) dp[i][0] = static_cast<int>(i);
    for (size_t j = 0; j <= n; ++j) dp[0][j] = static_cast<int>(j);

    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    return dp[m][n];
}

std::optional<int32_t> Tokenizer::find_pinyin_id_exact(std::string_view token) const {
    const auto it = pinyin_vocab_.find(std::string(token));
    if (it == pinyin_vocab_.end()) return std::nullopt;
    return it->second;
}

int32_t Tokenizer::find_pinyin_id_nearest(std::string_view token) const {
    if (pinyin_list_.empty()) {
        throw std::logic_error("Tokenizer: pinyin vocabulary is empty");
    }
    int32_t best_idx = 0;
    int best_dist = std::numeric_limits<int>::max();
    for (size_t i = 0; i < pinyin_list_.size(); ++i) {
        const int distance = edit_distance(std::string(token), pinyin_list_[i]);
        if (distance < best_dist) {
            best_dist = distance;
            best_idx = static_cast<int32_t>(i);
        }
    }
    return best_idx;
}

std::string Tokenizer::ids_to_text(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        auto it = id_to_chinese_.find(id);
        if (it != id_to_chinese_.end()) {
            out += it->second;
        }
    }
    return out;
}

std::string Tokenizer::id_to_chinese(int32_t id) const {
    auto it = id_to_chinese_.find(id);
    return it != id_to_chinese_.end() ? it->second : std::string();
}

int32_t Tokenizer::chinese_id_to_context_id(int32_t id) const {
    const std::string chinese = id_to_chinese(id);
    if (chinese.empty()) {
        return -1;
    }
    auto it = context_vocab_.find(chinese);
    return it != context_vocab_.end() ? it->second : -1;
}

}  // namespace phono::core
