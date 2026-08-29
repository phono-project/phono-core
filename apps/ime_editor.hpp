#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace phono::apps {

class ImeEditor {
public:
    const std::string& text() const { return text_; }
    size_t cursor() const { return cursor_; }
    std::string_view text_before_cursor() const {
        return std::string_view(text_).substr(0, cursor_);
    }

    void clear() {
        text_.clear();
        cursor_ = 0;
    }

    void insert(char ch) {
        text_.insert(text_.begin() + static_cast<std::string::difference_type>(cursor_), ch);
        ++cursor_;
    }

    void insert(std::string_view text) {
        text_.insert(cursor_, text);
        cursor_ += text.size();
    }

    void backspace() {
        if (cursor_ == 0) return;
        const size_t previous = previous_boundary(cursor_);
        text_.erase(previous, cursor_ - previous);
        cursor_ = previous;
    }

    void erase() {
        if (cursor_ < text_.size()) text_.erase(cursor_, next_boundary(cursor_) - cursor_);
    }

    void move_left() {
        if (cursor_ > 0) cursor_ = previous_boundary(cursor_);
    }

    void move_right() {
        if (cursor_ < text_.size()) cursor_ = next_boundary(cursor_);
    }

private:
    static bool is_continuation(char ch) {
        return (static_cast<unsigned char>(ch) & 0xC0U) == 0x80U;
    }

    size_t previous_boundary(size_t position) const {
        --position;
        while (position > 0 && is_continuation(text_[position])) --position;
        return position;
    }

    size_t next_boundary(size_t position) const {
        ++position;
        while (position < text_.size() && is_continuation(text_[position])) ++position;
        return position;
    }

    std::string text_;
    size_t cursor_ = 0;
};

}  // namespace phono::apps
