#pragma once

#include <cstddef>
#include <string>

namespace phono::apps {

class ImeEditor {
public:
    const std::string& text() const { return text_; }
    size_t cursor() const { return cursor_; }

    void insert(char ch) {
        text_.insert(text_.begin() + static_cast<std::string::difference_type>(cursor_), ch);
        ++cursor_;
    }

    void backspace() {
        if (cursor_ == 0) return;
        text_.erase(cursor_ - 1, 1);
        --cursor_;
    }

    void erase() {
        if (cursor_ < text_.size()) text_.erase(cursor_, 1);
    }

    void move_left() {
        if (cursor_ > 0) --cursor_;
    }

    void move_right() {
        if (cursor_ < text_.size()) ++cursor_;
    }

private:
    std::string text_;
    size_t cursor_ = 0;
};

}  // namespace phono::apps
