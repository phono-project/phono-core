#include <cstdlib>
#include <iostream>
#include <string>

#include "apps/ime_editor.hpp"

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    phono::apps::ImeEditor editor;
    editor.insert('a');
    editor.insert('b');
    editor.insert('c');
    check(editor.text() == "abc" && editor.cursor() == 3, "insert should advance cursor");

    editor.move_left();
    editor.move_left();
    editor.insert('x');
    check(editor.text() == "axbc" && editor.cursor() == 2,
          "insert should occur at cursor");

    editor.backspace();
    check(editor.text() == "abc" && editor.cursor() == 1,
          "backspace should erase before cursor");
    editor.erase();
    check(editor.text() == "ac" && editor.cursor() == 1,
          "delete should erase at cursor");

    editor.move_left();
    editor.move_left();
    editor.backspace();
    check(editor.text() == "ac" && editor.cursor() == 0,
          "left and backspace should stop at beginning");
    editor.move_right();
    editor.move_right();
    editor.move_right();
    editor.erase();
    check(editor.text() == "ac" && editor.cursor() == 2,
          "right and delete should stop at end");
    return 0;
}
