#include <cstdlib>
#include <iostream>
#include <string>

#include "apps/ime_editor.hpp"
#include "apps/ime_ui.hpp"

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
    editor.clear();
    check(editor.text().empty() && editor.cursor() == 0,
          "clear should reset text and cursor");

    const std::string ni = "\xE4\xBD\xA0";
    const std::string hao = "\xE5\xA5\xBD";
    editor.insert(ni + hao);
    editor.move_left();
    check(editor.cursor() == ni.size(), "left should cross one UTF-8 codepoint");
    editor.backspace();
    check(editor.text() == hao && editor.cursor() == 0,
          "backspace should erase one UTF-8 codepoint");
    editor.erase();
    check(editor.text().empty(), "delete should erase one UTF-8 codepoint");

    using phono::apps::LatencyLevel;
    check(phono::apps::latency_level(100.0) == LatencyLevel::Green,
          "100 ms should be green");
    check(phono::apps::latency_level(100.1) == LatencyLevel::Yellow,
          "latency above 100 ms should be yellow");
    check(phono::apps::latency_level(300.0) == LatencyLevel::Yellow,
          "300 ms should be yellow");
    check(phono::apps::latency_level(300.1) == LatencyLevel::Red,
          "latency above 300 ms should be red");
    return 0;
}
