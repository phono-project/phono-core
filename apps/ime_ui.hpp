#pragma once

#include <cstddef>

namespace phono::apps {

enum class LatencyLevel { Green, Yellow, Red };

inline LatencyLevel latency_level(double milliseconds) {
    if (milliseconds <= 100.0) return LatencyLevel::Green;
    if (milliseconds <= 300.0) return LatencyLevel::Yellow;
    return LatencyLevel::Red;
}

inline size_t previous_history(size_t current, size_t count) {
    return (current + count - 1) % count;
}

inline size_t next_history(size_t current, size_t count) {
    return (current + 1) % count;
}

}  // namespace phono::apps
