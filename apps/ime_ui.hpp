#pragma once

namespace phono::apps {

enum class LatencyLevel { Green, Yellow, Red };

inline LatencyLevel latency_level(double milliseconds) {
    if (milliseconds <= 100.0) return LatencyLevel::Green;
    if (milliseconds <= 300.0) return LatencyLevel::Yellow;
    return LatencyLevel::Red;
}

}  // namespace phono::apps
