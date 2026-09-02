// Usage:
//   performance_benchmark <model_package_dir> [iterations] [warmups]
//
// A non-interactive, CSV-producing benchmark for the model and runtime hot
// paths. Setup work is deliberately kept outside each timed sample so model
// execution, generation and incremental fill can be compared independently.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include "context/context.hpp"
#include "context/kv_cache.hpp"
#include "core/config.hpp"
#include "engine/inference_engine.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using phono::engine::InferenceError;

struct Options {
    std::string package_root;
    int32_t iterations = 20;
    int32_t warmups = 5;
};

struct Statistics {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p90_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

Options parse_options(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        throw std::invalid_argument(
            "usage: performance_benchmark <model_package_dir> [iterations] [warmups]");
    }
    Options options;
    options.package_root = argv[1];
    if (argc >= 3) options.iterations = std::stoi(argv[2]);
    if (argc >= 4) options.warmups = std::stoi(argv[3]);
    if (options.iterations <= 0 || options.warmups < 0) {
        throw std::invalid_argument("iterations must be positive and warmups non-negative");
    }
    return options;
}

Statistics summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Statistics stats;
    stats.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    stats.p50_us = samples[samples.size() / 2];
    stats.p90_us = samples[std::min(samples.size() - 1,
                                    static_cast<size_t>(samples.size() * 0.9))];
    stats.min_us = samples.front();
    stats.max_us = samples.back();
    return stats;
}

template <typename Setup, typename Body>
Statistics measure(int32_t warmups, int32_t iterations, Setup&& setup, Body&& body) {
    for (int32_t i = 0; i < warmups; ++i) {
        setup();
        body();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    for (int32_t i = 0; i < iterations; ++i) {
        setup();
        const auto begin = Clock::now();
        body();
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    return summarize(std::move(samples));
}

void print_result(const std::string& benchmark, int32_t history_length,
                  int32_t sequence_length, int32_t samples, const Statistics& stats) {
    std::cout << benchmark << ',' << history_length << ',' << sequence_length << ',' << samples
              << ',' << std::fixed << std::setprecision(3) << stats.mean_us << ','
              << stats.p50_us << ',' << stats.p90_us << ',' << stats.min_us << ','
              << stats.max_us << '\n';
}

void require_ok(InferenceError error, const char* operation) {
    if (error != InferenceError::Ok) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 phono::engine::inference_error_name(error));
    }
}

std::vector<int32_t> pinyin_prefix(const phono::core::Tokenizer& tokenizer, int32_t length) {
    static const std::vector<std::string> syllables = {
        "ni", "hao", "shi", "jie", "wo", "men", "yi", "qi",
        "qu", "chi", "fan", "ba", "ran", "hou", "hui", "jia",
        "kan", "dian", "ying", "de", "tian", "qi", "hen", "hao",
        "wo", "men", "chu", "fa", "xue", "xi", "zhong", "wen",
    };
    return tokenizer.encode_pinyin(
        std::vector<std::string>(syllables.begin(), syllables.begin() + length));
}

int64_t peak_rss_kib() {
#ifdef _WIN32
    return -1;
#else
    struct rusage usage {};
    return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
#endif
}

void run(const Options& options) {
    const auto load_begin = Clock::now();
    auto engine = std::make_unique<phono::engine::InferenceEngine>(options.package_root);
    const auto load_end = Clock::now();
    const double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_begin).count();

    const auto& cfg = engine->config();
    const auto& tokenizer = engine->tokenizer();
    const phono::core::CoreConfig core = phono::core::default_core_config(cfg);
    const int32_t context_id = tokenizer.encode_context("你").front();

    std::cout << "# model_version=" << cfg.model_version << '\n'
              << "# load_ms=" << std::fixed << std::setprecision(3) << load_ms << '\n'
              << "# peak_rss_kib_after_load=" << peak_rss_kib() << '\n'
              << "# pre_pass1_batch=" << engine->pre_pass1_batch_size() << '\n'
              << "# pre_pass2_batch=" << engine->pre_pass2_batch_size() << '\n'
              << "# beam_size=" << core.beam_size << '\n'
              << "benchmark,history_length,sequence_length,samples,mean_us,p50_us,p90_us,min_us,max_us\n";

    const std::vector<int32_t> sequence_lengths = {1, 2, 4, 8, 16, 32};
    for (int32_t length : sequence_lengths) {
        const std::vector<int32_t> pinyin = pinyin_prefix(tokenizer, length);
        phono::engine::PostModelOutput output;
        const Statistics stats = measure(
            options.warmups, options.iterations, []() {}, [&]() {
                require_ok(engine->run_post_model(pinyin, output), "post_model");
            });
        print_result("post_model", 0, length, options.iterations, stats);
    }

    const int32_t pre1_batch = engine->pre_pass1_batch_size();
    if (pre1_batch <= 0) throw std::runtime_error("pre pass1 has no fixed representative batch");
    phono::context::PersistentTensor pre1_cache = phono::context::make_zero_persistent_tensor(
        {cfg.pre_model.mhsa_layers, 2, pre1_batch, cfg.pre_model.max_seqlen,
         cfg.pre_model.mhsa_heads, cfg.pre_model.self_head_dim()});
    for (int32_t length : std::vector<int32_t>{1, 2, 8, 32, 64}) {
        std::vector<int32_t> input(static_cast<size_t>(pre1_batch) * length, context_id);
        const Statistics stats = measure(
            options.warmups, options.iterations, [&]() { pre1_cache.zero_(); }, [&]() {
                require_ok(engine->run_pre_pass1(input, pre1_cache, 0, pre1_batch),
                           "pre_model_pass1");
            });
        print_result("pre_model_pass1", 0, length, options.iterations, stats);
    }

    const int32_t pre2_batch = engine->pre_pass2_batch_size();
    if (pre2_batch <= 0) throw std::runtime_error("pre pass2 has no fixed representative batch");
    phono::context::PersistentTensor pre2_cache = phono::context::make_zero_persistent_tensor(
        {cfg.pre_model.mhsa_layers, 2, pre2_batch, cfg.pre_model.max_seqlen,
         cfg.pre_model.mhsa_heads, cfg.pre_model.self_head_dim()});
    for (int32_t length : sequence_lengths) {
        const std::vector<int32_t> pinyin = pinyin_prefix(tokenizer, length);
        phono::engine::PostModelOutput post;
        require_ok(engine->run_post_model(pinyin, post), "post_model setup");
        phono::engine::DecoderModelOutput output;
        const std::vector<int32_t> input(static_cast<size_t>(pre2_batch), context_id);
        const std::vector<int32_t> positions(static_cast<size_t>(pre2_batch), 0);
        const Statistics stats = measure(
            options.warmups, options.iterations, [&]() { pre2_cache.zero_(); }, [&]() {
                require_ok(engine->run_pre_pass2(input, pre2_cache, positions, post, 0, output),
                           "pre_model_pass2");
            });
        print_result("pre_model_pass2", 0, length, options.iterations, stats);
    }

    phono::context::PersistentTensor cache_microbenchmark =
        phono::context::make_zero_persistent_tensor(
            {cfg.pre_model.mhsa_layers, 2, core.beam_size, cfg.pre_model.max_seqlen,
             cfg.pre_model.mhsa_heads, cfg.pre_model.self_head_dim()});
    std::fill(cache_microbenchmark.storage->begin(), cache_microbenchmark.storage->end(), 1.0f);
    const Statistics zero_stats = measure(
        options.warmups, options.iterations, []() {},
        [&]() { cache_microbenchmark.zero_(); });
    print_result("kv_zero_full", 0, cfg.pre_model.max_seqlen, options.iterations, zero_stats);

    std::vector<int32_t> parents(static_cast<size_t>(core.beam_size), 0);
    for (int32_t beam = 0; beam < core.beam_size; ++beam) {
        parents[static_cast<size_t>(beam)] = (beam + 1) % core.beam_size;
    }
    const Statistics reorder_stats = measure(
        options.warmups, options.iterations, []() {},
        [&]() { cache_microbenchmark.reorder_batches(parents); });
    print_result("kv_reorder_full", 0, cfg.pre_model.max_seqlen, options.iterations,
                 reorder_stats);

    std::vector<float_t> reorder_scratch;
    const int32_t reorder_delta_length = std::min(8, cfg.pre_model.max_seqlen);
    const Statistics reorder_delta_stats = measure(
        options.warmups, options.iterations, []() {},
        [&]() {
            cache_microbenchmark.reorder_batch_slice(
                parents, 0, reorder_delta_length, reorder_scratch);
        });
    print_result("kv_reorder_delta", 0, reorder_delta_length, options.iterations,
                 reorder_delta_stats);

    phono::context::ContextManager manager(cfg, 1, core);
    phono::context::Context& context = manager.get_context_by_id(0);
    phono::engine::InferenceSession session(*engine, core);
    const Statistics reset_stats = measure(
        options.warmups, options.iterations, []() {},
        [&]() { require_ok(session.reset(context), "session reset"); });
    print_result("session_reset", 0, 0, options.iterations, reset_stats);
    for (int32_t history_length : std::vector<int32_t>{0, 16, 30, 64}) {
        const std::vector<int32_t> history(static_cast<size_t>(history_length), context_id);
        for (int32_t pinyin_length : std::vector<int32_t>{1, 2, 4, 8, 16, 30}) {
            const std::vector<int32_t> pinyin = pinyin_prefix(tokenizer, pinyin_length);
            const Statistics stats = measure(
                options.warmups, options.iterations,
                [&]() {
                    require_ok(session.reset(context), "generate reset");
                    if (!history.empty()) {
                        require_ok(session.fill(context, history), "generate history setup");
                    }
                },
                [&]() {
                    const auto result = session.generate(context, pinyin);
                    require_ok(result.error, "generate");
                });
            print_result("generate", history_length, pinyin_length, options.iterations, stats);
        }

        const std::vector<int32_t> one_token{context_id};
        const Statistics fill_stats = measure(
            options.warmups, options.iterations,
            [&]() {
                require_ok(session.reset(context), "fill reset");
                if (!history.empty()) {
                    require_ok(session.fill(context, history), "fill history setup");
                }
            },
            [&]() { require_ok(session.fill(context, one_token), "incremental fill"); });
        print_result("fill_incremental", history_length, 1, options.iterations, fill_stats);
    }

    std::cout << "# peak_rss_kib_final=" << peak_rss_kib() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "performance_benchmark: " << error.what() << '\n';
        return 2;
    }
}
