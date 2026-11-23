#include "test_framework.hpp"

#include <limits>
#include <vector>

#include "errors.h"
#include "sampler.h"

namespace {

QWEN3ASR_TEST_CASE(sampler_greedy_selects_highest_logit) {
    qwen3_asr::Sampler sampler;
    const std::vector<float> logits{0.5F, 3.0F, -1.0F, 3.0F};
    CHECK(sampler.sample(logits) == 1);
}

QWEN3ASR_TEST_CASE(sampler_top_k_filters_tail_and_is_reproducible) {
    qwen3_asr::SamplerConfig config;
    config.temperature = 1.0F;
    config.top_k = 2;
    config.seed = 1234;

    qwen3_asr::Sampler first(config);
    qwen3_asr::Sampler second(config);
    const std::vector<float> logits{10.0F, 9.0F, -100.0F, -100.0F};
    for (int index = 0; index < 128; ++index) {
        const std::int32_t first_token = first.sample(logits);
        const std::int32_t second_token = second.sample(logits);
        CHECK(first_token == second_token);
        CHECK(first_token == 0 || first_token == 1);
    }
}

QWEN3ASR_TEST_CASE(sampler_top_k_one_is_greedy) {
    qwen3_asr::SamplerConfig config;
    config.temperature = 2.0F;
    config.top_k = 1;
    qwen3_asr::Sampler sampler(config);
    const std::vector<float> logits{0.0F, 5.0F, 1.0F};
    for (int index = 0; index < 16; ++index) {
        CHECK(sampler.sample(logits) == 1);
    }
}

QWEN3ASR_TEST_CASE(sampler_top_p_filters_low_probability_tail) {
    qwen3_asr::SamplerConfig config;
    config.temperature = 1.0F;
    config.top_p = 0.75F;
    config.seed = 99;
    qwen3_asr::Sampler sampler(config);
    const std::vector<float> logits{0.7F, 0.6F, -20.0F};
    for (int index = 0; index < 128; ++index) {
        CHECK(sampler.sample(logits) != 2);
    }
}

QWEN3ASR_TEST_CASE(sampler_rejects_invalid_configuration) {
    qwen3_asr::SamplerConfig config;
    config.temperature = -1.0F;
    bool threw = false;
    try {
        qwen3_asr::Sampler sampler(config);
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() == qwen3_asr::ErrorCode::InvalidArgument;
    }
    CHECK(threw);
}

QWEN3ASR_TEST_CASE(sampler_rejects_nan_logits) {
    qwen3_asr::Sampler sampler;
    bool threw = false;
    try {
        static_cast<void>(sampler.sample(
            std::vector<float>{0.0F, std::numeric_limits<float>::quiet_NaN()}
        ));
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() == qwen3_asr::ErrorCode::InferenceFailed;
    }
    CHECK(threw);
}

}  // namespace
