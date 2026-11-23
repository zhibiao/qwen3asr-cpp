#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace qwen3_asr {

struct SamplerConfig {
    // A non-positive temperature selects the highest-probability token.
    float temperature = 0.0F;
    // Zero disables top-k filtering.
    int top_k = 0;
    // One disables nucleus filtering.
    float top_p = 1.0F;
    std::uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {});

    std::int32_t sample(const float * logits, std::size_t count);
    std::int32_t sample(const std::vector<float> & logits);

private:
    void validate_config() const;

    SamplerConfig config_;
    std::mt19937_64 generator_;
};

}  // namespace qwen3_asr
