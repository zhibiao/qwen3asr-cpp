#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "errors.h"

namespace qwen3_asr {

Sampler::Sampler(SamplerConfig config)
    : config_(config), generator_(config.seed) {
    validate_config();
}

void Sampler::validate_config() const {
    if (!std::isfinite(config_.temperature) ||
        config_.temperature < 0.0F) {
        throw Error(
            ErrorCode::InvalidArgument,
            "sampler temperature must be finite and non-negative"
        );
    }
    if (config_.top_k < 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "sampler top-k must be non-negative"
        );
    }
    if (!std::isfinite(config_.top_p) ||
        config_.top_p <= 0.0F ||
        config_.top_p > 1.0F) {
        throw Error(
            ErrorCode::InvalidArgument,
            "sampler top-p must be in the interval (0, 1]"
        );
    }
}

std::int32_t Sampler::sample(const std::vector<float> & logits) {
    return sample(logits.data(), logits.size());
}

std::int32_t Sampler::sample(const float * logits, std::size_t count) {
    if (logits == nullptr || count == 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "sampler logits are empty"
        );
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (std::isnan(logits[index]) ||
            logits[index] == std::numeric_limits<float>::infinity()) {
            throw Error(
                ErrorCode::InferenceFailed,
                "sampler logits contain NaN or positive infinity"
            );
        }
    }

    if (config_.temperature == 0.0F) {
        std::size_t best = 0;
        float best_logit = logits[0];
        for (std::size_t index = 1; index < count; ++index) {
            if (logits[index] > best_logit) {
                best = index;
                best_logit = logits[index];
            }
        }
        if (!std::isfinite(best_logit)) {
            throw Error(
                ErrorCode::InferenceFailed,
                "sampler has no finite token candidates"
            );
        }
        return static_cast<std::int32_t>(best);
    }

    struct Candidate {
        std::int32_t token_id = 0;
        float score = 0.0F;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(count);
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < count; ++index) {
        if (logits[index] == -std::numeric_limits<float>::infinity()) {
            continue;
        }
        candidates.push_back(Candidate{
            static_cast<std::int32_t>(index),
            logits[index],
        });
        maximum = std::max(maximum, logits[index]);
    }
    if (candidates.empty()) {
        throw Error(
            ErrorCode::InferenceFailed,
            "sampler has no finite token candidates"
        );
    }

    for (Candidate & candidate : candidates) {
        candidate.score =
            (candidate.score - maximum) / config_.temperature;
    }
    const auto higher_score = [](const Candidate & left,
                                 const Candidate & right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return left.token_id < right.token_id;
    };

    if (config_.top_k > 0 &&
        candidates.size() > static_cast<std::size_t>(config_.top_k)) {
        const auto cutoff = candidates.begin() + config_.top_k;
        std::nth_element(
            candidates.begin(),
            cutoff,
            candidates.end(),
            higher_score
        );
        candidates.resize(static_cast<std::size_t>(config_.top_k));
    }
    std::sort(candidates.begin(), candidates.end(), higher_score);

    std::vector<double> probabilities(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        probabilities[index] =
            std::exp(static_cast<double>(candidates[index].score));
    }

    if (config_.top_p < 1.0F) {
        const double total = std::accumulate(
            probabilities.begin(),
            probabilities.end(),
            0.0
        );
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw Error(
                ErrorCode::InferenceFailed,
                "sampler probability mass is invalid"
            );
        }
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < probabilities.size(); ++keep) {
            cumulative += probabilities[keep] / total;
            if (cumulative >= config_.top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max<std::size_t>(1, std::min(
            keep,
            probabilities.size()
        ));
        probabilities.resize(keep);
        candidates.resize(keep);
    }

    std::discrete_distribution<std::size_t> distribution(
        probabilities.begin(),
        probabilities.end()
    );
    const std::size_t selected = distribution(generator_);
    return candidates[selected].token_id;
}

}  // namespace qwen3_asr
