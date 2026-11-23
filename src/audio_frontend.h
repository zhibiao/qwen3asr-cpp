#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "configs.h"

namespace qwen3_asr {

constexpr std::uint32_t kQwen3AsrSampleRate = 16000;

struct AudioBuffer {
    std::uint32_t sample_rate = 0;
    std::uint32_t channels = 0;
    std::vector<float> samples;

    std::size_t frame_count() const noexcept {
        return channels == 0 ? 0 : samples.size() / channels;
    }

    bool empty() const noexcept {
        return samples.empty();
    }
};

struct MonoAudio {
    std::uint32_t sample_rate = 0;
    std::vector<float> samples;

    bool empty() const noexcept {
        return samples.empty();
    }
};

struct MelFilterBank {
    int n_frequency_bins = 0;
    int n_mel_bins = 0;

    // Row-major [n_frequency_bins, n_mel_bins].
    std::vector<float> data;
};

struct LogMelFeatures {
    int n_mel_bins = 0;
    int n_frames = 0;

    // Mel-major [n_mel_bins, n_frames].
    std::vector<float> data;
    std::vector<std::uint8_t> attention_mask;
};

class AudioLoader {
public:
    static AudioBuffer load(const std::filesystem::path & path);
};

class AudioProcessor {
public:
    static MonoAudio downmix_to_mono(const AudioBuffer & audio);
    static MonoAudio normalize_peak(MonoAudio audio);
    static MonoAudio prepare_mono(const AudioBuffer & audio);
};

class AudioResampler {
public:
    static MonoAudio resample(
        const MonoAudio & audio,
        std::uint32_t target_sample_rate
    );
};

class AudioFrontend {
public:
    AudioFrontend(
        AudioFrontendConfig config,
        MelFilterBank mel_filters
    );

    static MelFilterBank make_mel_filter_bank(
        const AudioFrontendConfig & config
    );

    LogMelFeatures extract(const MonoAudio & audio) const;

    const AudioFrontendConfig & config() const noexcept {
        return config_;
    }

private:
    AudioFrontendConfig config_;
    MelFilterBank mel_filters_;
};

}  // namespace qwen3_asr
