#include "audio_frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <miniaudio.h>

#include "errors.h"

namespace qwen3_asr {
namespace {

std::string miniaudio_error(ma_result result) {
    const char * description = ma_result_description(result);
    return description != nullptr ? description : "unknown miniaudio error";
}

void validate_finite_samples(
    const std::vector<float> & samples,
    const char * context
) {
    for (float sample : samples) {
        if (!std::isfinite(sample)) {
            throw Error(
                ErrorCode::InvalidAudioData,
                std::string(context) + " contains a non-finite sample"
            );
        }
    }
}

constexpr double kPi = 3.141592653589793238462643383279502884;

double hertz_to_mel_slaney(double frequency) {
    constexpr double min_log_hertz = 1000.0;
    constexpr double min_log_mel = 15.0;
    const double logstep = 27.0 / std::log(6.4);
    if (frequency >= min_log_hertz) {
        return min_log_mel +
               std::log(frequency / min_log_hertz) * logstep;
    }
    return 3.0 * frequency / 200.0;
}

double mel_to_hertz_slaney(double mel) {
    constexpr double min_log_hertz = 1000.0;
    constexpr double min_log_mel = 15.0;
    const double logstep = std::log(6.4) / 27.0;
    if (mel >= min_log_mel) {
        return min_log_hertz *
               std::exp(logstep * (mel - min_log_mel));
    }
    return 200.0 * mel / 3.0;
}

int smallest_fft_factor(int value) {
    if (value % 2 == 0) {
        return 2;
    }
    for (int factor = 3; factor * factor <= value; factor += 2) {
        if (value % factor == 0) {
            return factor;
        }
    }
    return value;
}

std::vector<std::complex<double>> fft_recursive(
    const std::vector<std::complex<double>> & input
) {
    const int size = static_cast<int>(input.size());
    if (size <= 1) {
        return input;
    }

    const int radix = smallest_fft_factor(size);
    if (radix == size) {
        std::vector<std::complex<double>> output(
            static_cast<std::size_t>(size)
        );
        for (int k = 0; k < size; ++k) {
            std::complex<double> sum{};
            for (int n = 0; n < size; ++n) {
                const double angle =
                    -2.0 * kPi * static_cast<double>(k * n) / size;
                sum += input[static_cast<std::size_t>(n)] *
                       std::complex<double>(
                           std::cos(angle),
                           std::sin(angle)
                       );
            }
            output[static_cast<std::size_t>(k)] = sum;
        }
        return output;
    }

    const int subsequence_size = size / radix;
    std::vector<std::vector<std::complex<double>>> transformed(
        static_cast<std::size_t>(radix)
    );
    for (int part = 0; part < radix; ++part) {
        std::vector<std::complex<double>> subsequence(
            static_cast<std::size_t>(subsequence_size)
        );
        for (int index = 0; index < subsequence_size; ++index) {
            subsequence[static_cast<std::size_t>(index)] =
                input[static_cast<std::size_t>(
                    index * radix + part
                )];
        }
        transformed[static_cast<std::size_t>(part)] =
            fft_recursive(subsequence);
    }

    std::vector<std::complex<double>> output(
        static_cast<std::size_t>(size)
    );
    for (int k = 0; k < subsequence_size; ++k) {
        for (int quotient = 0; quotient < radix; ++quotient) {
            std::complex<double> sum{};
            for (int part = 0; part < radix; ++part) {
                const double angle =
                    -2.0 * kPi * static_cast<double>(part) *
                    static_cast<double>(
                        k + quotient * subsequence_size
                    ) /
                    static_cast<double>(size);
                sum +=
                    transformed[static_cast<std::size_t>(part)]
                               [static_cast<std::size_t>(k)] *
                    std::complex<double>(
                        std::cos(angle),
                        std::sin(angle)
                    );
            }
            output[static_cast<std::size_t>(
                k + quotient * subsequence_size
            )] = sum;
        }
    }
    return output;
}

std::size_t reflect_index(std::int64_t index, std::size_t size) {
    if (size <= 1) {
        return 0;
    }
    const std::int64_t period =
        static_cast<std::int64_t>(2 * size - 2);
    std::int64_t reflected = index % period;
    if (reflected < 0) {
        reflected += period;
    }
    if (reflected >= static_cast<std::int64_t>(size)) {
        reflected = period - reflected;
    }
    return static_cast<std::size_t>(reflected);
}

double bessel_i0(double value) {
    double sum = 1.0;
    double term = 1.0;
    constexpr int iterations = 32;
    for (int index = 1; index <= iterations; ++index) {
        const double ratio =
            value / (2.0 * static_cast<double>(index));
        term *= ratio * ratio;
        sum += term;
        if (term < sum * 1.0e-15) {
            break;
        }
    }
    return sum;
}

double sinc(double value) {
    if (std::fabs(value) < 1.0e-12) {
        return 1.0;
    }
    return std::sin(kPi * value) / (kPi * value);
}

}  // namespace

AudioBuffer AudioLoader::load(const std::filesystem::path & path) {
    if (path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "audio path is empty");
    }

    std::error_code filesystem_error;
    if (!std::filesystem::exists(path, filesystem_error) ||
        !std::filesystem::is_regular_file(path, filesystem_error)) {
        throw Error(
            ErrorCode::AudioLoadFailed,
            "audio file does not exist: " + path.string()
        );
    }

    ma_decoder_config config =
        ma_decoder_config_init(ma_format_f32, 0, 0);
    ma_decoder decoder{};

#if defined(_WIN32)
    ma_result result =
        ma_decoder_init_file_w(path.c_str(), &config, &decoder);
#else
    result = ma_decoder_init_file(path.c_str(), &config, &decoder);
#endif

    if (result != MA_SUCCESS) {
        throw Error(
            ErrorCode::UnsupportedAudioFormat,
            "failed to open audio '" + path.string() + "': " +
                miniaudio_error(result)
        );
    }
    try {
        ma_format format = ma_format_unknown;
        ma_uint32 channels = 0;
        ma_uint32 sample_rate = 0;
        std::array<ma_channel, MA_MAX_CHANNELS> channel_map{};

        result = ma_decoder_get_data_format(
            &decoder,
            &format,
            &channels,
            &sample_rate,
            channel_map.data(),
            channel_map.size()
        );
        if (result != MA_SUCCESS) {
            throw Error(
                ErrorCode::AudioLoadFailed,
                "failed to query decoded audio format: " +
                    miniaudio_error(result)
            );
        }

        if (format != ma_format_f32 || channels == 0 || sample_rate == 0) {
            throw Error(
                ErrorCode::UnsupportedAudioFormat,
                "decoder did not produce mono/stereo float32 PCM"
            );
        }

        constexpr ma_uint64 kFramesPerRead = 4096;
        std::vector<float> read_buffer(
            static_cast<std::size_t>(kFramesPerRead) * channels
        );
        AudioBuffer audio;
        audio.sample_rate = static_cast<std::uint32_t>(sample_rate);
        audio.channels = static_cast<std::uint32_t>(channels);

        while (true) {
            ma_uint64 frames_read = 0;
            result = ma_decoder_read_pcm_frames(
                &decoder,
                read_buffer.data(),
                kFramesPerRead,
                &frames_read
            );

            if (result == MA_AT_END) {
                break;
            }
            if (result != MA_SUCCESS) {
                throw Error(
                    ErrorCode::AudioLoadFailed,
                    "failed while decoding audio: " +
                        miniaudio_error(result)
                );
            }
            if (frames_read == 0) {
                break;
            }

            const std::size_t sample_count =
                static_cast<std::size_t>(frames_read) * channels;
            audio.samples.insert(
                audio.samples.end(),
                read_buffer.begin(),
                read_buffer.begin() +
                    static_cast<std::ptrdiff_t>(sample_count)
            );
        }

        validate_finite_samples(audio.samples, "decoded audio");
        ma_decoder_uninit(&decoder);
        return audio;
    } catch (...) {
        ma_decoder_uninit(&decoder);
        throw;
    }
}

MonoAudio AudioProcessor::downmix_to_mono(const AudioBuffer & audio) {
    if (audio.channels == 0) {
        throw Error(
            ErrorCode::InvalidAudioData,
            "audio channel count must be greater than zero"
        );
    }
    if (audio.samples.size() % audio.channels != 0) {
        throw Error(
            ErrorCode::InvalidAudioData,
            "interleaved audio sample count is not divisible by channel count"
        );
    }

    MonoAudio mono;
    mono.sample_rate = audio.sample_rate;
    if (audio.samples.empty()) {
        return mono;
    }

    const std::size_t frames = audio.frame_count();
    mono.samples.resize(frames);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0.0;
        const std::size_t base = frame * audio.channels;
        for (std::uint32_t channel = 0; channel < audio.channels; ++channel) {
            const float sample = audio.samples[base + channel];
            if (!std::isfinite(sample)) {
                throw Error(
                    ErrorCode::InvalidAudioData,
                    "audio contains a non-finite sample"
                );
            }
            sum += sample;
        }
        mono.samples[frame] =
            static_cast<float>(sum / static_cast<double>(audio.channels));
    }

    return mono;
}

MonoAudio AudioProcessor::normalize_peak(MonoAudio audio) {
    validate_finite_samples(audio.samples, "mono audio");
    if (audio.samples.empty()) {
        return audio;
    }

    float peak = 0.0F;
    for (float sample : audio.samples) {
        peak = std::max(peak, std::fabs(sample));
    }

    if (peak > 1.0F) {
        const float scale = 1.0F / peak;
        for (float & sample : audio.samples) {
            sample *= scale;
        }
    }

    for (float & sample : audio.samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }

    return audio;
}

MonoAudio AudioProcessor::prepare_mono(const AudioBuffer & audio) {
    return normalize_peak(downmix_to_mono(audio));
}

MonoAudio AudioResampler::resample(
    const MonoAudio & audio,
    std::uint32_t target_sample_rate
) {
    if (audio.sample_rate == 0 || target_sample_rate == 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "source and target sample rates must be greater than zero"
        );
    }
    validate_finite_samples(audio.samples, "resampler input");

    if (audio.sample_rate == target_sample_rate) {
        return audio;
    }
    if (audio.samples.empty()) {
        return MonoAudio{target_sample_rate, {}};
    }

    constexpr int phase_count = 4096;
    constexpr int zero_crossings = 32;
    constexpr double kaiser_beta = 8.6;

    const std::uint64_t output_frame_count_u64 =
        (static_cast<std::uint64_t>(audio.samples.size()) *
             target_sample_rate +
         audio.sample_rate - 1) /
        audio.sample_rate;
    if (output_frame_count_u64 >
        std::numeric_limits<std::size_t>::max()) {
        throw Error(
            ErrorCode::ResampleFailed,
            "resampled audio length exceeds addressable memory"
        );
    }

    const double cutoff = std::min(
        1.0,
        static_cast<double>(target_sample_rate) /
            static_cast<double>(audio.sample_rate) *
            0.96
    );
    const int radius = std::max(
        1,
        static_cast<int>(std::ceil(
            static_cast<double>(zero_crossings) / cutoff
        ))
    );
    if (radius > 2048) {
        throw Error(
            ErrorCode::ResampleFailed,
            "sample-rate ratio requires an unsupported filter radius"
        );
    }
    const int taps = radius * 2;
    const double denominator = bessel_i0(kaiser_beta);
    std::vector<float> kernel(
        static_cast<std::size_t>(phase_count) * taps
    );

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    hardware_threads = std::max(1U, hardware_threads);
    const int kernel_workers = std::min(
        phase_count,
        static_cast<int>(hardware_threads)
    );
    std::atomic<int> next_phase{0};
    std::atomic<bool> kernel_failed{false};
    std::mutex kernel_error_mutex;
    std::exception_ptr kernel_error;
    const auto build_kernel = [&]() {
        while (!kernel_failed.load(std::memory_order_relaxed)) {
            const int phase = next_phase.fetch_add(
                1,
                std::memory_order_relaxed
            );
            if (phase >= phase_count) {
                break;
            }
            try {
                const double fraction =
                    static_cast<double>(phase) /
                    static_cast<double>(phase_count);
                double weight_sum = 0.0;
                for (int tap = 0; tap < taps; ++tap) {
                    const int offset = tap - radius + 1;
                    const double distance = fraction - offset;
                    double weight = 0.0;
                    if (std::fabs(distance) < radius) {
                        const double normalized_distance =
                            distance / static_cast<double>(radius);
                        const double window =
                            bessel_i0(
                                kaiser_beta *
                                std::sqrt(std::max(
                                    0.0,
                                    1.0 - normalized_distance *
                                              normalized_distance
                                ))
                            ) /
                            denominator;
                        weight =
                            cutoff *
                            sinc(cutoff * distance) *
                            window;
                    }
                    kernel[
                        static_cast<std::size_t>(phase) * taps + tap
                    ] = static_cast<float>(weight);
                    weight_sum += weight;
                }

                if (std::fabs(weight_sum) < 1.0e-12) {
                    throw Error(
                        ErrorCode::ResampleFailed,
                        "resampler kernel has zero DC gain"
                    );
                }
                for (int tap = 0; tap < taps; ++tap) {
                    kernel[
                        static_cast<std::size_t>(phase) * taps + tap
                    ] /= static_cast<float>(weight_sum);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(kernel_error_mutex);
                if (!kernel_error) {
                    kernel_error = std::current_exception();
                }
                kernel_failed.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };
    std::vector<std::thread> kernel_threads;
    kernel_threads.reserve(
        static_cast<std::size_t>(kernel_workers - 1)
    );
    for (int worker = 1; worker < kernel_workers; ++worker) {
        kernel_threads.emplace_back(build_kernel);
    }
    build_kernel();
    for (std::thread & worker : kernel_threads) {
        worker.join();
    }
    if (kernel_error) {
        std::rethrow_exception(kernel_error);
    }

    MonoAudio output;
    output.sample_rate = target_sample_rate;
    output.samples.resize(
        static_cast<std::size_t>(output_frame_count_u64)
    );

    const auto resample_range = [&](
        std::size_t begin,
        std::size_t end
    ) {
        for (std::size_t output_index = begin;
             output_index < end;
             ++output_index) {
            const std::uint64_t numerator =
                static_cast<std::uint64_t>(output_index) *
                audio.sample_rate;
            const std::int64_t base =
                static_cast<std::int64_t>(
                    numerator / target_sample_rate
                );
            const std::uint64_t remainder =
                numerator % target_sample_rate;
            const std::size_t phase = static_cast<std::size_t>(
                (remainder * phase_count) / target_sample_rate
            );
            const float * phase_kernel =
                kernel.data() + phase * taps;

            double sum = 0.0;
            const std::int64_t first_input =
                base - radius + 1;
            const std::int64_t last_input = base + radius;
            if (first_input >= 0 &&
                last_input <
                    static_cast<std::int64_t>(audio.samples.size())) {
                const float * samples =
                    audio.samples.data() + first_input;
                for (int tap = 0; tap < taps; ++tap) {
                    sum += static_cast<double>(samples[tap]) *
                        phase_kernel[tap];
                }
            } else {
                for (int tap = 0; tap < taps; ++tap) {
                    const int offset = tap - radius + 1;
                    const std::size_t input_index = reflect_index(
                        base + offset,
                        audio.samples.size()
                    );
                    sum +=
                        static_cast<double>(
                            audio.samples[input_index]
                        ) * phase_kernel[tap];
                }
            }
            output.samples[output_index] = static_cast<float>(sum);
        }
    };

    constexpr std::size_t output_block_size = 256;
    const std::size_t output_blocks =
        (output.samples.size() + output_block_size - 1) /
        output_block_size;
    const int output_workers = std::min(
        static_cast<int>(output_blocks),
        static_cast<int>(hardware_threads)
    );
    std::atomic<std::size_t> next_output_block{0};
    const auto run_output_worker = [&]() {
        while (true) {
            const std::size_t block = next_output_block.fetch_add(
                1,
                std::memory_order_relaxed
            );
            if (block >= output_blocks) {
                break;
            }
            const std::size_t begin = block * output_block_size;
            const std::size_t end = std::min(
                begin + output_block_size,
                output.samples.size()
            );
            resample_range(begin, end);
        }
    };
    std::vector<std::thread> output_threads;
    output_threads.reserve(
        static_cast<std::size_t>(output_workers - 1)
    );
    for (int worker = 1; worker < output_workers; ++worker) {
        output_threads.emplace_back(run_output_worker);
    }
    run_output_worker();
    for (std::thread & worker : output_threads) {
        worker.join();
    }

    validate_finite_samples(output.samples, "resampler output");
    return output;
}

MelFilterBank AudioFrontend::make_mel_filter_bank(
    const AudioFrontendConfig & config
) {
    if (config.feature_size <= 0 ||
        config.sampling_rate <= 0 ||
        config.n_fft <= 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid Whisper feature extractor configuration"
        );
    }
    if (config.mel_min_frequency < 0.0F ||
        config.mel_max_frequency <= config.mel_min_frequency ||
        config.mel_max_frequency > config.sampling_rate / 2.0F) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid mel frequency range"
        );
    }

    MelFilterBank filters;
    filters.n_frequency_bins = config.n_fft / 2 + 1;
    filters.n_mel_bins = config.feature_size;
    filters.data.resize(
        static_cast<std::size_t>(filters.n_frequency_bins) *
        filters.n_mel_bins
    );

    const int mel_points_count = filters.n_mel_bins + 2;
    const double mel_min =
        hertz_to_mel_slaney(config.mel_min_frequency);
    const double mel_max =
        hertz_to_mel_slaney(config.mel_max_frequency);

    std::vector<double> mel_points(
        static_cast<std::size_t>(mel_points_count)
    );
    for (int index = 0; index < mel_points_count; ++index) {
        const double ratio =
            static_cast<double>(index) /
            static_cast<double>(mel_points_count - 1);
        mel_points[static_cast<std::size_t>(index)] =
            mel_min + ratio * (mel_max - mel_min);
    }

    std::vector<double> filter_frequencies(
        static_cast<std::size_t>(mel_points_count)
    );
    for (int index = 0; index < mel_points_count; ++index) {
        filter_frequencies[static_cast<std::size_t>(index)] =
            mel_to_hertz_slaney(
                mel_points[static_cast<std::size_t>(index)]
            );
    }

    std::vector<double> frequency_diffs(
        static_cast<std::size_t>(filters.n_mel_bins + 1)
    );
    for (int index = 0; index <= filters.n_mel_bins; ++index) {
        frequency_diffs[static_cast<std::size_t>(index)] =
            filter_frequencies[static_cast<std::size_t>(index + 1)] -
            filter_frequencies[static_cast<std::size_t>(index)];
    }

    const double max_fft_frequency = config.sampling_rate / 2.0;
    for (int bin = 0; bin < filters.n_frequency_bins; ++bin) {
        const double fft_frequency =
            max_fft_frequency * static_cast<double>(bin) /
            static_cast<double>(filters.n_frequency_bins - 1);

        for (int mel = 0; mel < filters.n_mel_bins; ++mel) {
            const double lower =
                filter_frequencies[static_cast<std::size_t>(mel)];
            const double upper =
                filter_frequencies[static_cast<std::size_t>(mel + 2)];
            const double down_slope =
                (fft_frequency - lower) /
                frequency_diffs[static_cast<std::size_t>(mel)];
            const double up_slope =
                (upper - fft_frequency) /
                frequency_diffs[static_cast<std::size_t>(mel + 1)];
            double weight = std::max(
                0.0,
                std::min(down_slope, up_slope)
            );
            weight *= 2.0 / (upper - lower);

            filters.data[
                static_cast<std::size_t>(bin) *
                    filters.n_mel_bins +
                mel
            ] = static_cast<float>(weight);
        }
    }

    return filters;
}

AudioFrontend::AudioFrontend(
    AudioFrontendConfig config,
    MelFilterBank mel_filters
)
    : config_(config), mel_filters_(std::move(mel_filters)) {
    if (config_.feature_size <= 0 ||
        config_.sampling_rate <= 0 ||
        config_.n_fft <= 0 ||
        config_.hop_length <= 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid Whisper feature extractor configuration"
        );
    }
    if (config_.dither != 0.0F) {
        throw Error(
            ErrorCode::InvalidArgument,
            "non-zero dither is not supported"
        );
    }

    const int expected_frequency_bins = config_.n_fft / 2 + 1;
    if (mel_filters_.n_frequency_bins != expected_frequency_bins ||
        mel_filters_.n_mel_bins != config_.feature_size ||
        mel_filters_.data.size() !=
            static_cast<std::size_t>(expected_frequency_bins) *
                config_.feature_size) {
        throw Error(
            ErrorCode::InvalidArgument,
            "mel filter shape does not match feature extractor config"
        );
    }
}

LogMelFeatures AudioFrontend::extract(
    const MonoAudio & audio
) const {
    if (audio.sample_rate !=
        static_cast<std::uint32_t>(config_.sampling_rate)) {
        throw Error(
            ErrorCode::InvalidArgument,
            "audio sample rate does not match feature extractor config"
        );
    }
    if (audio.samples.empty()) {
        throw Error(
            ErrorCode::InvalidAudioData,
            "cannot extract features from empty audio"
        );
    }
    validate_finite_samples(audio.samples, "feature extractor input");

    LogMelFeatures result;
    result.n_mel_bins = config_.feature_size;
    result.n_frames = static_cast<int>(
        audio.samples.size() /
        static_cast<std::size_t>(config_.hop_length)
    );
    if (result.n_frames <= 0) {
        throw Error(
            ErrorCode::InvalidAudioData,
            "audio is too short for Whisper feature extraction"
        );
    }

    result.data.resize(
        static_cast<std::size_t>(result.n_mel_bins) * result.n_frames
    );
    result.attention_mask.assign(
        static_cast<std::size_t>(result.n_frames),
        1
    );

    const int half_window = config_.n_fft / 2;
    std::vector<double> centered(
        audio.samples.size() +
        static_cast<std::size_t>(2 * half_window)
    );
    for (std::int64_t index = -half_window;
         index < static_cast<std::int64_t>(audio.samples.size()) +
                     half_window;
         ++index) {
        centered[static_cast<std::size_t>(index + half_window)] =
            static_cast<double>(
                audio.samples[reflect_index(index, audio.samples.size())]
            );
    }

    std::vector<double> window(
        static_cast<std::size_t>(config_.n_fft)
    );
    for (int index = 0; index < config_.n_fft; ++index) {
        window[static_cast<std::size_t>(index)] =
            0.5 -
            0.5 * std::cos(
                2.0 * kPi * static_cast<double>(index) /
                static_cast<double>(config_.n_fft)
            );
    }

    struct FrameWorkspace {
        std::vector<std::complex<double>> frame;
        std::vector<double> mel_power;

        FrameWorkspace(std::size_t fft_size, std::size_t mel_bins)
            : frame(fft_size), mel_power(mel_bins) {}
    };

    const auto process_frame = [&](
        int time,
        FrameWorkspace & workspace
    ) {
        const std::size_t offset =
            static_cast<std::size_t>(time * config_.hop_length);
        for (int index = 0; index < config_.n_fft; ++index) {
            workspace.frame[static_cast<std::size_t>(index)] =
                centered[offset + static_cast<std::size_t>(index)] *
                window[static_cast<std::size_t>(index)];
        }

        const std::vector<std::complex<double>> spectrum =
            fft_recursive(workspace.frame);
        std::fill(
            workspace.mel_power.begin(),
            workspace.mel_power.end(),
            0.0
        );
        for (int bin = 0; bin < mel_filters_.n_frequency_bins; ++bin) {
            const std::complex<double> value =
                spectrum[static_cast<std::size_t>(bin)];
            const double bin_power =
                value.real() * value.real() +
                value.imag() * value.imag();
            const float * filter_row =
                mel_filters_.data.data() +
                static_cast<std::size_t>(bin) *
                    mel_filters_.n_mel_bins;
            for (int mel = 0; mel < result.n_mel_bins; ++mel) {
                workspace.mel_power[static_cast<std::size_t>(mel)] +=
                    static_cast<double>(
                        filter_row[mel]
                    ) * bin_power;
            }
        }

        for (int mel = 0; mel < result.n_mel_bins; ++mel) {
            const double value = std::log10(
                std::max(
                    workspace.mel_power[static_cast<std::size_t>(mel)],
                    1.0e-10
                )
            );
            result.data[
                static_cast<std::size_t>(mel) * result.n_frames +
                static_cast<std::size_t>(time)
            ] = static_cast<float>(value);
        }
    };

    unsigned int hardware_threads = std::thread::hardware_concurrency();
    hardware_threads = std::max(1U, hardware_threads);
    const int worker_count = std::min(
        result.n_frames,
        result.n_frames < 64 ? 1 : static_cast<int>(hardware_threads)
    );
    std::atomic<int> next_frame{0};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;
    const auto run_worker = [&]() {
        FrameWorkspace workspace(
            static_cast<std::size_t>(config_.n_fft),
            static_cast<std::size_t>(result.n_mel_bins)
        );
        while (!stop.load(std::memory_order_relaxed)) {
            const int time = next_frame.fetch_add(
                1,
                std::memory_order_relaxed
            );
            if (time >= result.n_frames) {
                break;
            }
            try {
                process_frame(time, workspace);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
                stop.store(true, std::memory_order_relaxed);
                break;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count - 1));
    for (int worker = 1; worker < worker_count; ++worker) {
        workers.emplace_back(run_worker);
    }
    run_worker();
    for (std::thread & worker : workers) {
        worker.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }

    double maximum =
        *std::max_element(result.data.begin(), result.data.end());
    const double floor = maximum - 8.0;
    for (float & value : result.data) {
        value = static_cast<float>(
            (std::max(static_cast<double>(value), floor) + 4.0) /
            4.0
        );
    }

    validate_finite_samples(result.data, "log-mel output");
    return result;
}

}  // namespace qwen3_asr
