#include "test_framework.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "errors.h"
#include "audio_frontend.h"

namespace {

using qwen3_asr::ErrorCode;

void append_u16(std::vector<std::uint8_t> & bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void append_u32(std::vector<std::uint8_t> & bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    bytes.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

void append_fourcc(std::vector<std::uint8_t> & bytes, const char * value) {
    bytes.insert(bytes.end(), value, value + 4);
}

std::vector<std::uint8_t> make_wav(
    std::uint16_t format,
    std::uint16_t channels,
    std::uint32_t sample_rate,
    std::uint16_t bits_per_sample,
    const std::vector<std::uint8_t> & data
) {
    const std::uint16_t block_align = static_cast<std::uint16_t>(
        channels * ((bits_per_sample + 7U) / 8U)
    );

    std::vector<std::uint8_t> bytes;
    append_fourcc(bytes, "RIFF");
    append_u32(bytes, static_cast<std::uint32_t>(36U + data.size()));
    append_fourcc(bytes, "WAVE");
    append_fourcc(bytes, "fmt ");
    append_u32(bytes, 16);
    append_u16(bytes, format);
    append_u16(bytes, channels);
    append_u32(bytes, sample_rate);
    append_u32(bytes, sample_rate * block_align);
    append_u16(bytes, block_align);
    append_u16(bytes, bits_per_sample);
    append_fourcc(bytes, "data");
    append_u32(bytes, static_cast<std::uint32_t>(data.size()));
    bytes.insert(bytes.end(), data.begin(), data.end());
    if (data.size() % 2U != 0U) {
        bytes.push_back(0);
    }
    return bytes;
}

class TempFile {
public:
    TempFile(const std::string & name, const std::vector<std::uint8_t> & bytes)
        : path_(std::filesystem::temp_directory_path() / name) {
        std::ofstream output(path_, std::ios::binary);
        output.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size())
        );
        if (!output) {
            throw std::runtime_error("failed to create temporary WAV");
        }
    }

    ~TempFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    const std::filesystem::path & path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

template <typename Function>
bool throws_error(ErrorCode expected, Function && function) {
    try {
        function();
    } catch (const qwen3_asr::Error & error) {
        return error.code() == expected;
    } catch (...) {
        return false;
    }
    return false;
}

bool nearly_equal(float lhs, float rhs, float tolerance = 1.0e-6F) {
    return std::fabs(lhs - rhs) <= tolerance;
}

std::vector<float> read_float_binary(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open float fixture: " + path.string()
        );
    }

    const std::streamsize size = input.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error(
            "invalid float fixture size: " + path.string()
        );
    }
    input.seekg(0, std::ios::beg);

    std::vector<float> values(
        static_cast<std::size_t>(size) / sizeof(float)
    );
    input.read(
        reinterpret_cast<char *>(values.data()),
        size
    );
    if (!input) {
        throw std::runtime_error(
            "failed to read float fixture: " + path.string()
        );
    }
    return values;
}

std::vector<std::uint8_t> read_binary(
    const std::filesystem::path & path
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open binary fixture: " + path.string()
        );
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error(
            "invalid binary fixture size: " + path.string()
        );
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> values(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char *>(values.data()), size);
    if (!input) {
        throw std::runtime_error(
            "failed to read binary fixture: " + path.string()
        );
    }
    return values;
}

nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "failed to open JSON fixture: " + path.string()
        );
    }
    nlohmann::json value;
    input >> value;
    return value;
}

QWEN3ASR_TEST_CASE(audio_loader_reads_pcm16_stereo) {
    const std::vector<std::uint8_t> data{
        0x00, 0x00, 0x00, 0x40,
        0x00, 0x20, 0x00, 0xE0,
    };
    TempFile file(
        "qwen3_asr_pcm16_stereo.wav",
        make_wav(1, 2, 16000, 16, data)
    );

    const qwen3_asr::AudioBuffer audio =
        qwen3_asr::AudioLoader::load(file.path());
    CHECK(audio.sample_rate == 16000);
    CHECK(audio.channels == 2);
    CHECK(audio.frame_count() == 2);
    REQUIRE(audio.samples.size() == 4);
    CHECK(nearly_equal(audio.samples[0], 0.0F));
    CHECK(nearly_equal(audio.samples[1], 0.5F));
    CHECK(nearly_equal(audio.samples[2], 0.25F));
    CHECK(nearly_equal(audio.samples[3], -0.25F));
}

QWEN3ASR_TEST_CASE(audio_loader_reads_unsigned_8_bit_pcm) {
    const std::vector<std::uint8_t> data{0, 128, 255};
    TempFile file(
        "qwen3_asr_u8_mono.wav",
        make_wav(1, 1, 8000, 8, data)
    );

    const qwen3_asr::AudioBuffer audio =
        qwen3_asr::AudioLoader::load(file.path());
    REQUIRE(audio.samples.size() == 3);
    CHECK(nearly_equal(audio.samples[0], -1.0F));
    CHECK(nearly_equal(audio.samples[1], 0.0F));
    CHECK(nearly_equal(audio.samples[2], 127.0F / 128.0F));
}

QWEN3ASR_TEST_CASE(audio_loader_reads_float32_wav) {
    const std::vector<float> values{-0.75F, 0.0F, 0.5F};
    std::vector<std::uint8_t> data(values.size() * sizeof(float));
    std::memcpy(data.data(), values.data(), data.size());
    TempFile file(
        "qwen3_asr_float32_mono.wav",
        make_wav(3, 1, 16000, 32, data)
    );

    const qwen3_asr::AudioBuffer audio =
        qwen3_asr::AudioLoader::load(file.path());
    REQUIRE(audio.samples.size() == values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(nearly_equal(audio.samples[index], values[index]));
    }
}

QWEN3ASR_TEST_CASE(audio_loader_rejects_missing_file) {
    CHECK(throws_error(ErrorCode::AudioLoadFailed, []() {
        qwen3_asr::AudioLoader::load(
            std::filesystem::temp_directory_path() /
            "qwen3_asr_file_that_does_not_exist.wav"
        );
    }));
}

QWEN3ASR_TEST_CASE(audio_loader_rejects_unknown_format) {
    const std::vector<std::uint8_t> data{'n', 'o', 't', ' ', 'a', 'u', 'd',
                                         'i', 'o'};
    TempFile file("qwen3_asr_unknown_audio.bin", data);
    CHECK(throws_error(ErrorCode::UnsupportedAudioFormat, [&]() {
        qwen3_asr::AudioLoader::load(file.path());
    }));
}

QWEN3ASR_TEST_CASE(audio_processor_downmixes_and_normalizes) {
    qwen3_asr::AudioBuffer input;
    input.sample_rate = 16000;
    input.channels = 2;
    input.samples = {2.0F, 0.0F, -0.5F, 0.5F};

    const qwen3_asr::MonoAudio mono =
        qwen3_asr::AudioProcessor::prepare_mono(input);
    CHECK(mono.sample_rate == 16000);
    REQUIRE(mono.samples.size() == 2);
    CHECK(nearly_equal(mono.samples[0], 1.0F));
    CHECK(nearly_equal(mono.samples[1], 0.0F));
}

QWEN3ASR_TEST_CASE(audio_processor_rejects_invalid_interleaving) {
    qwen3_asr::AudioBuffer input;
    input.sample_rate = 16000;
    input.channels = 2;
    input.samples = {0.1F};
    CHECK(throws_error(ErrorCode::InvalidAudioData, [&]() {
        qwen3_asr::AudioProcessor::downmix_to_mono(input);
    }));
}

QWEN3ASR_TEST_CASE(audio_resampler_bypasses_matching_rate) {
    qwen3_asr::MonoAudio input;
    input.sample_rate = 16000;
    input.samples = {0.1F, -0.2F, 0.3F};

    const qwen3_asr::MonoAudio output =
        qwen3_asr::AudioResampler::resample(input, 16000);
    CHECK(output.sample_rate == input.sample_rate);
    CHECK(output.samples == input.samples);
}

QWEN3ASR_TEST_CASE(audio_resampler_preserves_interior_dc) {
    qwen3_asr::MonoAudio input;
    input.sample_rate = 48000;
    input.samples.assign(4800, 0.25F);

    const qwen3_asr::MonoAudio output =
        qwen3_asr::AudioResampler::resample(
            input,
            qwen3_asr::kQwen3AsrSampleRate
        );
    CHECK(output.sample_rate == qwen3_asr::kQwen3AsrSampleRate);
    CHECK(!output.samples.empty());
    CHECK(std::abs(
        static_cast<long long>(output.samples.size()) - 1600LL
    ) <= 8LL);

    for (std::size_t index = 100; index + 100 < output.samples.size();
         ++index) {
        CHECK(nearly_equal(output.samples[index], 0.25F, 2.0e-3F));
        if (std::fabs(output.samples[index] - 0.25F) > 2.0e-3F) {
            break;
        }
    }
}

QWEN3ASR_TEST_CASE(audio_resampler_rejects_zero_rate) {
    qwen3_asr::MonoAudio input;
    input.sample_rate = 48000;
    input.samples = {0.0F, 0.1F};
    CHECK(throws_error(ErrorCode::InvalidArgument, [&]() {
        qwen3_asr::AudioResampler::resample(input, 0);
    }));
}

QWEN3ASR_TEST_CASE(audio_pipeline_handles_repository_test_wavs) {
    const std::set<std::string> expected_files{
        "ar1.wav",
        "cantonese.wav",
        "codeswitch.wav",
        "de.wav",
        "es1.wav",
        "f1_noise.wav",
        "fast1.wav",
        "fr1.wav",
        "ja1.wav",
        "noise1-en.wav",
        "noise2.wav",
        "qiqiu1.wav",
        "raokouling.wav",
        "rap1.wav",
        "ru1.wav",
    };

    const std::filesystem::path test_wav_dir = QWEN3ASR_TEST_WAV_DIR;
    CHECK(std::filesystem::is_directory(test_wav_dir));

    for (const std::string & name : expected_files) {
        const std::filesystem::path path = test_wav_dir / name;
        CHECK(std::filesystem::is_regular_file(path));
        if (!std::filesystem::is_regular_file(path)) {
            continue;
        }

        const qwen3_asr::AudioBuffer decoded =
            qwen3_asr::AudioLoader::load(path);
        CHECK(decoded.channels == 1);
        CHECK(
            decoded.sample_rate == 16000 ||
            decoded.sample_rate == 44100
        );
        CHECK(!decoded.empty());

        qwen3_asr::MonoAudio mono =
            qwen3_asr::AudioProcessor::prepare_mono(decoded);
        CHECK(mono.sample_rate == decoded.sample_rate);
        CHECK(mono.samples.size() == decoded.frame_count());

        const std::uint64_t expected_output_frames =
            (static_cast<std::uint64_t>(decoded.frame_count()) *
                 qwen3_asr::kQwen3AsrSampleRate +
             decoded.sample_rate - 1) /
            decoded.sample_rate;
        mono = qwen3_asr::AudioResampler::resample(
            mono,
            qwen3_asr::kQwen3AsrSampleRate
        );
        CHECK(mono.sample_rate == qwen3_asr::kQwen3AsrSampleRate);
        CHECK(mono.samples.size() == expected_output_frames);

        bool samples_valid = true;
        for (float sample : mono.samples) {
            if (!std::isfinite(sample) ||
                sample < -1.0F ||
                sample > 1.0F) {
                samples_valid = false;
                break;
            }
        }
        CHECK(samples_valid);
    }
}

QWEN3ASR_TEST_CASE(audio_frontend_mel_filter_bank_matches_pytorch_golden) {
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const std::vector<float> expected =
        read_float_binary(fixture_dir / "mel_filters_201x128.f32");

    qwen3_asr::AudioFrontendConfig config;
    const qwen3_asr::MelFilterBank actual =
        qwen3_asr::AudioFrontend::make_mel_filter_bank(config);

    CHECK(actual.n_frequency_bins == 201);
    CHECK(actual.n_mel_bins == 128);
    REQUIRE(actual.data.size() == expected.size());

    double maximum_error = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(
                static_cast<double>(actual.data[index]) -
                static_cast<double>(expected[index])
            )
        );
    }
    CHECK(maximum_error <= 1.0e-7);
}

QWEN3ASR_TEST_CASE(audio_frontend_handles_silence_and_partial_frame) {
    qwen3_asr::AudioFrontendConfig config;
    const qwen3_asr::MelFilterBank filters =
        qwen3_asr::AudioFrontend::make_mel_filter_bank(config);
    const qwen3_asr::AudioFrontend frontend(config, filters);

    qwen3_asr::MonoAudio audio;
    audio.sample_rate = 16000;
    audio.samples.assign(1000, 0.0F);
    const qwen3_asr::LogMelFeatures features =
        frontend.extract(audio);

    CHECK(features.n_mel_bins == 128);
    CHECK(features.n_frames == 6);
    CHECK(features.attention_mask.size() == 6);
    REQUIRE(features.data.size() == 128 * 6);
    for (float value : features.data) {
        CHECK(nearly_equal(value, -1.5F, 1.0e-5F));
        if (!nearly_equal(value, -1.5F, 1.0e-5F)) {
            break;
        }
    }
}

QWEN3ASR_TEST_CASE(audio_frontend_rejects_invalid_config_and_rate) {
    qwen3_asr::AudioFrontendConfig config;
    const qwen3_asr::MelFilterBank filters =
        qwen3_asr::AudioFrontend::make_mel_filter_bank(config);
    qwen3_asr::MonoAudio audio;
    audio.sample_rate = 44100;
    audio.samples.assign(1000, 0.0F);
    CHECK(throws_error(ErrorCode::InvalidArgument, [&]() {
        qwen3_asr::AudioFrontend frontend(config, filters);
        frontend.extract(audio);
    }));

    config.dither = 0.1F;
    CHECK(throws_error(ErrorCode::InvalidArgument, [&]() {
        qwen3_asr::AudioFrontend frontend(config, filters);
    }));
}

QWEN3ASR_TEST_CASE(audio_frontend_matches_all_pytorch_goldens) {
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const nlohmann::json metadata =
        read_json(fixture_dir / "metadata.json");

    qwen3_asr::AudioFrontendConfig config;
    config.feature_size = metadata.at("feature_size").get<int>();
    config.sampling_rate = metadata.at("sample_rate").get<int>();
    config.n_fft = metadata.at("n_fft").get<int>();
    config.hop_length = metadata.at("hop_length").get<int>();

    const qwen3_asr::MelFilterBank filters =
        qwen3_asr::AudioFrontend::make_mel_filter_bank(config);
    const qwen3_asr::AudioFrontend frontend(config, filters);

    double overall_maximum_error = 0.0;
    int checked_files = 0;
    for (const auto & entry : metadata.at("files")) {
        const std::string name = entry.at("name").get<std::string>();
        const std::vector<float> pcm =
            read_float_binary(
                fixture_dir /
                entry.at("pcm_file").get<std::string>()
            );
        const std::vector<float> expected =
            read_float_binary(
                fixture_dir /
                entry.at("log_mel_file").get<std::string>()
            );
        const std::string stem =
            std::filesystem::path(name).stem().string();
        const std::vector<std::uint8_t> expected_mask =
            read_binary(
                fixture_dir /
                ("attention_mask_" + stem + ".u8")
            );

        qwen3_asr::MonoAudio audio;
        audio.sample_rate = config.sampling_rate;
        audio.samples = pcm;
        const qwen3_asr::LogMelFeatures actual = frontend.extract(audio);

        const int expected_mel =
            entry.at("log_mel_shape").at(0).get<int>();
        const int expected_frames =
            entry.at("log_mel_shape").at(1).get<int>();
        CHECK(actual.n_mel_bins == expected_mel);
        CHECK(actual.n_frames == expected_frames);
        CHECK(actual.attention_mask == expected_mask);
        REQUIRE(actual.data.size() == expected.size());

        double maximum_error = 0.0;
        for (std::size_t index = 0;
             index < expected.size();
             ++index) {
            maximum_error = std::max(
                maximum_error,
                std::fabs(
                    static_cast<double>(actual.data[index]) -
                    static_cast<double>(expected[index])
                )
            );
        }
        overall_maximum_error =
            std::max(overall_maximum_error, maximum_error);
        std::cout << "golden " << name
                  << " max_logmel_error=" << maximum_error << '\n';
        CHECK(maximum_error <= 5.0e-5);
        ++checked_files;
    }
    CHECK(checked_files == 15);
    std::cout << "golden overall_max_logmel_error="
              << overall_maximum_error << '\n';
}

QWEN3ASR_TEST_CASE(audio_frontend_end_to_end_matches_pytorch_goldens) {
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const nlohmann::json metadata =
        read_json(fixture_dir / "metadata.json");

    qwen3_asr::AudioFrontendConfig config;
    config.feature_size = metadata.at("feature_size").get<int>();
    config.sampling_rate = metadata.at("sample_rate").get<int>();
    config.n_fft = metadata.at("n_fft").get<int>();
    config.hop_length = metadata.at("hop_length").get<int>();
    const qwen3_asr::MelFilterBank filters =
        qwen3_asr::AudioFrontend::make_mel_filter_bank(config);
    const qwen3_asr::AudioFrontend frontend(config, filters);

    for (const auto & entry : metadata.at("files")) {
        const std::string name = entry.at("name").get<std::string>();
        const qwen3_asr::AudioBuffer decoded =
            qwen3_asr::AudioLoader::load(
                std::filesystem::path(QWEN3ASR_TEST_WAV_DIR) / name
            );
        qwen3_asr::MonoAudio mono =
            qwen3_asr::AudioProcessor::prepare_mono(decoded);
        mono = qwen3_asr::AudioResampler::resample(
            mono,
            config.sampling_rate
        );

        const std::vector<float> expected_pcm =
            read_float_binary(
                fixture_dir /
                entry.at("pcm_file").get<std::string>()
            );
        REQUIRE(mono.samples.size() == expected_pcm.size());

        const std::size_t edge = std::min<std::size_t>(
            mono.samples.size() / 8,
            2048
        );
        double signal_energy = 0.0;
        double error_energy = 0.0;
        for (std::size_t index = edge;
             index + edge < mono.samples.size();
             ++index) {
            const double expected = expected_pcm[index];
            const double error =
                static_cast<double>(mono.samples[index]) - expected;
            signal_energy += expected * expected;
            error_energy += error * error;
        }
        const double snr_db =
            10.0 *
            std::log10(
                signal_energy /
                std::max(error_energy, 1.0e-30)
            );

        const std::vector<float> expected_log_mel =
            read_float_binary(
                fixture_dir /
                entry.at("log_mel_file").get<std::string>()
            );
        const qwen3_asr::LogMelFeatures actual =
            frontend.extract(mono);
        REQUIRE(actual.data.size() == expected_log_mel.size());

        double maximum_log_mel_error = 0.0;
        double mean_log_mel_error = 0.0;
        for (std::size_t index = 0;
             index < expected_log_mel.size();
             ++index) {
            const double error = std::fabs(
                static_cast<double>(actual.data[index]) -
                static_cast<double>(expected_log_mel[index])
            );
            maximum_log_mel_error =
                std::max(maximum_log_mel_error, error);
            mean_log_mel_error += error;
        }
        mean_log_mel_error /= static_cast<double>(expected_log_mel.size());
        std::cout << "pipeline " << name
                  << " pcm_snr_db=" << snr_db
                  << " max_logmel_error=" << maximum_log_mel_error
                  << " mean_logmel_error=" << mean_log_mel_error
                  << '\n';

        if (entry.at("source_sample_rate").get<int>() == 16000) {
            CHECK(snr_db >= 90.0);
            CHECK(maximum_log_mel_error <= 5.0e-5);
        } else {
            CHECK(snr_db >= 30.0);
            CHECK(mean_log_mel_error <= 2.0e-3);
            CHECK(maximum_log_mel_error <= 0.5);
        }
    }
}

}  // namespace
