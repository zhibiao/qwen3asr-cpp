#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "errors.h"
#include "audio_frontend.h"

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  qwen3-asr-audio-probe <audio-file>\n"
        << "      --resample <rate>\n"
        << "      --dump-pcm <float32-file>\n\n"
        << "The probe never truncates or splits the input audio.\n";
}

bool parse_rate(const std::string & text, std::uint32_t & value) {
    try {
        const unsigned long parsed = std::stoul(text);
        if (parsed == 0 ||
            parsed > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        value = static_cast<std::uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

void dump_pcm(
    const std::filesystem::path & path,
    const std::vector<float> & samples
) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw qwen3_asr::Error(
            qwen3_asr::ErrorCode::AudioLoadFailed,
            "failed to open PCM output file: " + path.string()
        );
    }

    output.write(
        reinterpret_cast<const char *>(samples.data()),
        static_cast<std::streamsize>(samples.size() * sizeof(float))
    );
    if (!output) {
        throw qwen3_asr::Error(
            qwen3_asr::ErrorCode::AudioLoadFailed,
            "failed to write PCM output file: " + path.string()
        );
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage();
        return 0;
    }

    const std::filesystem::path input_path = argv[1];
    std::uint32_t target_rate = qwen3_asr::kQwen3AsrSampleRate;
    std::filesystem::path dump_path;

    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--resample") {
            if (index + 1 >= argc ||
                !parse_rate(argv[++index], target_rate)) {
                std::cerr << "invalid sample rate\n";
                return 2;
            }
        } else if (argument == "--dump-pcm") {
            if (index + 1 >= argc) {
                std::cerr << "missing PCM output path\n";
                return 2;
            }
            dump_path = argv[++index];
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return 2;
        }
    }

    try {
        const qwen3_asr::AudioBuffer decoded =
            qwen3_asr::AudioLoader::load(input_path);
        qwen3_asr::MonoAudio mono =
            qwen3_asr::AudioProcessor::prepare_mono(decoded);
        mono = qwen3_asr::AudioResampler::resample(mono, target_rate);
        if (!dump_path.empty()) {
            dump_pcm(dump_path, mono.samples);
        }

        const double duration =
            decoded.sample_rate == 0
                ? 0.0
                : static_cast<double>(decoded.frame_count()) /
                      decoded.sample_rate;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "input_sample_rate: " << decoded.sample_rate << '\n';
        std::cout << "input_channels: " << decoded.channels << '\n';
        std::cout << "input_frames: " << decoded.frame_count() << '\n';
        std::cout << "duration_seconds: " << duration << '\n';
        std::cout << "output_sample_rate: " << mono.sample_rate << '\n';
        std::cout << "output_samples: " << mono.samples.size() << '\n';
    } catch (const qwen3_asr::Error & error) {
        std::cerr << "audio probe failed: " << error.what() << '\n';
        return 3;
    }

    return 0;
}
