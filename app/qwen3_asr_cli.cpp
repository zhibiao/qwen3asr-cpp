#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include "errors.h"
#include "qwen3_asr.h"

namespace {

void initialize_console_encoding() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
}

struct CliOptions {
    std::filesystem::path encoder;
    std::filesystem::path lm;
    std::optional<std::filesystem::path> audio;
    std::optional<int> num_ctx;
    std::string device = "CPU";
    qwen3_asr::Qwen3AsrGenerateOptions generate;
};

void print_usage() {
    std::cout
        << "qwen3-asr\n\n"
        << "Usage:\n"
        << "  qwen3-asr --encoder <encoder.gguf> --lm <lm.gguf>\n"
        << "  qwen3-asr --encoder <encoder.gguf> --lm <lm.gguf> "
           "--audio <audio-file>\n"
        << "            [--device cpu|cuda|CUDA0]\n"
        << "            [--context <text>] [--language <name>]\n"
        << "            [--num-ctx <count>]\n"
        << "            [--max-new-tokens <count>] "
           "[--temperature <value>]\n"
        << "            [--top-k <count>] [--top-p <value>] "
           "[--seed <value>]\n"
        << "  qwen3-asr --help\n\n"
        << "Without --audio, the command only loads and inspects the model.\n";
}

std::string require_value(
    int argc,
    char ** argv,
    int & index,
    const std::string & argument
) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + argument);
    }
    return argv[++index];
}

int parse_int(const std::string & value, const std::string & argument) {
    std::size_t consumed = 0;
    try {
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            throw std::runtime_error("out of range");
        }
        return static_cast<int>(parsed);
    } catch (...) {
        throw std::runtime_error("invalid integer for " + argument + ": " + value);
    }
}

std::uint64_t parse_u64(
    const std::string & value,
    const std::string & argument
) {
    std::size_t consumed = 0;
    try {
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error("invalid suffix");
        }
        return static_cast<std::uint64_t>(parsed);
    } catch (...) {
        throw std::runtime_error(
            "invalid unsigned integer for " + argument + ": " + value
        );
    }
}

float parse_float(const std::string & value, const std::string & argument) {
    std::size_t consumed = 0;
    try {
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) {
            throw std::runtime_error("invalid number");
        }
        return static_cast<float>(parsed);
    } catch (...) {
        throw std::runtime_error("invalid number for " + argument + ": " + value);
    }
}

std::string parse_device(const std::string & value) {
    if (value == "cpu" || value == "CPU") {
        return "CPU";
    }
    if (value == "cuda" || value == "gpu") {
        return "CUDA0";
    }
    return value;
}

CliOptions parse_arguments(int argc, char ** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--encoder") {
            options.encoder = require_value(argc, argv, index, argument);
        } else if (argument == "--lm") {
            options.lm = require_value(argc, argv, index, argument);
        } else if (argument == "--audio") {
            options.audio = require_value(argc, argv, index, argument);
        } else if (argument == "--num-ctx") {
            options.num_ctx = parse_int(
                require_value(argc, argv, index, argument),
                argument
            );
        } else if (argument == "--device") {
            options.device = parse_device(
                require_value(argc, argv, index, argument)
            );
        } else if (argument == "--context") {
            options.generate.context =
                require_value(argc, argv, index, argument);
        } else if (argument == "--language") {
            options.generate.language =
                require_value(argc, argv, index, argument);
        } else if (argument == "--max-new-tokens") {
            options.generate.generation.max_new_tokens = parse_int(
                require_value(argc, argv, index, argument),
                argument
            );
        } else if (argument == "--temperature") {
            options.generate.generation.temperature = parse_float(
                require_value(argc, argv, index, argument),
                argument
            );
        } else if (argument == "--top-k") {
            options.generate.generation.top_k = parse_int(
                require_value(argc, argv, index, argument),
                argument
            );
        } else if (argument == "--top-p") {
            options.generate.generation.top_p = parse_float(
                require_value(argc, argv, index, argument),
                argument
            );
        } else if (argument == "--seed") {
            options.generate.generation.seed = parse_u64(
                require_value(argc, argv, index, argument),
                argument
            );
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.encoder.empty() || options.lm.empty()) {
        throw std::runtime_error("--encoder and --lm are required");
    }
    return options;
}

void print_model_info(
    const qwen3_asr::Qwen3Asr & asr,
    double load_ms
) {
    std::cout << "backend\t" << asr.ggml_session().backend_name() << '\n';
    std::cout << "encoder.weights\t"
              << asr.encoder_gguf().weight_tensor_count << '\n';
    std::cout << "lm.weights\t" << asr.lm_gguf().weight_tensor_count << '\n';
    std::cout << "bundle\t" << asr.encoder_gguf().bundle_id << '\n';
    std::cout << "timing.model_load_ms\t" << load_ms << '\n';
}

void print_result(
    const qwen3_asr::Qwen3AsrResult & result
) {
    std::cout << "language\t" << result.language << '\n';
    std::cout << "text\t" << result.text << '\n';
    std::cout << "tokens.generated\t" << result.token_ids.size() << '\n';
    std::cout << "tokens.audio\t" << result.audio_token_count << '\n';
    std::cout << "tokens.prompt\t"
              << result.lm_generation.prompt_tokens << '\n';
    std::cout << "decode.calls\t"
              << result.lm_generation.decode_calls << '\n';
    std::cout << "audio.seconds\t"
              << result.performance.audio_seconds << '\n';
    std::cout << "timing.model_load_ms\t"
              << result.performance.model_load_ms << '\n';
    std::cout << "timing.audio_load_ms\t"
              << result.performance.audio_load_ms << '\n';
    std::cout << "timing.resample_ms\t"
              << result.performance.resample_ms << '\n';
    std::cout << "timing.frontend_ms\t"
              << result.performance.frontend_ms << '\n';
    std::cout << "timing.encoder_ms\t"
              << result.performance.encoder_ms << '\n';
    std::cout << "timing.prefill_ms\t"
              << result.lm_generation.prefill_ms << '\n';
    std::cout << "timing.decode_ms\t"
              << result.lm_generation.decode_ms << '\n';
    std::cout << "timing.generation_ms\t"
              << result.performance.generation_ms << '\n';
    std::cout << "timing.total_ms\t" << result.performance.total_ms << '\n';
    if (result.performance.audio_seconds > 0.0) {
        std::cout << "rtf\t"
                  << result.performance.total_ms /
                         (result.performance.audio_seconds * 1000.0)
                  << '\n';
    }
}

}  // namespace

int main(int argc, char ** argv) {
    initialize_console_encoding();
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" ||
         std::string(argv[1]) == "-h")) {
        print_usage();
        return 0;
    }

    try {
        const CliOptions options = parse_arguments(argc, argv);
        qwen3_asr::Qwen3AsrConfig config;
        config.encoder_gguf = options.encoder;
        config.lm_gguf = options.lm;
        config.text_num_ctx = options.num_ctx;
        config.session.device = options.device;
        qwen3_asr::Qwen3Asr asr(config);

        if (!options.audio.has_value()) {
            print_model_info(asr, asr.model_load_ms());
            return 0;
        }

        const qwen3_asr::Qwen3AsrResult result =
            asr.generate(*options.audio, options.generate);
        print_result(result);
        return 0;
    } catch (const qwen3_asr::Error & error) {
        std::cerr << "qwen3-asr failed: " << error.what() << '\n';
        return 5;
    } catch (const std::exception & error) {
        std::cerr << "qwen3-asr argument error: " << error.what() << '\n';
        print_usage();
        return 2;
    }
}
