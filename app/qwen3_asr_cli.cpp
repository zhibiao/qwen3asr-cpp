#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include "errors.h"
#include "ggml-backend.h"
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
        << "  qwen3-asr --list-devices\n"
        << "  qwen3-asr --help\n\n"
        << "Without --audio, the command only loads and inspects the model.\n";
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:
            return "META";
    }
    return "UNKNOWN";
}

void list_devices() {
    ggml_backend_load_all();
    const std::size_t count = ggml_backend_dev_count();
    std::cout << "devices\t" << count << '\n';
    for (std::size_t index = 0; index < count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        std::cout << "device\t" << index
                  << "\tname=" << (props.name != nullptr ? props.name : "")
                  << "\ttype=" << device_type_name(props.type)
                  << "\tdescription="
                  << (props.description != nullptr ? props.description : "")
                  << "\tfree_bytes=" << props.memory_free
                  << "\ttotal_bytes=" << props.memory_total;
        if (props.device_id != nullptr) {
            std::cout << "\tdevice_id=" << props.device_id;
        }
        std::cout << '\n';
    }
}

std::string require_value(
    const std::vector<std::string> & arguments,
    std::size_t & index,
    const std::string & argument
) {
    if (index + 1 >= arguments.size()) {
        throw std::runtime_error("missing value for " + argument);
    }
    return arguments[++index];
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

CliOptions parse_arguments(const std::vector<std::string> & arguments) {
    CliOptions options;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string & argument = arguments[index];
        if (argument == "--encoder") {
            options.encoder = std::filesystem::u8path(
                require_value(arguments, index, argument)
            );
        } else if (argument == "--lm") {
            options.lm = std::filesystem::u8path(
                require_value(arguments, index, argument)
            );
        } else if (argument == "--audio") {
            options.audio = std::filesystem::u8path(
                require_value(arguments, index, argument)
            );
        } else if (argument == "--num-ctx") {
            options.num_ctx = parse_int(
                require_value(arguments, index, argument),
                argument
            );
        } else if (argument == "--device") {
            options.device = parse_device(
                require_value(arguments, index, argument)
            );
        } else if (argument == "--context") {
            options.generate.context =
                require_value(arguments, index, argument);
        } else if (argument == "--language") {
            options.generate.language =
                require_value(arguments, index, argument);
        } else if (argument == "--max-new-tokens") {
            options.generate.generation.max_new_tokens = parse_int(
                require_value(arguments, index, argument),
                argument
            );
        } else if (argument == "--temperature") {
            options.generate.generation.temperature = parse_float(
                require_value(arguments, index, argument),
                argument
            );
        } else if (argument == "--top-k") {
            options.generate.generation.top_k = parse_int(
                require_value(arguments, index, argument),
                argument
            );
        } else if (argument == "--top-p") {
            options.generate.generation.top_p = parse_float(
                require_value(arguments, index, argument),
                argument
            );
        } else if (argument == "--seed") {
            options.generate.generation.seed = parse_u64(
                require_value(arguments, index, argument),
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

int run(const std::vector<std::string> & arguments) {
    initialize_console_encoding();
    if (arguments.size() == 2 && arguments[1] == "--list-devices") {
        list_devices();
        return 0;
    }
    if (arguments.size() == 2 &&
        (arguments[1] == "--help" || arguments[1] == "-h")) {
        print_usage();
        return 0;
    }

    try {
        const CliOptions options = parse_arguments(arguments);
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

#if defined(_WIN32)

namespace {

std::string utf8_from_wide(const wchar_t * value) {
    if (value == nullptr) {
        return {};
    }
    const int length = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        value,
        -1,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    if (length <= 0) {
        throw std::runtime_error(
            "failed to convert command-line argument to UTF-8"
        );
    }
    std::string result(static_cast<std::size_t>(length), '\0');
    if (WideCharToMultiByte(
            CP_UTF8,
            WC_ERR_INVALID_CHARS,
            value,
            -1,
            result.data(),
            length,
            nullptr,
            nullptr
        ) <= 0) {
        throw std::runtime_error(
            "failed to convert command-line argument to UTF-8"
        );
    }
    result.resize(static_cast<std::size_t>(length - 1));
    return result;
}

}  // namespace

int wmain(int argc, wchar_t ** argv) {
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.push_back(utf8_from_wide(argv[index]));
        }
        return run(arguments);
    } catch (const std::exception & error) {
        std::cerr << "qwen3-asr argument error: " << error.what() << '\n';
        return 2;
    }
}

#else

int main(int argc, char ** argv) {
    try {
        return run(
            std::vector<std::string>(argv, argv + argc)
        );
    } catch (const std::exception & error) {
        std::cerr << "qwen3-asr argument error: " << error.what() << '\n';
        return 2;
    }
}

#endif
