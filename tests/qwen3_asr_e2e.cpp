#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <uni-algo-1.2.0/case.h>
#include <uni-algo-1.2.0/conv.h>
#include <uni-algo-1.2.0/norm.h>
#include <uni-algo-1.2.0/prop.h>

#include "errors.h"
#include "qwen3_asr.h"

namespace {

struct ReferenceCase {
    std::string file;
    std::string text;
    std::string language;
};

struct RegressionOptions {
    bool forced_language = false;
    std::string filter;
    int max_new_tokens = 1024;
    double cer_threshold = 0.35;
};

std::string expected_language(std::string_view file) {
    if (file == "fast1.wav" ||
        file == "raokouling.wav" ||
        file == "noise2.wav" ||
        file == "qiqiu1.wav") {
        return "Chinese";
    }
    if (file == "cantonese.wav") {
        return "Cantonese";
    }
    if (file == "ar1.wav") {
        return "Arabic";
    }
    if (file == "de.wav") {
        return "German";
    }
    if (file == "es1.wav") {
        return "Spanish";
    }
    if (file == "fr1.wav") {
        return "French";
    }
    if (file == "ja1.wav") {
        return "Japanese";
    }
    if (file == "ru1.wav") {
        return "Russian";
    }
    if (file == "f1_noise.wav" ||
        file == "noise1-en.wav" ||
        file == "rap1.wav") {
        return "English";
    }
    return {};
}

std::vector<ReferenceCase> load_references(
    const std::filesystem::path & path
) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "failed to open reference transcript: " + path.string()
        );
    }

    std::vector<ReferenceCase> references;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find(' ');
        if (separator == std::string::npos) {
            continue;
        }
        ReferenceCase reference;
        reference.file = line.substr(0, separator);
        reference.text = line.substr(separator + 1);
        reference.language = expected_language(reference.file);
        references.push_back(std::move(reference));
    }
    return references;
}

std::u32string normalized_codepoints(
    std::string_view text,
    bool preserve_spaces
) {
    const std::string lowered = una::cases::to_lowercase_utf8(text);
    const std::string normalized = una::norm::to_nfc_utf8(lowered);
    const std::u32string codepoints = una::utf8to32u(normalized);

    std::u32string output;
    output.reserve(codepoints.size());
    bool pending_space = false;
    for (char32_t codepoint : codepoints) {
        if (una::codepoint::is_alphanumeric(codepoint)) {
            if (preserve_spaces && pending_space && !output.empty()) {
                output.push_back(U' ');
            }
            pending_space = false;
            output.push_back(codepoint);
        } else if (preserve_spaces) {
            pending_space = true;
        }
    }
    return output;
}

std::size_t edit_distance(
    const std::u32string & expected,
    const std::u32string & actual
) {
    std::vector<std::size_t> previous(expected.size() + 1);
    std::vector<std::size_t> current(expected.size() + 1);
    for (std::size_t index = 0; index <= expected.size(); ++index) {
        previous[index] = index;
    }
    for (std::size_t row = 1; row <= actual.size(); ++row) {
        current[0] = row;
        for (std::size_t column = 1; column <= expected.size(); ++column) {
            current[column] = std::min({
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] +
                    (actual[row - 1] == expected[column - 1] ? 0 : 1),
            });
        }
        previous.swap(current);
    }
    return previous[expected.size()];
}

double character_error_rate(
    std::string_view expected,
    std::string_view actual
) {
    const std::u32string expected_chars =
        normalized_codepoints(expected, false);
    const std::u32string actual_chars =
        normalized_codepoints(actual, false);
    if (expected_chars.empty()) {
        return actual_chars.empty() ? 0.0 :
            std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(
        edit_distance(expected_chars, actual_chars)
    ) / static_cast<double>(expected_chars.size());
}

std::vector<std::u32string> split_words(const std::u32string & text) {
    std::vector<std::u32string> words;
    std::size_t start = 0;
    while (start < text.size()) {
        while (start < text.size() && text[start] == U' ') {
            ++start;
        }
        if (start >= text.size()) {
            break;
        }
        std::size_t end = text.find(U' ', start);
        if (end == std::u32string::npos) {
            end = text.size();
        }
        words.push_back(text.substr(start, end - start));
        start = end;
    }
    return words;
}

double word_error_rate(
    std::string_view expected,
    std::string_view actual
) {
    const std::vector<std::u32string> expected_words =
        split_words(normalized_codepoints(expected, true));
    const std::vector<std::u32string> actual_words =
        split_words(normalized_codepoints(actual, true));
    if (expected_words.empty()) {
        return actual_words.empty() ? 0.0 :
            std::numeric_limits<double>::infinity();
    }

    std::size_t matches = 0;
    std::vector<std::size_t> previous(expected_words.size() + 1);
    std::vector<std::size_t> current(expected_words.size() + 1);
    for (std::size_t index = 0; index <= expected_words.size(); ++index) {
        previous[index] = index;
    }
    for (std::size_t row = 1; row <= actual_words.size(); ++row) {
        current[0] = row;
        for (std::size_t column = 1;
             column <= expected_words.size();
             ++column) {
            current[column] = std::min({
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] +
                    (actual_words[row - 1] == expected_words[column - 1]
                         ? 0
                         : 1),
            });
        }
        previous.swap(current);
    }
    matches = previous[expected_words.size()];
    return static_cast<double>(matches) /
        static_cast<double>(expected_words.size());
}

RegressionOptions parse_options(int argc, char ** argv) {
    RegressionOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--forced-language") {
            options.forced_language = true;
        } else if (argument == "--filter") {
            if (++index >= argc) {
                throw std::runtime_error("--filter requires a value");
            }
            options.filter = argv[index];
        } else if (argument == "--max-new-tokens") {
            if (++index >= argc) {
                throw std::runtime_error(
                    "--max-new-tokens requires a value"
                );
            }
            options.max_new_tokens = std::stoi(argv[index]);
        } else if (argument == "--cer-threshold") {
            if (++index >= argc) {
                throw std::runtime_error(
                    "--cer-threshold requires a value"
                );
            }
            options.cer_threshold = std::stod(argv[index]);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const RegressionOptions options = parse_options(argc, argv);
        const std::filesystem::path wav_dir = QWEN3ASR_TEST_WAV_DIR;
        const std::vector<ReferenceCase> references =
            load_references(wav_dir / "transcript.txt");

        qwen3_asr::Qwen3AsrConfig config;
        config.encoder_gguf = QWEN3ASR_ENCODER_GGUF;
        config.lm_gguf = QWEN3ASR_LM_GGUF;
        qwen3_asr::Qwen3Asr asr(config);

        std::size_t checked = 0;
        std::size_t failed = 0;
        double cer_sum = 0.0;
        double wer_sum = 0.0;
        for (const ReferenceCase & reference : references) {
            if (!options.filter.empty() &&
                reference.file.find(options.filter) == std::string::npos) {
                continue;
            }
            if (options.forced_language && reference.language.empty()) {
                continue;
            }

            qwen3_asr::Qwen3AsrGenerateOptions generate_options;
            generate_options.generation.max_new_tokens =
                options.max_new_tokens;
            if (options.forced_language) {
                generate_options.language = reference.language;
            }
            const qwen3_asr::Qwen3AsrResult result = asr.generate(
                wav_dir / reference.file,
                generate_options
            );
            const double cer = character_error_rate(
                reference.text,
                result.text
            );
            const double wer = word_error_rate(
                reference.text,
                result.text
            );
            const bool language_ok =
                options.forced_language ||
                reference.language.empty() ||
                result.language == reference.language;
            const bool cer_ok = cer <= options.cer_threshold;
            const bool ok = language_ok && cer_ok;

            ++checked;
            cer_sum += cer;
            wer_sum += wer;
            if (!ok) {
                ++failed;
            }
            std::cout
                << (ok ? "PASS" : "FAIL")
                << '\t' << reference.file
                << "\tlanguage=" << result.language
                << "\texpected=" << reference.language
                << "\tcer=" << std::fixed << std::setprecision(4) << cer
                << "\twer=" << wer
                << "\taudio_s=" << result.performance.audio_seconds
                << "\trtf="
                << (result.performance.audio_seconds > 0.0
                        ? result.performance.total_ms /
                              (result.performance.audio_seconds * 1000.0)
                        : 0.0)
                << "\ttokens=" << result.token_ids.size()
                << std::endl;
            if (!ok) {
                std::cerr << "expected text: " << reference.text << '\n';
                std::cerr << "actual text:   " << result.text << '\n';
            }
        }

        if (checked == 0) {
            std::cerr << "no regression cases matched\n";
            return 2;
        }
        std::cout << "summary\tcases=" << checked
                  << "\tfailed=" << failed
                  << "\tmean_cer="
                  << cer_sum / static_cast<double>(checked)
                  << "\tmean_wer="
                  << wer_sum / static_cast<double>(checked)
                  << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const qwen3_asr::Error & error) {
        std::cerr << "Qwen3-ASR regression failed: "
                  << error.what() << '\n';
        return 5;
    } catch (const std::exception & error) {
        std::cerr << "regression argument error: "
                  << error.what() << '\n';
        return 2;
    }
}
