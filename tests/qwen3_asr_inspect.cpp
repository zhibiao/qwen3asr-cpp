#include <iostream>
#include <string>

#include "errors.h"
#include "gguf_model.h"

namespace {

void print_usage() {
    std::cout
        << "qwen3-asr-inspect\n\n"
        << "Usage:\n"
        << "  qwen3-asr-inspect <model.gguf>\n"
        << "  qwen3-asr-inspect --encoder <encoder.gguf> "
           "--lm <lm.gguf>\n";
}

int print_model(const qwen3_asr::GgufModel & model) {
    std::cout << "format_version\t" << model.format_version << '\n';
    std::cout << "name\t" << model.gf_name << '\n';
    std::cout << "family\t" << model.model_family << '\n';
    std::cout << "model_type\t" << model.model_type << '\n';
    std::cout << "architecture\t" << model.architecture << '\n';
    std::cout << "languages\t" << model.supported_languages.size()
              << '\n';
    if (!model.bundle_id.empty()) {
        std::cout << "bundle.id\t" << model.bundle_id << '\n';
    }
    std::cout << "weights.layout\t" << model.weight_layout << '\n';
    std::cout << "weights.dtype\t" << model.weight_dtype << '\n';
    std::cout << "weights.tensors\t"
              << model.weight_tensor_count << '\n';
    std::cout << "weights.audio_tensors\t"
              << model.audio_weight_tensor_count << '\n';
    std::cout << "weights.text_tensors\t"
              << model.text_weight_tensor_count << '\n';
    std::cout << "weights.lm_head_tensors\t"
              << model.lm_head_weight_tensor_count << '\n';
    std::cout << "weights.bytes\t"
              << model.weight_byte_count << '\n';
    std::cout << "audio.d_model\t" << model.audio.d_model << '\n';
    std::cout << "audio.encoder_layers\t"
              << model.audio.encoder_layers << '\n';
    std::cout << "audio.n_window\t" << model.audio.n_window << '\n';
    std::cout << "text.hidden_size\t" << model.text.hidden_size << '\n';
    std::cout << "text.layers\t" << model.text.layers << '\n';
    std::cout << "text.num_ctx\t" << model.text.num_ctx << '\n';
    std::cout << "preprocessor.sample_rate\t"
              << model.preprocessor.sample_rate << '\n';
    std::cout << "preprocessor.feature_size\t"
              << model.preprocessor.feature_size << '\n';
    std::cout << "preprocessor.n_fft\t"
              << model.preprocessor.n_fft << '\n';
    std::cout << "preprocessor.hop_length\t"
              << model.preprocessor.hop_length << '\n';
    std::cout << "mel_filters\t"
              << model.mel_filters.n_frequency_bins << 'x'
              << model.mel_filters.n_mel_bins << '\n';
    std::cout << "tokenizer.vocab\t"
              << model.tokenizer_data.vocab.size() << '\n';
    std::cout << "tokenizer.merges\t"
              << model.tokenizer_data.merges.size() << '\n';
    std::cout << "tokenizer.added_tokens\t"
              << model.tokenizer_data.added_tokens.size() << '\n';
    std::cout << "prompt.version\t" << model.prompt_version << '\n';
    return 0;
}

int inspect_model(const std::string & path) {
    return print_model(qwen3_asr::load_qwen3_asr_gguf(path));
}

int inspect_model_pair(
    const std::string & encoder_path,
    const std::string & lm_path
) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            encoder_path,
            lm_path
        );
    std::cout << "component\tencoder\n";
    print_model(pair.encoder);
    std::cout << "component\tlm\n";
    print_model(pair.lm);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 5) {
        print_usage();
        return 2;
    }

    const std::string command = argv[1];
    if (argc == 5 &&
        command == "--encoder" &&
        std::string(argv[3]) == "--lm") {
        try {
            return inspect_model_pair(argv[2], argv[4]);
        } catch (const qwen3_asr::Error & error) {
            std::cerr << "failed to inspect split GGUF: "
                      << error.what() << '\n';
            return 5;
        }
    }
    if (argc != 2) {
        print_usage();
        return 2;
    }
    if (command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }

    try {
        return inspect_model(command);
    } catch (const qwen3_asr::Error & error) {
        std::cerr << "failed to inspect GGUF: " << error.what() << '\n';
        return 5;
    }
}
