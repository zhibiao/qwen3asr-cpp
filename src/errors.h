#pragma once

#include <stdexcept>
#include <string>

namespace qwen3_asr {

enum class ErrorCode {
    None = 0,
    BackendUnavailable,
    BackendInitFailed,
    BackendComputeFailed,
    InvalidArgument,
    UnsupportedAudioFormat,
    AudioLoadFailed,
    InvalidAudioData,
    ResampleFailed,
    AudioTooLong,
    ContextLengthExceeded,
    ModelLoadFailed,
    InferenceFailed,
};

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string & message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

}  // namespace qwen3_asr
