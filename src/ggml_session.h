#pragma once

#include <cstddef>

#include "ggml-backend.h"

#include "configs.h"

namespace qwen3_asr {

class GgmlSession {
public:
    explicit GgmlSession(const GgmlSessionConfig & config = {});
    ~GgmlSession();

    void initialize(const GgmlSessionConfig & config = {});
    void reset() noexcept;

    bool initialized() const noexcept {
        return backend_ != nullptr;
    }

    ggml_backend * native_backend() const noexcept {
        return backend_;
    }

    ggml_backend * cpu_backend() const noexcept {
        return cpu_backend_;
    }

    ggml_backend_sched_t create_scheduler(
        std::size_t graph_size
    ) const;

    std::size_t graph_size() const noexcept {
        return graph_size_;
    }

    std::string backend_name() const;

private:
    ggml_backend * backend_ = nullptr;
    ggml_backend * cpu_backend_ = nullptr;
    std::size_t graph_size_ = 4096;
};

}  // namespace qwen3_asr
