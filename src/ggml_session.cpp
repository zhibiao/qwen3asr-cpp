#include "ggml_session.h"

#include <cstring>
#include <mutex>
#include "ggml-backend.h"

#include "errors.h"

namespace qwen3_asr {
namespace {

std::once_flag g_backends_once;

}  // namespace

GgmlSession::GgmlSession(const GgmlSessionConfig & config) {
    initialize(config);
}

GgmlSession::~GgmlSession() {
    reset();
}

void GgmlSession::initialize(const GgmlSessionConfig & config) {
    std::call_once(g_backends_once, []() {
        ggml_backend_load_all();
    });

    ggml_backend_t backend = config.device.empty()
        ? ggml_backend_init_best()
        : ggml_backend_init_by_name(
            config.device.c_str(),
            config.params.empty() ? nullptr : config.params.c_str()
        );
    if (backend == nullptr) {
        throw Error(
            ErrorCode::BackendInitFailed,
            config.device.empty()
                ? "failed to initialize the best ggml backend"
                : "failed to initialize ggml backend device: " +
                    config.device
        );
    }

    ggml_backend_t cpu_backend = nullptr;
    const char * backend_name = ggml_backend_name(backend);
    if (backend_name != nullptr &&
        std::strcmp(backend_name, "CPU") != 0) {
        cpu_backend = ggml_backend_init_by_name("CPU", nullptr);
        if (cpu_backend == nullptr) {
            ggml_backend_free(backend);
            throw Error(
                ErrorCode::BackendInitFailed,
                "failed to initialize ggml CPU fallback"
            );
        }
    } else {
        cpu_backend = backend;
    }

    reset();
    backend_ = backend;
    cpu_backend_ = cpu_backend;
    graph_size_ = static_cast<std::size_t>(config.graph_size);
}

void GgmlSession::reset() noexcept {
    if (cpu_backend_ != nullptr && cpu_backend_ != backend_) {
        ggml_backend_free(cpu_backend_);
    }
    cpu_backend_ = nullptr;
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    graph_size_ = 4096;
}

ggml_backend_sched_t GgmlSession::create_scheduler(
    std::size_t graph_size
) const {
    if (backend_ == nullptr || cpu_backend_ == nullptr) {
        return nullptr;
    }

    ggml_backend_t backends[2] = {backend_, cpu_backend_};
    ggml_backend_buffer_type_t buffer_types[2] = {nullptr, nullptr};
    const int count = backend_ == cpu_backend_ ? 1 : 2;
    buffer_types[0] = ggml_backend_get_default_buffer_type(backend_);
    if (count == 2) {
        ggml_backend_dev_t device = ggml_backend_get_device(backend_);
        ggml_backend_buffer_type_t host =
            device == nullptr
                ? nullptr
                : ggml_backend_dev_host_buffer_type(device);
        buffer_types[1] = host != nullptr
            ? host
            : ggml_backend_get_default_buffer_type(cpu_backend_);
    }

    return ggml_backend_sched_new(
        backends,
        buffer_types,
        count,
        graph_size,
        false,
        true
    );
}

std::string GgmlSession::backend_name() const {
    if (backend_ == nullptr) {
        return {};
    }
    const char * name = ggml_backend_name(backend_);
    return name != nullptr ? name : std::string{};
}

}  // namespace qwen3_asr
