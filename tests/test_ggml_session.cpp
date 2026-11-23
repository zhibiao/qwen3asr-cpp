#include "test_framework.hpp"

#include <string>

#include "ggml.h"
#include "ggml_session.h"

namespace {

QWEN3ASR_TEST_CASE(ggml_session_initializes_cpu_backend) {
    qwen3_asr::GgmlSession session;
    CHECK(session.initialized());
    CHECK(session.native_backend() != nullptr);
    CHECK(session.cpu_backend() != nullptr);
    CHECK(session.native_backend() == session.cpu_backend());
    CHECK(!session.backend_name().empty());
}

QWEN3ASR_TEST_CASE(ggml_session_can_reset_and_reinitialize) {
    qwen3_asr::GgmlSession session;
    session.reset();
    CHECK(!session.initialized());
    CHECK(session.backend_name().empty());

    qwen3_asr::GgmlSessionConfig config;
    config.device = "CPU";
    session.initialize(config);
    CHECK(session.initialized());
}

QWEN3ASR_TEST_CASE(ggml_scheduler_instances_are_independent) {
    qwen3_asr::GgmlSession session;
    ggml_backend_sched_t first =
        session.create_scheduler(session.graph_size());
    ggml_backend_sched_t second =
        session.create_scheduler(session.graph_size());
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    CHECK(first != second);

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * first_context = ggml_init(params);
    ggml_context * second_context = ggml_init(params);
    REQUIRE(first_context != nullptr);
    REQUIRE(second_context != nullptr);

    ggml_tensor * first_input = ggml_new_tensor_1d(
        first_context,
        GGML_TYPE_F32,
        1
    );
    ggml_tensor * first_output = ggml_scale(
        first_context,
        first_input,
        2.0F
    );
    ggml_cgraph * first_graph = ggml_new_graph(first_context);
    ggml_build_forward_expand(first_graph, first_output);

    ggml_tensor * second_input = ggml_new_tensor_1d(
        second_context,
        GGML_TYPE_F32,
        1
    );
    ggml_tensor * second_output = ggml_scale(
        second_context,
        second_input,
        3.0F
    );
    ggml_cgraph * second_graph = ggml_new_graph(second_context);
    ggml_build_forward_expand(second_graph, second_output);

    REQUIRE(ggml_backend_sched_alloc_graph(first, first_graph));
    REQUIRE(ggml_backend_sched_alloc_graph(second, second_graph));
    const float first_value = 1.0F;
    const float second_value = 2.0F;
    ggml_backend_tensor_set(
        first_input,
        &first_value,
        0,
        sizeof(first_value)
    );
    ggml_backend_tensor_set(
        second_input,
        &second_value,
        0,
        sizeof(second_value)
    );
    REQUIRE(
        ggml_backend_sched_graph_compute(first, first_graph) ==
        GGML_STATUS_SUCCESS
    );
    REQUIRE(
        ggml_backend_sched_graph_compute(second, second_graph) ==
        GGML_STATUS_SUCCESS
    );

    ggml_backend_sched_reset(first);
    float second_result = 0.0F;
    ggml_backend_tensor_get(
        second_output,
        &second_result,
        0,
        sizeof(second_result)
    );
    CHECK(second_result == 6.0F);

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    ggml_free(first_context);
    ggml_free(second_context);
}

}  // namespace
