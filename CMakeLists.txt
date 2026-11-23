cmake_minimum_required(VERSION 3.21)

set(CMAKE_POLICY_DEFAULT_CMP0194 NEW)

project(qwen3_asr_cpp VERSION 0.1.0 LANGUAGES C CXX)

if(MSVC AND CMAKE_CXX_COMPILER)
    set(
        CMAKE_ASM_COMPILER
        "${CMAKE_CXX_COMPILER}"
        CACHE FILEPATH
        "MSVC-compatible assembler compiler"
        FORCE
    )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

set(
    QWEN3ASR_MODEL_DIR
    "${PROJECT_SOURCE_DIR}/models"
    CACHE PATH
    "Directory containing Qwen3-ASR runtime models"
)
set(
    QWEN3ASR_TEST_DATA_DIR
    "${PROJECT_SOURCE_DIR}/testdata"
    CACHE PATH
    "Directory containing golden test fixtures"
)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")

if(BUILD_SHARED_LIBS)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

option(
    QWEN3ASR_ENABLE_CUDA
    "Build Qwen3-ASR with the GGML CUDA backend"
    OFF
)
option(
    QWEN3ASR_ENABLE_E2E_REGRESSION
    "Register the long end-to-end ASR regression with CTest"
    OFF
)
set(
    QWEN3ASR_CUDA_ARCHITECTURES
    "native"
    CACHE STRING
    "CUDA architectures: native or a semicolon-separated list"
)

if(QWEN3ASR_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${QWEN3ASR_CUDA_ARCHITECTURES}"
        CACHE STRING "" FORCE
    )
    message(STATUS "Qwen3-ASR CUDA backend enabled")
    message(
        STATUS
        "Qwen3-ASR CUDA architectures: ${QWEN3ASR_CUDA_ARCHITECTURES}"
    )
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
    message(STATUS "Qwen3-ASR CUDA backend disabled")
endif()

if(POLICY CMP0194)
    cmake_policy(SET CMP0194 NEW)
endif()

add_subdirectory(vendor/ggml-0.29.0)

add_library(
    qwen3_asr_miniaudio
    STATIC
    "${PROJECT_SOURCE_DIR}/vendor/miniaudio-0.11.26/miniaudio.c"
)
target_include_directories(
    qwen3_asr_miniaudio
    PUBLIC
        "${PROJECT_SOURCE_DIR}/vendor/miniaudio-0.11.26"
)
target_compile_definitions(
    qwen3_asr_miniaudio
    PRIVATE
        MA_NO_DEVICE_IO
)
set_target_properties(
    qwen3_asr_miniaudio
    PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
)

function(qwen3_asr_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_include_directories(
        ${target}
        PUBLIC
            "${PROJECT_SOURCE_DIR}/src"
        PRIVATE
            "${PROJECT_SOURCE_DIR}/vendor"
    )
    target_compile_definitions(
        ${target}
        PRIVATE
            UNI_ALGO_STATIC_DATA
    )
    if(MSVC)
        target_compile_options(
            ${target}
            PRIVATE
                /W4
                /permissive-
                /utf-8
        )
    else()
        target_compile_options(
            ${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()
endfunction()

add_library(
    qwen3_asr_audio
    STATIC
    "${PROJECT_SOURCE_DIR}/src/audio_frontend.cpp"
)
qwen3_asr_configure_target(qwen3_asr_audio)
target_link_libraries(qwen3_asr_audio PRIVATE qwen3_asr_miniaudio)

add_library(
    qwen3_asr_text
    STATIC
    "${PROJECT_SOURCE_DIR}/src/prompt_builder.cpp"
    "${PROJECT_SOURCE_DIR}/src/text_tokenizer.cpp"
)
qwen3_asr_configure_target(qwen3_asr_text)

add_library(
    qwen3_asr_model
    STATIC
    "${PROJECT_SOURCE_DIR}/src/gguf_model.cpp"
)
qwen3_asr_configure_target(qwen3_asr_model)
target_link_libraries(
    qwen3_asr_model
    PUBLIC
        ggml
        qwen3_asr_audio
        qwen3_asr_text
)

add_library(
    qwen3_asr_core
    STATIC
    "${PROJECT_SOURCE_DIR}/src/aut_encoder.cpp"
    "${PROJECT_SOURCE_DIR}/src/ggml_session.cpp"
    "${PROJECT_SOURCE_DIR}/src/qwen3_asr.cpp"
    "${PROJECT_SOURCE_DIR}/src/qwen3_lm.cpp"
    "${PROJECT_SOURCE_DIR}/src/sampler.cpp"
)
qwen3_asr_configure_target(qwen3_asr_core)
target_include_directories(
    qwen3_asr_core
    PRIVATE
        "${PROJECT_SOURCE_DIR}/vendor/ggml-0.29.0/include"
)
target_link_libraries(
    qwen3_asr_core
    PUBLIC
        ggml
        qwen3_asr_audio
        qwen3_asr_text
        qwen3_asr_model
)
set_target_properties(
    qwen3_asr_core
    PROPERTIES
        OUTPUT_NAME qwen3_asr
)

add_executable(
    qwen3-asr
    "${PROJECT_SOURCE_DIR}/app/qwen3_asr_cli.cpp"
)
qwen3_asr_configure_target(qwen3-asr)
target_link_libraries(qwen3-asr PRIVATE qwen3_asr_core)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
