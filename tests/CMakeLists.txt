add_executable(
    qwen3-asr-tests
    test_aut_encoder.cpp
    test_ggml_session.cpp
    test_main.cpp
    test_qwen3_asr.cpp
    test_qwen3_lm.cpp
    test_sampler.cpp
)
qwen3_asr_configure_target(qwen3-asr-tests)
target_include_directories(
    qwen3-asr-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(qwen3-asr-tests PRIVATE qwen3_asr_core)
target_compile_definitions(
    qwen3-asr-tests
    PRIVATE
        QWEN3ASR_ENCODER_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-encoder-bf16.gguf"
        QWEN3ASR_LM_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-lm-bf16.gguf"
        QWEN3ASR_TEST_WAV_DIR="${PROJECT_SOURCE_DIR}/test_wavs"
        QWEN3ASR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)

add_test(NAME qwen3_asr.unit COMMAND qwen3-asr-tests)
set_tests_properties(
    qwen3_asr.unit
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-tests>"
        TIMEOUT 300
)

add_test(
    NAME qwen3_asr.cli_load_split
    COMMAND
        qwen3-asr
        --encoder
        "${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-encoder-bf16.gguf"
        --lm
        "${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-lm-bf16.gguf"
)

add_test(NAME qwen3_asr.cli_help COMMAND qwen3-asr --help)
set_tests_properties(
    qwen3_asr.cli_help
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr>"
        TIMEOUT 30
)

add_test(NAME qwen3_asr.cli_list_devices COMMAND qwen3-asr --list-devices)
set_tests_properties(
    qwen3_asr.cli_list_devices
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr>"
        TIMEOUT 30
)

find_package(Python3 COMPONENTS Interpreter QUIET)
set(QWEN3ASR_QUANTIZE_PYTHON "")
if(WIN32 AND EXISTS "${PROJECT_SOURCE_DIR}/.venv/Scripts/python.exe")
    set(
        QWEN3ASR_QUANTIZE_PYTHON
        "${PROJECT_SOURCE_DIR}/.venv/Scripts/python.exe"
    )
elseif(EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
    set(
        QWEN3ASR_QUANTIZE_PYTHON
        "${PROJECT_SOURCE_DIR}/.venv/bin/python"
    )
elseif(Python3_Interpreter_FOUND)
    set(QWEN3ASR_QUANTIZE_PYTHON "${Python3_EXECUTABLE}")
endif()
if(QWEN3ASR_QUANTIZE_PYTHON)
    execute_process(
        COMMAND
            "${QWEN3ASR_QUANTIZE_PYTHON}"
            -c
            "import gguf"
        RESULT_VARIABLE QWEN3ASR_GGUF_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()
if(QWEN3ASR_QUANTIZE_PYTHON AND
   QWEN3ASR_GGUF_IMPORT_RESULT EQUAL 0)
    add_test(
        NAME qwen3_asr.quantize_help
        COMMAND
            "${QWEN3ASR_QUANTIZE_PYTHON}"
            "${PROJECT_SOURCE_DIR}/quantize_gguf.py"
            --help
    )
    set_tests_properties(
        qwen3_asr.quantize_help
        PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            TIMEOUT 30
    )
endif()
set_tests_properties(
    qwen3_asr.cli_load_split
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr>"
        TIMEOUT 30
)

add_executable(
    qwen3-asr-audio-tests
    test_main.cpp
    test_audio_frontend.cpp
)
qwen3_asr_configure_target(qwen3-asr-audio-tests)
target_include_directories(
    qwen3-asr-audio-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(qwen3-asr-audio-tests PRIVATE qwen3_asr_audio)
target_compile_definitions(
    qwen3-asr-audio-tests
    PRIVATE
        QWEN3ASR_TEST_WAV_DIR="${PROJECT_SOURCE_DIR}/test_wavs"
        QWEN3ASR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)

add_test(NAME qwen3_asr.audio COMMAND qwen3-asr-audio-tests)
set_tests_properties(
    qwen3_asr.audio
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-audio-tests>"
        TIMEOUT 60
)

add_executable(
    qwen3-asr-inspect
    qwen3_asr_inspect.cpp
)
qwen3_asr_configure_target(qwen3-asr-inspect)
target_link_libraries(qwen3-asr-inspect PRIVATE qwen3_asr_model)

add_test(NAME qwen3_asr.inspect_help COMMAND qwen3-asr-inspect --help)
set_tests_properties(
    qwen3_asr.inspect_help
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-inspect>"
        TIMEOUT 30
)

set(
    QWEN3ASR_ENCODER_Q8_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-encoder-q8_0.gguf"
)
set(
    QWEN3ASR_LM_Q8_GGUF
    "${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-lm-q8_0.gguf"
)
if(EXISTS "${QWEN3ASR_ENCODER_Q8_GGUF}" AND
   EXISTS "${QWEN3ASR_LM_Q8_GGUF}")
    add_test(
        NAME qwen3_asr.q8_inspect
        COMMAND
            qwen3-asr-inspect
            --encoder
            "${QWEN3ASR_ENCODER_Q8_GGUF}"
            --lm
            "${QWEN3ASR_LM_Q8_GGUF}"
    )
    set_tests_properties(
        qwen3_asr.q8_inspect
        PROPERTIES
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-inspect>"
            TIMEOUT 30
    )
endif()

add_executable(
    qwen3-asr-audio-probe
    qwen3_asr_audio_probe.cpp
)
qwen3_asr_configure_target(qwen3-asr-audio-probe)
target_link_libraries(qwen3-asr-audio-probe PRIVATE qwen3_asr_audio)

add_test(
    NAME qwen3_asr.audio_probe_help
    COMMAND qwen3-asr-audio-probe --help
)
set_tests_properties(
    qwen3_asr.audio_probe_help
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-audio-probe>"
        TIMEOUT 30
)

add_executable(
    qwen3-asr-tokenizer-tests
    test_main.cpp
    test_text_tokenizer.cpp
)
qwen3_asr_configure_target(qwen3-asr-tokenizer-tests)
target_include_directories(
    qwen3-asr-tokenizer-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
    qwen3-asr-tokenizer-tests
    PRIVATE
        QWEN3ASR_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/Qwen3-ASR-0.6B"
        QWEN3ASR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
target_link_libraries(qwen3-asr-tokenizer-tests PRIVATE qwen3_asr_text)

add_test(NAME qwen3_asr.tokenizer COMMAND qwen3-asr-tokenizer-tests)
set_tests_properties(
    qwen3_asr.tokenizer
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-tokenizer-tests>"
        TIMEOUT 60
)

add_executable(
    qwen3-asr-prompt-tests
    test_main.cpp
    test_prompt_builder.cpp
)
qwen3_asr_configure_target(qwen3-asr-prompt-tests)
target_include_directories(
    qwen3-asr-prompt-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
    qwen3-asr-prompt-tests
    PRIVATE
        QWEN3ASR_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/Qwen3-ASR-0.6B"
        QWEN3ASR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
target_link_libraries(qwen3-asr-prompt-tests PRIVATE qwen3_asr_text)

add_test(NAME qwen3_asr.prompt COMMAND qwen3-asr-prompt-tests)
set_tests_properties(
    qwen3_asr.prompt
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-prompt-tests>"
        TIMEOUT 60
)

add_executable(
    qwen3-asr-gguf-tests
    test_main.cpp
    test_gguf_model.cpp
)
qwen3_asr_configure_target(qwen3-asr-gguf-tests)
target_compile_definitions(
    qwen3-asr-gguf-tests
    PRIVATE
        QWEN3ASR_ENCODER_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-encoder-bf16.gguf"
        QWEN3ASR_LM_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-lm-bf16.gguf"
        QWEN3ASR_TEST_WAV_DIR="${PROJECT_SOURCE_DIR}/test_wavs"
        QWEN3ASR_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata"
)
target_link_libraries(qwen3-asr-gguf-tests PRIVATE qwen3_asr_model)

add_test(NAME qwen3_asr.gguf COMMAND qwen3-asr-gguf-tests)
set_tests_properties(
    qwen3_asr.gguf
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-gguf-tests>"
        TIMEOUT 60
)

add_executable(
    qwen3-asr-e2e-regression
    qwen3_asr_e2e.cpp
)
qwen3_asr_configure_target(qwen3-asr-e2e-regression)
target_link_libraries(
    qwen3-asr-e2e-regression
    PRIVATE
        qwen3_asr_core
)
target_compile_definitions(
    qwen3-asr-e2e-regression
    PRIVATE
        QWEN3ASR_ENCODER_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-encoder-bf16.gguf"
        QWEN3ASR_LM_GGUF="${PROJECT_SOURCE_DIR}/models/qwen3-asr-0.6b-lm-bf16.gguf"
        QWEN3ASR_TEST_WAV_DIR="${PROJECT_SOURCE_DIR}/test_wavs"
)

if(QWEN3ASR_ENABLE_E2E_REGRESSION)
    add_test(
        NAME qwen3_asr.e2e
        COMMAND qwen3-asr-e2e-regression
    )
    set_tests_properties(
        qwen3_asr.e2e
        PROPERTIES
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:qwen3-asr-e2e-regression>"
            TIMEOUT 7200
            LABELS e2e
    )
endif()
