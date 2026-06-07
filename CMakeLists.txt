cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static libs are linked into the qwen3tts shared lib (C API), so they must be PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
if(EXISTS "${GGML_DIR}/build-static/src/libggml.a")
    set(GGML_DEFAULT_BUILD_DIR "${GGML_DIR}/build-static")
else()
    set(GGML_DEFAULT_BUILD_DIR "${GGML_DIR}/build")
endif()
set(GGML_BUILD_DIR "${GGML_DEFAULT_BUILD_DIR}" CACHE PATH "Path to a configured GGML build directory")
option(QWEN3_TTS_STATIC_GGML "Link GGML static archives into qwen3-tts binaries" ON)
option(QWEN3_TTS_BUILD_SHARED_API "Build qwen3tts shared library C API" OFF)

# Detect GGML GPU backends in the separately built ggml tree.
if(APPLE AND (EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib"
        OR EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.a"))
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

set(GGML_VULKAN_LIB_DIR "${GGML_BUILD_DIR}/src/ggml-vulkan")
if(EXISTS "${GGML_VULKAN_LIB_DIR}/libggml-vulkan.so"
        OR EXISTS "${GGML_VULKAN_LIB_DIR}/libggml-vulkan.dylib"
        OR EXISTS "${GGML_VULKAN_LIB_DIR}/libggml-vulkan.a"
        OR EXISTS "${GGML_VULKAN_LIB_DIR}/libggml-vulkan.dll.a"
        OR EXISTS "${GGML_VULKAN_LIB_DIR}/ggml-vulkan.dll"
        OR EXISTS "${GGML_VULKAN_LIB_DIR}/ggml-vulkan.lib"
        OR EXISTS "${GGML_BUILD_DIR}/bin/ggml-vulkan.dll")
    set(GGML_HAS_VULKAN ON)
    message(STATUS "GGML Vulkan backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_VULKAN OFF)
endif()

set(GGML_BACKEND_LINK_DIRS
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/bin
)
if(GGML_HAS_METAL)
    list(APPEND GGML_BACKEND_LINK_DIRS ${GGML_BUILD_DIR}/src/ggml-metal)
endif()
if(GGML_HAS_VULKAN)
    list(APPEND GGML_BACKEND_LINK_DIRS ${GGML_VULKAN_LIB_DIR})
endif()
if(QWEN3_TTS_STATIC_GGML)
    foreach(required_ggml_archive
            ${GGML_BUILD_DIR}/src/libggml.a
            ${GGML_BUILD_DIR}/src/libggml-cpu.a
            ${GGML_BUILD_DIR}/src/libggml-base.a)
        if(NOT EXISTS "${required_ggml_archive}")
            message(FATAL_ERROR "Static GGML archive not found: ${required_ggml_archive}. Build GGML with -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON or set GGML_BUILD_DIR to a static GGML build.")
        endif()
    endforeach()
    set(GGML_BACKEND_LIBS
        ${GGML_BUILD_DIR}/src/libggml.a
        ${GGML_BUILD_DIR}/src/libggml-cpu.a
    )
    if(EXISTS "${GGML_BUILD_DIR}/src/ggml-blas/libggml-blas.a")
        list(APPEND GGML_BACKEND_LIBS
            ${GGML_BUILD_DIR}/src/ggml-blas/libggml-blas.a
        )
        if(APPLE)
            list(APPEND GGML_BACKEND_LIBS "-framework Accelerate")
        endif()
    endif()
    if(GGML_HAS_METAL)
        list(APPEND GGML_BACKEND_LIBS
            ${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.a
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
        )
    endif()
    if(GGML_HAS_VULKAN)
        list(APPEND GGML_BACKEND_LIBS ${GGML_BUILD_DIR}/src/ggml-vulkan/libggml-vulkan.a)
    endif()
    list(APPEND GGML_BACKEND_LIBS
        ${GGML_BUILD_DIR}/src/libggml-base.a
        Threads::Threads
    )
    if(UNIX)
        list(APPEND GGML_BACKEND_LIBS m)
    endif()
else()
    set(GGML_BACKEND_LIBS
        ggml
        ggml-base
        ggml-cpu
        Threads::Threads
    )
    if(GGML_HAS_METAL)
        list(APPEND GGML_BACKEND_LIBS ggml-metal "-framework Metal" "-framework MetalKit")
    endif()
    if(GGML_HAS_VULKAN)
        list(APPEND GGML_BACKEND_LIBS ggml-vulkan)
    endif()
endif()

function(target_link_ggml_backends target_name)
    target_link_directories(${target_name} PUBLIC ${GGML_BACKEND_LINK_DIRS})
    target_link_libraries(${target_name} PUBLIC ${GGML_BACKEND_LIBS})
endfunction()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_ggml_backends(text_tokenizer)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_ggml_backends(tts_transformer)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_ggml_backends(audio_tokenizer_encoder)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_ggml_backends(audio_tokenizer_decoder)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)

# Shared library with C API (for Nim FFI)
if(QWEN3_TTS_BUILD_SHARED_API)
    add_library(qwen3tts_shared SHARED
        src/qwen3tts_c_api.cpp
    )
    set_target_properties(qwen3tts_shared PROPERTIES
        OUTPUT_NAME "qwen3tts"
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
    )
    target_include_directories(qwen3tts_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(qwen3tts_shared PRIVATE
        qwen3_tts
    )
    if(APPLE)
        target_link_libraries(qwen3tts_shared PRIVATE
            "-framework Foundation"
            objc
        )
    endif()
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Persistent binary-framed worker executable
add_executable(qwen3-tts-worker
    src/worker_main.cpp
)
target_link_libraries(qwen3-tts-worker PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Install targets
set(QWEN3_TTS_INSTALL_TARGETS
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    qwen3_tts
    qwen3-tts-cli
    qwen3-tts-worker
)
if(QWEN3_TTS_BUILD_SHARED_API)
    list(APPEND QWEN3_TTS_INSTALL_TARGETS qwen3tts_shared)
endif()
install(TARGETS ${QWEN3_TTS_INSTALL_TARGETS}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
