# pibot integration & Vulkan port notes

This is a **fork** of `qwen3-tts.cpp` used by [pibot](https://github.com/badlogic/pibot)
as a C++/GGML/Metal Qwen3-TTS engine. This file documents everything needed to
(a) understand what we changed, (b) build/run it, and (c) pick up the **Vulkan
port** on a Linux/Windows machine in a fresh session.

Do not read the upstream `README.md` / `OPTIMIZATION.md` for our changes — they
describe the original project. This file is the source of truth for the pibot fork.

---

## 1. Repository layout (3-level nested submodules)

```
badlogic/pibot                      (parent app)
└─ native/qwen3-tts.cpp             submodule, fork of the qwen3-tts.cpp project
   branch: cleanup-metal-no-coreml  remote: github.com/badlogic/qwen3-tts.cpp
   └─ ggml                          submodule, fork of ggml-org/ggml
      branch: qwen3-tts-metal-pad   remote: github.com/badlogic/ggml
                                    (upstream remote name: `upstream` -> ggml-org/ggml)
```

Current commits at time of writing:

```
ggml  qwen3-tts-metal-pad
  6b21cf3a metal: parallelize conv_transpose_1d across output positions
  6493e113 Add Metal PAD left padding support
  1e33fed3 (upstream base) ggml : bump version to 0.13.1

qwen  cleanup-metal-no-coreml
  ... Run Qwen3-TTS decoder on Metal with fast conv_transpose
  ... Point ggml submodule at badlogic fork
```

Compare-against-upstream URLs:
- ggml:  https://github.com/badlogic/ggml/compare/1e33fed3...qwen3-tts-metal-pad
- qwen:  https://github.com/badlogic/qwen3-tts.cpp/compare/main...cleanup-metal-no-coreml

---

## 2. What this engine does

Qwen3-TTS pipeline, all in C++/GGML:
- **Text tokenizer** (`src/text_tokenizer.cpp`)
- **Talker / TTS transformer** (`src/tts_transformer.cpp`) — autoregressive; per
  frame predicts codebook 0, then a small "code predictor" predicts codebooks
  1-15. Conditioning is **x-vector (speaker embedding) only**. ICL is NOT
  implemented (see Open Items).
- **Audio encoder** (`src/audio_tokenizer_encoder.cpp`) — reference WAV -> speaker
  embedding (and could produce codec tokens for ICL).
- **Decoder / vocoder** (`src/audio_tokenizer_decoder.cpp`) — codec frames -> 24kHz
  waveform. Pre-transformer (causal attention) + ConvNeXt upsample blocks +
  decoder blocks (conv_transpose_1d upsample + residual conv1d + Snake).
- Orchestration: `src/qwen3_tts.cpp` / `.h`.

Two GGUF files are required in the model dir:
```
qwen3-tts-0.6b-q8_0.gguf        talker + encoder + decoder weights (Q8_0)
qwen3-tts-tokenizer-f16.gguf    tokenizer + decoder/vocoder weights (F16)
```
HF repo: `badlogicgames/qwen3-tts-0.6b-q8_0-gguf` (auto-downloaded by the pibot
server — see below).

Model locations:
- **pibot server (worker):** `~/models/qwen3-tts-0.6b-q8_0-gguf/` — auto-downloaded
  from HF on first run by `src/server/tts.ts` (`ensureHuggingFaceModel`), same
  convention as the rust/python workers. Override with `QWEN3_TTS_CPP_MODEL_PATH`
  / `QWEN3_TTS_CPP_MODEL_REPO`.
- **local CLI / benchmark / dev:** this repo's `models/` dir (gitignored). The CLI
  takes `-m <dir>`; the benchmark points at `build/qwen3-tts-cli` + `models/`.
The C++ loader (`load_models`) expects the two filenames above directly inside the
model dir (prefers `qwen3-tts-0.6b-q8_0.gguf`, falls back to `...-f16.gguf`).

---

## 3. What we changed in the fork

### ggml fork (badlogic/ggml, branch qwen3-tts-metal-pad) — Metal-only patches
1. **`Add Metal PAD left padding support`** — the decoder uses `ggml_pad_ext` with
   left-only (causal) padding; upstream Metal `kernel_pad` ignored lp/rp offsets.
   Files: `ggml-metal.metal` (kernel_pad), `ggml-metal-impl.h` (kargs lp0..lp3),
   `ggml-metal-ops.cpp` (set lp), `ggml-metal-device.m` (supports check).
2. **`metal: parallelize conv_transpose_1d`** — THE big win. The upstream Metal
   `kernel_conv_transpose_1d` dispatched `{OL, OC, 1}` workgroups with **1 thread
   each** (31/32 SIMD lanes idle). Rewrote to many-threads-per-threadgroup over
   the output length. Files: `ggml-metal.metal`, `ggml-metal-impl.h` (OC/OL),
   `ggml-metal-ops.cpp` (dispatch).

### qwen fork
- `src/worker_main.cpp` (NEW) — persistent binary-framed worker (see §6).
- `src/qwen3_tts.{h,cpp}` — added `synthesize_streaming_with_embedding()` (streams
  by decoding the causal code prefix every N frames, emits only new samples).
- `src/tts_transformer.{h,cpp}` — added a per-frame callback to `generate()`.
- `CMakeLists.txt` — added `qwen3-tts-worker` target.
- `src/gguf_loader.cpp` — on macOS sets `GGML_METAL_TENSOR_DISABLE=1` by default
  (see §5).

---

## 4. Performance findings (Metal, M5 Max, Q8_0, ~18s utterance)

```
generate (talker): ~4.6 s
decode (vocoder):  ~5.7 s  ->  ~0.7 s   after conv_transpose fix
total:             ~10 s   ->  ~5.4 s   (3.36x realtime)
```
Decode was almost entirely `conv_transpose_1d` (~5.1s of 5.7s) due to the 1-thread
occupancy bug. After the fix decode is ~0.7s; the talker (`generate`) is now the
larger component. MLX reference: generate 4.1s / decode 0.3s / total 4.5s.

Key diagnostic detail: the residual convs lower to `im2col + mul_mat`; on Metal the
matmul is cheap (~4ms) and im2col dominates the *conv* time, but the whole conv
stack is only ~0.23s — it was never the bottleneck. Snake (sin/exp) is ~0.08s.
conv_transpose was the whole story.

---

## 5. Important runtime env vars

- `GGML_METAL_TENSOR_DISABLE=1` — REQUIRED on M5/Metal-4 for correct *talker*
  output. The newer Metal Tensor-API matmul path produces wrong codec generation
  (truncated audio). `src/gguf_loader.cpp::configure_metal_backend_defaults()` sets
  this automatically on macOS, so you normally don't need to export it.
- `QWEN3_TTS_USE_COREML=0` — only relevant if built WITH CoreML. The pibot `build/`
  dir is configured `-DQWEN3_TTS_COREML=OFF`, so CoreML is not compiled and this is
  a no-op there.
- `QWEN3_TTS_PROFILE_DECODER=1` — prints decoder timing (frames/compute ms).

---

## 6. The worker protocol (phases 1 & 2 — DONE)

`qwen3-tts-worker --serve` speaks the exact pibot binary frame protocol (same as
the Rust/MLX worker and `src/server/tts.ts`):

Frame = `uint8 type, uint32 request_id, uint32 payload_len` (LE) + payload.
- In:  SPEAK=1 (payload=utf8 text), CANCEL=2, SHUTDOWN=3
- Out: READY=1, AUDIO_START=2 (payload=uint32 sample_rate), AUDIO_CHUNK=3
  (payload=int16 PCM LE), AUDIO_DONE=4, ERROR=5 (payload=utf8)

Details:
- The real stdout is `dup()`'d for the protocol, then stdout is redirected to
  stderr so library/log noise can't corrupt the stream. All logs/JSON go to stderr.
- Conditioning: x-vector. At load it reads `--ref-audio`, extracts the speaker
  embedding once, and reuses it for every SPEAK.
- Streaming (phase 2): generation is autoregressive; every `--streaming-chunk-size`
  frames (default 16) the vocoder decodes the growing **causal** code prefix and
  emits only the newly produced samples. Verified bit-equivalent to a single full
  decode (relRMSdiff 0.0003) — the decoder is causal so prefix re-decode is stable.
  TTFA ~0.47s measured.
- PCM: leading-silence trim + 40ms preroll, blocksize-quantized int16, linear
  resample to `--output-sample-rate` if != 24000.

Worker flags (Rust/Python-worker compatible; extras accepted-and-ignored):
`--serve -m/--model-name <dir> --ref-audio <wav> --language <de|en|...>
--output-sample-rate --temperature --top-k --max-tokens --blocksize
--streaming-chunk-size`. Ignored: `--ref-text(-file) --seed --top-p
--repetition-penalty --speaker --instruct --mlx-quantization --xvec-only ...`.

One-shot mode (no `--serve`): `... -t "text" -o out.wav`.

---

## 7. Build (current: macOS / Metal)

ggml is built SEPARATELY into `ggml/build` as **shared** libs, then the qwen CMake
links the prebuilt libs. (This is why the worker has `@rpath/libggml*.dylib` deps —
NOT a static binary yet; see Open Items.)

```bash
cd native/qwen3-tts.cpp
# 1. build ggml (Metal)
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build ggml/build -j

# 2. build qwen worker + cli (CoreML off)
cmake -S . -B build -DQWEN3_TTS_COREML=OFF
cmake --build build --target qwen3-tts-worker qwen3-tts-cli -j
```

Or from the pibot root: `npm run build:tts-cpp` (does submodules + both steps,
CoreML off, but currently only configures the qwen step; ensure ggml/build exists).

> NOTE: `npm run build:tts-cpp` configures/builds the qwen project but assumes
> `ggml/build` is already built. If starting clean, run the ggml step above first.
> TODO: fold the ggml build into the script.

---

## 8. Vulkan port — how to pick this up (Linux/Windows)

### 8.1 Why it should "just work"
We investigated the three decoder ops that needed Metal-specific work; on Vulkan
they are already fine (source-read; NOT yet tested on real hardware):

1. **conv_transpose_1d** — Vulkan shader
   `ggml/src/ggml-vulkan/vulkan-shaders/conv_transpose_1d.comp` dispatches
   `{Cout,1,1}` workgroups with `local_size_x=128`, cooperating over the input
   length with a shared-memory accumulation window. It does NOT have the Metal
   1-thread-per-output bug. No port of our Metal fix is needed.
2. **PAD left-padding** — `vulkan-shaders/pad.comp` already has full `lp0..lp3` /
   `rp0..rp3` (and circular) support upstream. No patch needed.
3. **DIAG_MASK_INF** — natively supported on Vulkan (unlike Metal, where it falls
   back to CPU). Pre-transformer attention runs fully on GPU.

So the Qwen3-TTS decoder should run on the Vulkan backend with NO equivalent
patches. The Metal patches in our ggml fork are Metal-only and harmless to Vulkan.

### 8.2 Concrete steps
1. Get a Linux box with Vulkan (CI uses `ubuntu-24.04` + `libvulkan-dev glslc
   libshaderc-dev spirv-tools mesa-vulkan-drivers vulkan-tools`). Windows uses the
   Vulkan SDK (see the parakeet workflow for the exact setup).
2. Build ggml with Vulkan:
   ```bash
   cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
   cmake --build ggml/build -j
   ```
3. **Teach the qwen CMake about Vulkan.** Currently `CMakeLists.txt` only detects
   Metal:
   ```cmake
   if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
       set(GGML_HAS_METAL ON) ...
   ```
   Add a parallel `GGML_HAS_VULKAN` detection (look for
   `${GGML_BUILD_DIR}/src/ggml-vulkan/libggml-vulkan.{so,dll}`) and, where it links
   `ggml-metal` + Metal frameworks, link `ggml-vulkan` instead. The link blocks are
   repeated for each lib target (text_tokenizer, tts_transformer,
   audio_tokenizer_encoder, audio_tokenizer_decoder) plus the `qwen3_tts` lib.
   Grep for `GGML_HAS_METAL` and `ggml-metal` to find all spots.
4. Build the worker: `cmake -S . -B build -DQWEN3_TTS_COREML=OFF` (CoreML is
   macOS-only anyway), `cmake --build build --target qwen3-tts-worker qwen3-tts-cli`.
5. **Verify correctness first** with the CLI one-shot:
   ```bash
   ./build/qwen3-tts-cli -m models -t "Hallo Welt." \
     -r ../../data/voices/elevenlabs-pibot-reference-de.wav -l de -o /tmp/vk.wav
   ```
   Listen. The main risk is the **talker matmul** on Vulkan (the analog of the
   Metal Tensor-API correctness bug). Vulkan has its own matmul (coopmat / scalar
   fallback). If the talker output is wrong/truncated, try disabling coopmat
   (`GGML_VK_DISABLE_COOPMAT=1` / `GGML_VK_DISABLE_COOPMAT2=1`) and compare. This is
   the one unknown.
6. Profile/benchmark decode. The Vulkan conv_transpose is parallel but uses per-K
   barriers + shared-mem scatter — decent, not necessarily optimal. Only optimize
   if decode is a bottleneck (it likely won't be the conv_transpose disaster).
7. There is no CoreML / no Tensor-disable concern on Vulkan.

### 8.3 Testing the worker protocol
Use a small harness (Python): spawn `qwen3-tts-worker --serve -m models
--ref-audio <wav> --language de`, wait for READY (frame type 1), send SPEAK
(type 1, request_id, utf8 text), collect AUDIO_CHUNK (type 3) int16 payloads until
AUDIO_DONE (type 4), write a WAV. (A copy of such a harness was used during Metal
dev; re-create from the protocol in §6.)

---

## 9. Open items / TODO

- [ ] **Vulkan port** (§8) — the main next task.
- [ ] **Static binary** — ggml is currently built with `BUILD_SHARED_LIBS=ON`, so
  the worker links `@rpath` dylibs and is not self-contained/downloadable. For
  distributable binaries: build ggml static (`-DBUILD_SHARED_LIBS=OFF`) and link
  statically, then verify with `otool -L` / `ldd` (only system libs + Metal/Vulkan).
- [ ] **CI release workflow** — mirror `.github/workflows/parakeet-cpp-stt-release.yml`
  for qwen3-tts: matrix per platform (macOS arm64 Metal now; Linux/Windows Vulkan
  later), build `qwen3-tts-worker` + `qwen3-tts-cli`, tar/zip + sha256, GH release.
  The model GGUFs are downloaded separately (HF), not shipped in the archive.
- [ ] **Make cpp the default pibot TTS** — `src/server/config.ts` still defaults to
  `rust` on macOS; flip to `cpp` after an A/B listen test (x-vector vs MLX ICL).
  Also add `build:tts-cpp` to `scripts/build-native.mjs`.
- [ ] **Phase 3: ICL conditioning** — the worker is x-vector only. MLX clones via a
  dual-stream ICL prefill (ref transcript tokens + ref audio codec tokens). The
  building blocks exist (audio encoder, codec embeddings) but the dual-stream
  prefill must be added to the talker. See the MLX reference:
  `native/qwen3_tts_rs/src/inference.rs::build_input_embeddings_with_icl`.

---

## 10. Key files

```
src/worker_main.cpp                       worker protocol + streaming (phases 1&2)
src/qwen3_tts.{h,cpp}                      orchestration + synthesize_streaming_*
src/tts_transformer.{h,cpp}               talker; generate() has per-frame callback
src/audio_tokenizer_decoder.{h,cpp}       vocoder (decode(codes,n_frames,samples))
src/gguf_loader.cpp                        GGML_METAL_TENSOR_DISABLE default
CMakeLists.txt                            targets; Metal-only ggml detection (§8.3)
ggml/src/ggml-metal/ggml-metal.metal      kernel_conv_transpose_1d, kernel_pad
ggml/src/ggml-vulkan/vulkan-shaders/conv_transpose_1d.comp   (already parallel)
ggml/src/ggml-vulkan/vulkan-shaders/pad.comp                 (already has lp/rp)
```

pibot side:
```
src/server/tts.ts        worker kinds incl. "cpp"; frame protocol client
src/server/config.ts     qwen3TtsCppWorkerPath / qwen3TtsCppModelPath, default kind
src/server/index.ts      passes cppWorkerPath/cppModelPath
package.json             build:tts-cpp
```
