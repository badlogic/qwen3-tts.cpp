// Persistent binary-framed Qwen3-TTS worker (C++/GGML/Metal backend).
//
// Mirrors the pibot worker protocol used by src/server/tts.ts and the Rust/MLX
// worker: frame header is uint8 type, uint32 request_id, uint32 payload_len
// (little-endian), followed by the payload.
//
// Conditioning is x-vector (speaker embedding) only; ICL is intentionally out of
// scope for now. Generation is streamed: the talker produces codec frames
// autoregressively and the (causal) vocoder decodes the growing prefix every
// few frames, emitting only the newly produced PCM.

#include "qwen3_tts.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace {

constexpr uint8_t WORKER_INPUT_SPEAK = 1;
constexpr uint8_t WORKER_INPUT_CANCEL = 2;
constexpr uint8_t WORKER_INPUT_SHUTDOWN = 3;
constexpr uint8_t WORKER_OUTPUT_READY = 1;
constexpr uint8_t WORKER_OUTPUT_AUDIO_START = 2;
constexpr uint8_t WORKER_OUTPUT_AUDIO_CHUNK = 3;
constexpr uint8_t WORKER_OUTPUT_AUDIO_DONE = 4;
constexpr uint8_t WORKER_OUTPUT_ERROR = 5;
constexpr size_t FRAME_HEADER_BYTES = 9;

struct Args {
    bool serve = false;
    std::string model_dir;
    std::string ref_audio;
    std::string language = "en";
    std::string text;
    std::string text_file;
    std::string output = "output.wav";
    int output_sample_rate = 24000;
    float temperature = 0.9f;
    int top_k = 50;
    int max_tokens = 2048;
    int blocksize = 512;
    int streaming_chunk_frames = 16;
};

int32_t language_to_id(const std::string & lang) {
    if (lang == "en" || lang == "english") return 2050;
    if (lang == "ru" || lang == "russian") return 2069;
    if (lang == "zh" || lang == "chinese") return 2055;
    if (lang == "ja" || lang == "japanese") return 2058;
    if (lang == "ko" || lang == "korean") return 2064;
    if (lang == "de" || lang == "german") return 2053;
    if (lang == "fr" || lang == "french") return 2061;
    if (lang == "es" || lang == "spanish") return 2054;
    if (lang == "it" || lang == "italian") return 2070;
    if (lang == "pt" || lang == "portuguese") return 2071;
    return 2050;
}

bool read_text_file(const std::string & path, std::string & out) {
    std::ifstream file(path);
    if (!file) return false;
    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
    return true;
}

// Linear resampler (speech quality is fine; matches the Rust worker's approach).
std::vector<float> resample_linear(const std::vector<float> & in, int from_rate, int to_rate) {
    if (from_rate == to_rate || in.empty()) return in;
    const double ratio = (double)to_rate / (double)from_rate;
    const size_t n_out = (size_t)((double)in.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src = (double)i / ratio;
        const size_t i0 = (size_t)src;
        const double frac = src - (double)i0;
        const float a = in[i0];
        const float b = (i0 + 1 < in.size()) ? in[i0 + 1] : a;
        out[i] = (float)(a + (b - a) * frac);
    }
    return out;
}

// Writes binary protocol frames to a raw fd (the original stdout, preserved
// before stdout is redirected to stderr).
class BinaryWriter {
public:
    explicit BinaryWriter(int fd) : fd_(fd) {}

    bool write_frame(uint8_t type, uint32_t request_id, const uint8_t * payload, size_t payload_len) {
        uint8_t header[FRAME_HEADER_BYTES];
        header[0] = type;
        std::memcpy(header + 1, &request_id, 4);
        uint32_t len = (uint32_t)payload_len;
        std::memcpy(header + 5, &len, 4);
        if (!write_all(header, FRAME_HEADER_BYTES)) return false;
        if (payload_len > 0 && !write_all(payload, payload_len)) return false;
        return true;
    }

private:
    bool write_all(const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *)data;
        size_t off = 0;
        std::lock_guard<std::mutex> lock(mu_);
        while (off < len) {
            ssize_t n = ::write(fd_, p + off, len - off);
            if (n <= 0) return false;
            off += (size_t)n;
        }
        return true;
    }

    int fd_;
    std::mutex mu_;
};

struct Frame {
    uint8_t type = 0;
    uint32_t request_id = 0;
    std::vector<uint8_t> payload;
};

bool read_exact(int fd, uint8_t * buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::read(fd, buf + off, len - off);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

bool read_frame(int fd, Frame & frame) {
    uint8_t header[FRAME_HEADER_BYTES];
    if (!read_exact(fd, header, FRAME_HEADER_BYTES)) return false;
    frame.type = header[0];
    std::memcpy(&frame.request_id, header + 1, 4);
    uint32_t payload_len = 0;
    std::memcpy(&payload_len, header + 5, 4);
    frame.payload.resize(payload_len);
    if (payload_len > 0 && !read_exact(fd, frame.payload.data(), payload_len)) return false;
    return true;
}

// Converts streamed float chunks to blocksize-quantized i16 PCM frames, trimming
// the leading silence (with a short preroll) like the Rust worker.
class PcmStreamer {
public:
    PcmStreamer(int model_rate, int output_rate, int blocksize, uint32_t request_id, BinaryWriter & writer)
        : model_rate_(model_rate), output_rate_(output_rate),
          blocksize_(blocksize > 0 ? blocksize : 512), request_id_(request_id), writer_(writer) {}

    bool push(const float * samples, int n_samples) {
        std::vector<float> buf(samples, samples + n_samples);
        if (model_rate_ != output_rate_) buf = resample_linear(buf, model_rate_, output_rate_);

        std::vector<int16_t> pcm = to_i16(buf);
        if (!found_speech_) {
            const int16_t threshold = (int16_t)(32768.0 * 0.01);
            size_t first = pcm.size();
            for (size_t i = 0; i < pcm.size(); ++i) {
                if (std::abs((int)pcm[i]) > threshold) { first = i; break; }
            }
            if (first == pcm.size()) return true; // still silent
            const size_t preroll = (size_t)((double)output_rate_ * 0.040);
            const size_t start = first > preroll ? first - preroll : 0;
            pcm.erase(pcm.begin(), pcm.begin() + (long)start);
            found_speech_ = true;
        }

        if (!leftover_.empty()) {
            leftover_.insert(leftover_.end(), pcm.begin(), pcm.end());
            pcm.swap(leftover_);
            leftover_.clear();
        }

        const size_t complete = (pcm.size() / (size_t)blocksize_) * (size_t)blocksize_;
        for (size_t off = 0; off < complete; off += (size_t)blocksize_) {
            audio_samples_ += blocksize_;
            if (!emit(pcm.data() + off, blocksize_)) return false;
        }
        leftover_.assign(pcm.begin() + (long)complete, pcm.end());
        return true;
    }

    bool finish() {
        if (leftover_.empty()) return true;
        audio_samples_ += (int)leftover_.size();
        leftover_.resize((size_t)blocksize_, 0);
        return emit(leftover_.data(), blocksize_);
    }

    int audio_samples() const { return audio_samples_; }

private:
    std::vector<int16_t> to_i16(const std::vector<float> & in) {
        std::vector<int16_t> out(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            float v = in[i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            float s = v * 32768.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            out[i] = (int16_t)s;
        }
        return out;
    }

    bool emit(const int16_t * pcm, int n) {
        return writer_.write_frame(WORKER_OUTPUT_AUDIO_CHUNK, request_id_,
                                   (const uint8_t *)pcm, (size_t)n * sizeof(int16_t));
    }

    int model_rate_;
    int output_rate_;
    int blocksize_;
    uint32_t request_id_;
    BinaryWriter & writer_;
    bool found_speech_ = false;
    std::vector<int16_t> leftover_;
    int audio_samples_ = 0;
};

void log_json(const std::string & json) {
    fprintf(stderr, "%s\n", json.c_str());
}

Args parse_args(int argc, char ** argv) {
    Args args;
    auto next = [&](int & i) -> std::string {
        if (++i >= argc) { fprintf(stderr, "Error: missing value for %s\n", argv[i - 1]); exit(1); }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--serve") args.serve = true;
        else if (a == "-m" || a == "--model" || a == "--model-name" || a == "--model-path") args.model_dir = next(i);
        else if (a == "--ref-audio" || a == "-r" || a == "--reference") args.ref_audio = next(i);
        else if (a == "--language" || a == "-l") args.language = next(i);
        else if (a == "--output-sample-rate") args.output_sample_rate = std::stoi(next(i));
        else if (a == "--temperature") args.temperature = std::stof(next(i));
        else if (a == "--top-k") args.top_k = std::stoi(next(i));
        else if (a == "--max-tokens" || a == "--max-new-tokens") args.max_tokens = std::stoi(next(i));
        else if (a == "--blocksize") args.blocksize = std::stoi(next(i));
        else if (a == "--streaming-chunk-size") args.streaming_chunk_frames = std::stoi(next(i));
        else if (a == "-t" || a == "--text") args.text = next(i);
        else if (a == "--text-file") args.text_file = next(i);
        else if (a == "-o" || a == "--output") args.output = next(i);
        // Accepted-but-ignored Python/Rust worker compatibility flags.
        else if (a == "--ref-text" || a == "--ref-text-file" || a == "--seed" || a == "--top-p" ||
                 a == "--repetition-penalty" || a == "--speed" || a == "--device" || a == "--dtype" ||
                 a == "--speaker" || a == "--instruct" || a == "--mlx-quantization" ||
                 a == "--attn-implementation" || a == "--output-dir") {
            (void)next(i);
        } else if (a == "--xvec-only" || a == "--parity-mode" || a == "--non-streaming-mode") {
            // boolean compat flags, ignored
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", a.c_str());
            exit(1);
        }
    }
    return args;
}

class Worker {
public:
    bool load(const Args & args) {
        args_ = args;
        params_.temperature = args.temperature;
        params_.top_k = args.top_k;
        params_.max_audio_tokens = args.max_tokens;
        params_.language_id = language_to_id(args.language);
        params_.print_progress = false;
        params_.print_timing = false;

        if (args.model_dir.empty()) { error_ = "model directory is required (-m/--model-name)"; return false; }
        if (args.ref_audio.empty()) { error_ = "reference audio is required (--ref-audio)"; return false; }

        if (!tts_.load_models(args.model_dir)) { error_ = tts_.get_error(); return false; }

        std::vector<float> ref_samples;
        int ref_rate = 0;
        if (!qwen3_tts::load_audio_file(args.ref_audio, ref_samples, ref_rate)) {
            error_ = "failed to load reference audio: " + args.ref_audio;
            return false;
        }
        if (ref_rate != 24000) ref_samples = resample_linear(ref_samples, ref_rate, 24000);

        if (!tts_.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), embedding_, params_)) {
            error_ = "failed to extract speaker embedding: " + tts_.get_error();
            return false;
        }
        model_sample_rate_ = 24000;
        return true;
    }

    const std::string & error() const { return error_; }

    // Generate once to a WAV file (non-serve mode, for testing).
    bool generate_once() {
        std::string text = args_.text;
        if (text.empty() && !args_.text_file.empty()) {
            if (!read_text_file(args_.text_file, text)) { error_ = "failed to read text file"; return false; }
        }
        if (text.empty()) { error_ = "provide --text or --text-file"; return false; }
        qwen3_tts::tts_result result = tts_.synthesize_with_embedding(
            text, embedding_.data(), (int32_t)embedding_.size(), params_);
        if (!result.success) { error_ = result.error_msg; return false; }
        std::vector<float> out = result.audio;
        if (result.sample_rate != args_.output_sample_rate) {
            out = resample_linear(out, result.sample_rate, args_.output_sample_rate);
        }
        if (!qwen3_tts::save_audio_file(args_.output, out, args_.output_sample_rate)) {
            error_ = "failed to save output: " + args_.output;
            return false;
        }
        fprintf(stderr, "Output saved to: %s\n", args_.output.c_str());
        return true;
    }

    void handle_speak(uint32_t request_id, const std::vector<uint8_t> & payload, BinaryWriter & writer,
                      std::mutex & cancel_mu, std::unordered_set<uint32_t> & cancelled) {
        auto is_cancelled = [&]() {
            std::lock_guard<std::mutex> lock(cancel_mu);
            return cancelled.count(request_id) > 0;
        };
        if (is_cancelled()) {
            writer.write_frame(WORKER_OUTPUT_AUDIO_DONE, request_id, nullptr, 0);
            return;
        }

        std::string text((const char *)payload.data(), payload.size());
        // trim
        size_t b = text.find_first_not_of(" \t\r\n");
        size_t e = text.find_last_not_of(" \t\r\n");
        text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
        if (text.empty()) {
            writer.write_frame(WORKER_OUTPUT_ERROR, request_id, (const uint8_t *)"empty text", 10);
            return;
        }

        uint32_t sr = (uint32_t)args_.output_sample_rate;
        writer.write_frame(WORKER_OUTPUT_AUDIO_START, request_id, (const uint8_t *)&sr, sizeof(sr));

        PcmStreamer streamer(model_sample_rate_, args_.output_sample_rate, args_.blocksize, request_id, writer);
        bool write_failed = false;

        qwen3_tts::tts_result result = tts_.synthesize_streaming_with_embedding(
            text, embedding_.data(), (int32_t)embedding_.size(),
            args_.streaming_chunk_frames,
            [&](const float * samples, int32_t n, int32_t /*rate*/) -> bool {
                if (is_cancelled()) return false;
                if (!streamer.push(samples, n)) { write_failed = true; return false; }
                return true;
            },
            params_);

        if (write_failed) {
            return; // stdout broken; nothing more we can do
        }
        if (is_cancelled()) {
            writer.write_frame(WORKER_OUTPUT_AUDIO_DONE, request_id, nullptr, 0);
            return;
        }
        if (!result.success) {
            writer.write_frame(WORKER_OUTPUT_ERROR, request_id,
                               (const uint8_t *)result.error_msg.data(), result.error_msg.size());
            return;
        }
        streamer.finish();
        log_json("{\"type\":\"generated\",\"backend\":\"cpp_ggml\",\"audioSamples\":" +
                 std::to_string(streamer.audio_samples()) + "}");
        writer.write_frame(WORKER_OUTPUT_AUDIO_DONE, request_id, nullptr, 0);
    }

    const Args & args() const { return args_; }

private:
    Args args_;
    qwen3_tts::Qwen3TTS tts_;
    qwen3_tts::tts_params params_;
    std::vector<float> embedding_;
    int model_sample_rate_ = 24000;
    std::string error_;
};

int serve(Worker & worker, BinaryWriter & writer) {
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::queue<Frame> queue;
    std::mutex cancel_mu;
    std::unordered_set<uint32_t> cancelled;
    bool reader_done = false;

    std::thread reader([&]() {
        for (;;) {
            Frame frame;
            if (!read_frame(STDIN_FILENO, frame)) {
                frame.type = WORKER_INPUT_SHUTDOWN;
                frame.request_id = 0;
            }
            if (frame.type == WORKER_INPUT_CANCEL) {
                std::lock_guard<std::mutex> lock(cancel_mu);
                cancelled.insert(frame.request_id);
                continue;
            }
            bool shutdown = frame.type == WORKER_INPUT_SHUTDOWN;
            {
                std::lock_guard<std::mutex> lock(queue_mu);
                queue.push(std::move(frame));
            }
            queue_cv.notify_one();
            if (shutdown) break;
        }
    });

    writer.write_frame(WORKER_OUTPUT_READY, 0, nullptr, 0);
    log_json("{\"type\":\"server_ready\",\"backend\":\"cpp_ggml\"}");

    for (;;) {
        Frame frame;
        {
            std::unique_lock<std::mutex> lock(queue_mu);
            queue_cv.wait(lock, [&]() { return !queue.empty(); });
            frame = std::move(queue.front());
            queue.pop();
        }
        if (frame.type == WORKER_INPUT_SHUTDOWN) break;
        if (frame.type == WORKER_INPUT_SPEAK) {
            worker.handle_speak(frame.request_id, frame.payload, writer, cancel_mu, cancelled);
            std::lock_guard<std::mutex> lock(cancel_mu);
            cancelled.erase(frame.request_id);
        } else {
            std::string msg = "unknown frame type " + std::to_string((int)frame.type);
            writer.write_frame(WORKER_OUTPUT_ERROR, frame.request_id, (const uint8_t *)msg.data(), msg.size());
        }
    }

    reader_done = true;
    (void)reader_done;
    reader.detach();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    // Preserve the real stdout for the binary protocol, then redirect stdout to
    // stderr so any library chatter (ggml/metal logs, printf) cannot corrupt it.
    int saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0) { fprintf(stderr, "failed to dup stdout\n"); return 1; }
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) { fprintf(stderr, "failed to redirect stdout\n"); return 1; }

    BinaryWriter writer(saved_stdout);
    Args args = parse_args(argc, argv);

    Worker worker;
    if (!worker.load(args)) {
        const std::string & msg = worker.error();
        writer.write_frame(WORKER_OUTPUT_ERROR, 0, (const uint8_t *)msg.data(), msg.size());
        fprintf(stderr, "worker load failed: %s\n", msg.c_str());
        return 1;
    }

    if (args.serve) {
        return serve(worker, writer);
    }
    if (!worker.generate_once()) {
        fprintf(stderr, "generate failed: %s\n", worker.error().c_str());
        return 1;
    }
    return 0;
}
