#include "stats.h"
#include <cstdio>

void yolo_stats::print(FILE* f) const {
    fprintf(f, "--- Statistics ---\n");
    fprintf(f, "GGUF Model Name:     %s\n", model_name.c_str());
    fprintf(f, "GGUF Model Size:     %.2f MB\n", model_size / (1024.0 * 1024.0));
    fprintf(f, "GGML Allocated Mem:  %.2f MB (of %.2f MB buf)\n", mem_used / (1024.0 * 1024.0), mem_total / (1024.0 * 1024.0));
    fprintf(f, "Inference Time:      %.2f ms (simplified graph)\n", inference_time_ms);
    fprintf(f, "------------------\n");
}

void yolo_stats::save_to_file(const std::string& filename) const {
    if (filename.empty()) return;
    FILE* f = fopen(filename.c_str(), "w");
    if (f) {
        print(f);
        fclose(f);
        printf("Statistics saved to %s\n", filename.c_str());
    } else {
        fprintf(stderr, "Failed to open stats file for writing: %s\n", filename.c_str());
    }
}

yolo_stats collect_stats(const std::string& model_path, struct ggml_context* ctx, double duration_ms, size_t buf_size) {
    yolo_stats s;
    s.model_name = std::filesystem::path(model_path).filename().string();
    s.model_size = std::filesystem::file_size(model_path);
    s.mem_used = ggml_used_mem(ctx);
    s.mem_total = buf_size;
    s.inference_time_ms = duration_ms;
    return s;
}
