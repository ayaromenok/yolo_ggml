#pragma once

#include <string>
#include <chrono>
#include <filesystem>
#include "ggml.h"

struct yolo_stats {
    std::string model_name;
    size_t model_size;
    size_t mem_used;
    size_t mem_total;
    double inference_time_ms;

    void print(FILE* f) const;
    void save_to_file(const std::string& filename) const;
};

yolo_stats collect_stats(const std::string& model_path, struct ggml_context* ctx, double duration_ms, size_t buf_size);
