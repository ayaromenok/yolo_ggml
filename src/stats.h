#pragma once

#include <string>
#include <chrono>
#include <filesystem>
#include "ggml.h"

struct yolo_stats {
    std::string model_name;
    std::string backend_name;
    size_t model_size;
    size_t mem_used;
    size_t mem_total;
    size_t backend_mem_used;
    size_t backend_mem_total;
    double inference_time_ms;
    int iterations;

    void print(FILE* f) const;
    void save_to_file(const std::string& filename) const;
};

yolo_stats collect_stats(const std::string& model_path, const std::string& backend_name, 
                         struct ggml_context* ctx, double duration_ms, size_t buf_size, 
                         size_t backend_mem_used, size_t backend_mem_total, int iterations);
