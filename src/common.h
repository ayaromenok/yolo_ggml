#pragma once

#include <string>
#include <vector>

struct app_params {
    std::string model = "../data/gguf/dummy_yolo.gguf";
};

bool app_params_parse(int argc, char ** argv, app_params & params);
void app_print_usage(int argc, char ** argv, const app_params & params);
