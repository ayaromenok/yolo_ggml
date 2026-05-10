#pragma once

#include <string>
#include <vector>

struct app_params {
    std::string model  = "../data/gguf/yolo26n_F16.gguf";
    std::string image  = "../data/img/ancelotti_zidane_2014.tga";
    std::string output = "../data/out/output.tga";
    std::string labels = "../data/coco/ms_coco_classnames.txt";
};

bool app_params_parse(int argc, char ** argv, app_params & params);
void app_print_usage(int argc, char ** argv, const app_params & params);
