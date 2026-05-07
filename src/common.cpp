#include "common.h"
#include <cstdio>
#include <cstring>

void app_print_usage(int /*argc*/, char ** argv, const app_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

bool app_params_parse(int argc, char ** argv, app_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "error: argument %s requires a value\n", arg.c_str());
                return false;
            }
            params.model = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            app_print_usage(argc, argv, params);
            return false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            app_print_usage(argc, argv, params);
            return false;
        }
    }

    return true;
}
