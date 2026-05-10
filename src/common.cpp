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
    fprintf(stderr, "  -i FNAME, --image FNAME\n");
    fprintf(stderr, "                        input image path (default: %s)\n", params.image.c_str());
    fprintf(stderr, "  -o FNAME, --output FNAME\n");
    fprintf(stderr, "                        output image path (default: %s)\n", params.output.c_str());
    fprintf(stderr, "  -l FNAME, --labels FNAME\n");
    printf("  -l, --labels FNAME    COCO labels file (default: %s)\n", params.labels.c_str());
    printf("  -s, --stats FNAME     statistics output file (default: %s)\n", params.stats.c_str());
    printf("\n");
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
        } else if (arg == "-i" || arg == "--image") {
            if (++i >= argc) {
                fprintf(stderr, "error: argument %s requires a value\n", arg.c_str());
                return false;
            }
            params.image = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "error: argument %s requires a value\n", arg.c_str());
                return false;
            }
            params.output = argv[i];
        } else if (arg == "-l" || arg == "--labels") {
            if (++i >= argc) { app_print_usage(argc, argv, params); return false; }
            params.labels = argv[i];
        } else if (arg == "-s" || arg == "--stats") {
            if (++i >= argc) { app_print_usage(argc, argv, params); return false; }
            params.stats = argv[i];
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
