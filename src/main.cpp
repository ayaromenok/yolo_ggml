#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <vector>
#include <string>

int main(int argc, char ** argv) {
    const char * fname = "../data/gguf/dummy_yolo.gguf";
    if (argc > 1) {
        fname = argv[1];
    }

    printf("Loading model from: %s\n", fname);

    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf) {
        fprintf(stderr, "Failed to load GGUF file: %s\n", fname);
        return 1;
    }

    printf("GGUF version: %u\n", gguf_get_version(ctx_gguf));
    printf("Number of tensors: %ld\n", (long)gguf_get_n_tensors(ctx_gguf));
    printf("Number of KV pairs: %ld\n", (long)gguf_get_n_kv(ctx_gguf));

    printf("\nKey-Value Pairs:\n");
    for (int i = 0; i < gguf_get_n_kv(ctx_gguf); ++i) {
        const char * key = gguf_get_key(ctx_gguf, i);
        printf("  %d: %s\n", i, key);
    }

    printf("\nTensors:\n");
    for (int i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            printf("  %d: %s (NOT FOUND in context)\n", i, name);
            continue;
        }
        printf("  %d: %-20s [%5ld, %5ld, %5ld, %5ld] type=%s\n", 
            i, name, (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
            ggml_type_name(t->type));
    }

    printf("\nModel loaded and verified successfully.\n");

    gguf_free(ctx_gguf);
    ggml_free(ctx_data);

    return 0;
}
