allocate GGML based on model calculated size
```// Prepare GGML graph    
    //64 MB - OK for N-size
    //256 MB - OK for N too L-size
    static size_t buf_size = 384ULL * 1024 * 1024; ```
