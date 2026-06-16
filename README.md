# yolo_ggml
command line app for main development of GGML/YOLO app

# Hardware

## TinkerBoard

With latest Armbian/trixier it's no Vulkan/OpenCL acceleration, so CPU only and 32bit

 | model | size, MB | perf, ms |
 |-------|----------|----------|
 | 26n   |  4.99    |  240     |
 | 26s   | 19.21    |  491     |
 | 26m   | 41.96    | 1228     |

# YOLO GGML Android
YGA - is native android application to work with YOLO models with a help GGML library. Use GPU acceleration via Vulkan backend for now
