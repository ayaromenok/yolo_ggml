#ifndef TGA_H
#define TGA_H

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <iostream>

#pragma pack(push, 1)
struct TGAHeader {
    uint8_t  id_length;
    uint8_t  color_map_type;
    uint8_t  image_type;
    uint16_t color_map_first_index;
    uint16_t color_map_length;
    uint8_t  color_map_entry_size;
    uint16_t x_origin;
    uint16_t y_origin;
    uint16_t width;
    uint16_t height;
    uint8_t  pixel_depth;
    uint8_t  image_descriptor;
};
#pragma pack(pop)

struct TGAImage {
    int width = 0;
    int height = 0;
    int channels = 0; // 3 for RGB, 4 for RGBA, 1 for Grayscale
    std::vector<uint8_t> data;

    bool is_valid() const { return !data.empty(); }
};

/**
 * @brief Loads a TGA image from a file.
 * Supports uncompressed and RLE compressed 24/32-bit RGB(A) and 8-bit Grayscale.
 */
inline TGAImage tga_load(const std::string& filename) {
    TGAImage img;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "TGA: Failed to open file " << filename << std::endl;
        return img;
    }

    TGAHeader header;
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        std::cerr << "TGA: Failed to read header" << std::endl;
        return img;
    }

    // Skip ID field if present
    if (header.id_length > 0) {
        file.seekg(header.id_length, std::ios::cur);
    }

    // Check image type
    // 2: Uncompressed True-color
    // 3: Uncompressed Black-and-white
    // 10: RLE True-color
    // 11: RLE Black-and-white
    bool is_rle = (header.image_type == 10 || header.image_type == 11);
    bool is_gray = (header.image_type == 3 || header.image_type == 11);

    if (header.image_type != 2 && header.image_type != 3 && header.image_type != 10 && header.image_type != 11) {
        std::cerr << "TGA: Unsupported image type " << (int)header.image_type << std::endl;
        return img;
    }

    img.width = header.width;
    img.height = header.height;
    img.channels = header.pixel_depth / 8;

    if (img.channels != 1 && img.channels != 3 && img.channels != 4) {
        std::cerr << "TGA: Unsupported pixel depth " << (int)header.pixel_depth << std::endl;
        return img;
    }

    size_t total_pixels = static_cast<size_t>(img.width) * img.height;
    size_t total_bytes = total_pixels * img.channels;
    img.data.resize(total_bytes);

    if (!is_rle) {
        if (!file.read(reinterpret_cast<char*>(img.data.data()), total_bytes)) {
            std::cerr << "TGA: Failed to read pixel data" << std::endl;
            img.data.clear();
            return img;
        }
    } else {
        // RLE Decoding
        size_t bytes_read = 0;
        while (bytes_read < total_bytes) {
            uint8_t chunk_header;
            if (!file.read(reinterpret_cast<char*>(&chunk_header), 1)) break;

            int count = (chunk_header & 0x7F) + 1;
            if (chunk_header & 0x80) {
                // RLE packet
                uint8_t pixel[4];
                file.read(reinterpret_cast<char*>(pixel), img.channels);
                for (int i = 0; i < count; ++i) {
                    if (bytes_read + img.channels <= total_bytes) {
                        std::memcpy(&img.data[bytes_read], pixel, img.channels);
                        bytes_read += img.channels;
                    }
                }
            } else {
                // Raw packet
                size_t raw_bytes = count * img.channels;
                if (bytes_read + raw_bytes <= total_bytes) {
                    file.read(reinterpret_cast<char*>(&img.data[bytes_read]), raw_bytes);
                    bytes_read += raw_bytes;
                } else {
                    break;
                }
            }
        }
    }

    // TGA stores pixels in BGR(A) order, convert to RGB(A)
    if (img.channels >= 3) {
        for (size_t i = 0; i < total_pixels; ++i) {
            uint8_t* p = &img.data[i * img.channels];
            std::swap(p[0], p[2]);
        }
    }

    // Handle orientation: default is bottom-up (descriptor bit 5 = 0)
    // If bit 5 is 0, we flip to top-down for common usage
    bool top_down = (header.image_descriptor & 0x20) != 0;
    if (!top_down) {
        std::vector<uint8_t> flipped(total_bytes);
        size_t row_bytes = img.width * img.channels;
        for (int y = 0; y < img.height; ++y) {
            std::memcpy(&flipped[y * row_bytes], &img.data[(img.height - 1 - y) * row_bytes], row_bytes);
        }
        img.data = std::move(flipped);
    }

    return img;
}

/**
 * @brief Saves a TGA image to a file as uncompressed.
 */
inline bool tga_save(const std::string& filename, const TGAImage& img) {
    if (!img.is_valid()) return false;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "TGA: Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    TGAHeader header;
    std::memset(&header, 0, sizeof(header));
    header.image_type = (img.channels == 1) ? 3 : 2;
    header.width = static_cast<uint16_t>(img.width);
    header.height = static_cast<uint16_t>(img.height);
    header.pixel_depth = static_cast<uint8_t>(img.channels * 8);
    header.image_descriptor = 0x20; // Top-down

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Convert RGB(A) back to BGR(A) for saving
    std::vector<uint8_t> out_data = img.data;
    if (img.channels >= 3) {
        size_t total_pixels = static_cast<size_t>(img.width) * img.height;
        for (size_t i = 0; i < total_pixels; ++i) {
            uint8_t* p = &out_data[i * img.channels];
            std::swap(p[0], p[2]);
        }
    }

    file.write(reinterpret_cast<const char*>(out_data.data()), out_data.size());
    return true;
}

#endif // TGA_H
