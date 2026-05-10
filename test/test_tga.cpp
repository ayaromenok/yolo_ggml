#include "../src/tga.h"
#include <iostream>
#include <cassert>

int main() {
    std::string input_filename = "data/img/ancelotti_zidane_2014.tga";
    std::string output_filename = "test_out.tga";

    std::cout << "Loading " << input_filename << "..." << std::endl;
    TGAImage img = tga_load(input_filename);
    if (!img.is_valid()) {
        std::cerr << "Failed to load original TGA: " << input_filename << std::endl;
        return 1;
    }

    std::cout << "Original info: " << img.width << "x" << img.height << " (" << img.channels << " channels)" << std::endl;

    std::cout << "Saving as " << output_filename << "..." << std::endl;
    if (!tga_save(output_filename, img)) {
        std::cerr << "Failed to save TGA to " << output_filename << std::endl;
        return 1;
    }

    std::cout << "Reloading " << output_filename << "..." << std::endl;
    TGAImage reloaded = tga_load(output_filename);
    if (!reloaded.is_valid()) {
        std::cerr << "Failed to reload saved TGA" << std::endl;
        return 1;
    }

    std::cout << "Verifying..." << std::endl;
    assert(reloaded.width == img.width);
    assert(reloaded.height == img.height);
    assert(reloaded.channels == img.channels);
    assert(reloaded.data.size() == img.data.size());

    for (size_t i = 0; i < img.data.size(); ++i) {
        if (reloaded.data[i] != img.data[i]) {
            std::cerr << "Mismatch at byte " << i << ": expected " << (int)img.data[i] << ", got " << (int)reloaded.data[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Success! TGA loader/saver verified with " << input_filename << std::endl;
    return 0;
}
