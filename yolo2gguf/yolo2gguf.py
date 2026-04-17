import argparse
import os
import torch
import numpy as np
import gguf

def main():
    parser = argparse.ArgumentParser(description="YOLO to GGUF converter")
    parser.add_argument("--input", required=True, help="Path to input yoloXXX.pt file")
    parser.add_argument("--output", required=True, help="Path to output yoloXXX .gguf file")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
