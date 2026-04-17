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
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    print(f"Loading PyTorch model from {args.input}...")

    # Do no use CUDA/Vulkan/etc backends - models is quite small 
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)

    # Typically, Ultralytics models store the actual model or state_dict under 'model' key.

    state_dict = None
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
            elif isinstance(model, dict):
                state_dict = model
            else:
                state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        # If it's a raw model object
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            print("Error: Could not extract state_dict from the checkpoint.")
            return

    if not isinstance(state_dict, dict):
        print("Error: Extracted state_dict is not a dictionary.")
        return

    print(f"Extracted {len(state_dict)} tensors.")

    print(f"Initializing GGUF writer for architecture 'yolo'...")
    writer = gguf.GGUFWriter(args.output, "yolo")

    # Add some basic metadata
    writer.add_name("YOLO Model")
    writer.add_description("Converted from YOLO/pt to GGUF format")

    print("Writing tensors...")
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Convert PyTorch tensor to NumPy array, enforce fp32 or fp16 here. 
            data = tensor.detach().cpu().numpy()

            # gguf expects c-contiguous arrays
            if not data.flags.c_contiguous:
                data = np.ascontiguousarray(data)

            writer.add_tensor(name, data)
            print(f"  Added tensor: {name} | shape: {data.shape} | dtype: {data.dtype}")
        else:
            print(f"  Skipping {name} (not a torch.Tensor, type: {type(tensor)})")

    print(f"Saving GGUF file to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("Conversion successful!")

if __name__ == "__main__":
    main()
