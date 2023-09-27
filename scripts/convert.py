import argparse
import os
import pathlib

from onnx2tf import convert


def main():
    parser = argparse.ArgumentParser(description="ONNX to TensorFlow model converter")
    parser.add_argument("-i", "--input-onnx-file", type=str, required=True, help="Path to the ONNX file")
    parser.add_argument("-o", "--output-directory", type=str, default="", help="Path to save the modified model")

    args = parser.parse_args()
    
    input_onnx_file = args.input_onnx_file
    
    if not args.output_directory:
        directory_path, file_name = os.path.split(input_onnx_file)
        base_name = os.path.basename(file_name)
        extension = os.path.splitext(base_name)[1]
        output_directory = os.path.join(directory_path, base_name[:-len(extension)])
    else:
        output_directory = args.output_directory

    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Input ONNX file: {input_onnx_file}")
    print(f"Output directory: {output_directory}")
    
    convert(
        input_onnx_file_path=input_onnx_file,
        output_folder_path=output_directory,
        output_signaturedefs=True,
    )
    print(f"Finished, converted model path: {output_directory}")


if __name__ == "__main__":
    main()
