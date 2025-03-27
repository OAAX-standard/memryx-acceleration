def cli():
    from .utils import onnx_to_mxa
    from .logger import logs

    from os.path import join
    from os import makedirs
    import argparse

    parser = argparse.ArgumentParser(description='Compile ONNX to DFP')
    parser.add_argument('--onnx-path', required=True, help='Path to the ONNX file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    onnx_path = args.onnx_path
    output_dir = args.output_dir
    logs_path = join(output_dir, 'logs.json')

    logs.add_message('Converting ONNX to DFP', {'ONNX Path': onnx_path, 'Output Directory': output_dir})
    makedirs(output_dir, exist_ok=True)

    onnx_to_mxa(onnx_path, output_dir)
    logs.add_message('Conversion complete', {'Output Directory': output_dir})
    logs.save_as_json(path=logs_path)
    print('Conversion complete. Logs saved as JSON file.')
    print(logs)
    print('Exiting...')
