def cli():
    # Local imports keep CLI startup lightweight and avoid import-time side effects.
    from .utils import onnx_to_mxa
    from .logger import logs

    import argparse
    import traceback
    from os import makedirs
    from os.path import join

    # Argument parsing for the console entrypoint.
    parser = argparse.ArgumentParser(description='Compile ONNX to DFP')
    parser.add_argument('--onnx-path', required=True, help='Path to the ONNX file')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    # Optional args: default num_chips to 4, and parse as int.
    parser.add_argument('--num_chips', default=4, type=int,
                        help='Number of MX3 chips to compile for')

    # Optional string; None when not provided.
    parser.add_argument('--extensions',
                        action='append',
                        help='MemryX neural compiler extension (repeatable; can also be comma-separated or JSON list)')

    args = parser.parse_args()

    onnx_path = args.onnx_path
    output_dir = args.output_dir
    num_chips = args.num_chips
    extensions = args.extensions

    # Ensure output directory exists before writing artifacts (logs.json, zip, etc.).
    makedirs(output_dir, exist_ok=True)

    # Logs file lives alongside other artifacts in the output directory.
    logs_path = join(output_dir, 'logs.json')

    # Record the start of conversion with key parameters.
    logs.add_message(
        'Converting ONNX to DFP',
        {
            'ONNX Path': onnx_path,
            'Output Directory': output_dir,
            'Number of Chips': num_chips,
            'Extensions Provided': bool(extensions),
        }
    )

    try:
        # Perform the conversion; exceptions should propagate to keep a non-zero exit code.
        onnx_to_mxa(onnx_path, output_dir, num_chips, extensions)

        logs.add_message('Conversion complete', {'Output Directory': output_dir})
        print('Conversion complete. Logs saved as JSON file.')

    except Exception as e:
        # Capture exception details in logs for debugging.
        logs.add_message(
            'Conversion failed',
            {
                'Error': str(e),
                'Traceback': traceback.format_exc(),
            }
        )
        # Re-raise so callers/scripts see failure via exit code.
        raise

    finally:
        # Always attempt to persist logs, even on failure.
        logs.save_as_json(path=logs_path)
        print(logs)
        print('Exiting...')
