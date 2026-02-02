def cli():
    # Local imports keep CLI startup lightweight and avoid import-time side effects.
    from .utils import onnx_to_mxa
    from .logger import logs

    import argparse
    import traceback
    from os import makedirs
    from os.path import join

    parser = argparse.ArgumentParser(description="Compile model archive (zip) to MemryX artifacts")
    parser.add_argument("model_zip", help="Path to .zip containing model.onnx and ncconfig.json")
    parser.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()
    model_zip = args.model_zip
    output_dir = args.output_dir

    # Ensure output directory exists before writing artifacts (logs.json, zip, etc.).
    makedirs(output_dir, exist_ok=True)

    # Logs file lives alongside other artifacts in the output directory.
    logs_path = join(output_dir, "logs.json")

    logs.add_message(
        "Converting model archive to DFP",
        {
            "Model Archive": model_zip,
            "Output Directory": output_dir,
        },
    )

    try:
        onnx_to_mxa(model_zip, output_dir)
        logs.add_message("Conversion complete", {"Output Directory": output_dir})
        print("Conversion complete. Logs saved as JSON file.")

    except Exception as e:
        logs.add_message(
            "Conversion failed",
            {
                "Error": str(e),
                "Traceback": traceback.format_exc(),
            },
        )
        raise

    finally:
        logs.save_as_json(path=logs_path)
        print(logs)
        print("Exiting...")
