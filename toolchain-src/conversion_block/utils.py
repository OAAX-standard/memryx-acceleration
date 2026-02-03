import json
import os
import re
import shutil
import zipfile
from glob import glob
from os.path import basename, dirname, join, split, splitext
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import onnx
from memryx import NeuralCompiler

from .logger import logs

def _extract_model_archive(model_zip_path: str, work_dir: str) -> None:
    """Extract the input model archive into the workspace directory.

    Args:
        model_zip_path: Path to the input .zip archive.
        work_dir: Directory where archive contents will be extracted.

    Raises:
        FileNotFoundError: If model_zip_path does not exist.
        zipfile.BadZipFile: If the archive is not a valid zip.
    """
    if not os.path.isfile(model_zip_path):
        raise FileNotFoundError(f"Model archive not found: {model_zip_path}")

    with zipfile.ZipFile(model_zip_path, "r") as zf:
        zf.extractall(work_dir)


def _load_ncconfig(work_dir: str) -> Dict[str, Any]:
    """Load ncconfig.json from the extracted workspace (flat config).

    Args:
        work_dir: Workspace directory where the archive was extracted.

    Returns:
        Flat JSON config as a dict.

    Raises:
        FileNotFoundError: If ncconfig.json is missing.
        ValueError: If ncconfig.json is not a JSON object.
        json.JSONDecodeError: If ncconfig.json is not valid JSON.
    """
    config_path = join(work_dir, "ncconfig.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing ncconfig.json in archive (expected at top level): {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("ncconfig.json must contain a JSON object (dictionary).")

    return cfg


def _resolve_models_from_config(cfg: Dict[str, Any], work_dir: str) -> List[str]:
    """Resolve config 'models' entries to extracted file paths in the workspace.

    Users are expected to include all referenced .onnx files in the archive and list
    them in cfg['models'] by filename (or relative path within the archive).

    Args:
        cfg: Flat config dict loaded from ncconfig.json.
        work_dir: Workspace directory where archive contents were extracted.

    Returns:
        List of absolute paths to ONNX model files in the workspace.

    Raises:
        KeyError: If 'models' is missing in cfg.
        ValueError: If 'models' is not a non-empty list.
        FileNotFoundError: If any referenced model file is missing in the workspace.
    """
    if "models" not in cfg:
        raise KeyError("ncconfig.json missing required key: 'models'")

    models = cfg["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("'models' must be a non-empty JSON list in ncconfig.json")

    resolved: List[str] = []
    for m in models:
        if not isinstance(m, str) or not m.strip():
            raise ValueError("Each entry in 'models' must be a non-empty string")
        path = join(work_dir, m)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model listed in ncconfig.json not found in archive: {m}")
        resolved.append(path)

    return resolved


def _timestamp_tag() -> str:
    """Return a filesystem-safe, human-readable UTC timestamp."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _forced_num_processes() -> int:
    """Force num_processes to (cpu_count - 2), with a floor of 1."""
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def _dfp_filename_for_models(models: List[str], timestamp: str) -> str:
    """Compute the DFP filename given resolved model paths.

    If a single model is compiled, use:
        <timestamp>_<modelstem>.dfp

    If multiple models are co-mapped, use:
        <timestamp>_models.dfp
    """
    if len(models) == 1:
        model_stem = splitext(basename(models[0]))[0]
        return f"{timestamp}_{model_stem}.dfp"
    return f"{timestamp}_models.dfp"

def _apply_ncconfig(nc: NeuralCompiler, cfg: Dict[str, Any]) -> None:
    """Apply a flat ncconfig.json dict to a NeuralCompiler instance using set_config().

    This function intentionally defers validation to the NeuralCompiler. If a key/value
    is invalid, NeuralCompiler will raise; the caller should log and surface that error.
    """
    for key, value in cfg.items():
        # Skip keys we explicitly manage/override elsewhere in the toolchain.
        if key in {"dfp_fname", "num_processes", "models"}:
            continue

        # # Normalize extensions into a list; NeuralCompiler expects a list.
        # if key == "extensions":
        #     value = _normalize_extensions(value)

        nc.set_config(**{key: value})


def _override_toolchain_config(
    nc: NeuralCompiler,
    *,
    dfp_path: str,
    num_processes: int,
) -> None:
    """Force toolchain-controlled config values onto NeuralCompiler.

    Args:
        dfp_path: Absolute path for the output .dfp file in the workspace.
        num_processes: Forced compilation process count (cpu_count - 2, floored at 1).
    """
    nc.set_config(dfp_fname=dfp_path)
    nc.set_config(num_processes=num_processes)


def _resolved_compiler_settings_for_logs(cfg: Dict[str, Any], *, num_processes: int, dfp_path: str) -> Dict[str, Any]:
    """Create a compact log-friendly view of compiler settings.

    This explains what the toolchain will apply (including forced overrides),
    without attempting to fully mirror NeuralCompiler's internal config.
    """
    out: Dict[str, Any] = dict(cfg)

    # Toolchain overrides win.
    out["dfp_fname"] = dfp_path
    out["num_processes"] = num_processes
    return out


@dataclass(frozen=True)
class CompilePlan:
    work_dir: str
    cfg: Dict[str, Any]
    models: List[str]
    onnx_model_name: str
    dfp_path: str
    chain_json_path: str
    num_processes: int
    timestamp: str


def _plan_from_archive(model_zip_path: str) -> CompilePlan:
    """Create a compile plan from a model archive (.zip).

    The archive is extracted into a unique workspace directory, ncconfig.json is loaded,
    and the referenced ONNX model files are resolved.

    Toolchain-controlled values are computed here:
      - timestamp tag
      - forced num_processes = cpu_count - 2 (floored at 1)
      - dfp_fname override path (workspace-local)

    Args:
        model_zip_path: Path to a zip archive containing .onnx file(s) and ncconfig.json.

    Returns:
        A CompilePlan containing resolved paths, config, and toolchain overrides.
    """
    work_dir = tempfile.mkdtemp(prefix="mxa_compile_")

    _extract_model_archive(model_zip_path, work_dir)
    cfg = _load_ncconfig(work_dir)
    models = _resolve_models_from_config(cfg, work_dir)

    ts = _timestamp_tag()
    num_processes = _forced_num_processes()

    # Use the first model name for folder naming and output packaging.
    onnx_model_name = splitext(basename(models[0]))[0]

    # Force dfp output name based on single vs multi-model.
    dfp_basename = _dfp_filename_for_models(models, ts)
    dfp_path = join(work_dir, dfp_basename)

    # chain.json is always workspace-local; packaged into the final output zip.
    chain_json_path = join(work_dir, "chain.json")

    return CompilePlan(
        work_dir=work_dir,
        cfg=cfg,
        models=models,
        onnx_model_name=onnx_model_name,
        dfp_path=dfp_path,
        chain_json_path=chain_json_path,
        num_processes=num_processes,
        timestamp=ts,
    )

def _package_artifacts(
    *,
    output_dir: str,
    model_name: str,
    dfp_path: str,
    chain_json_path: str,
    pre_onnx_path: Optional[str] = None,
    post_onnx_path: Optional[str] = None,
) -> str:
    """Package compiler artifacts into a zip under a model-specific output directory.

    Output layout:
        <output_dir>/<model_name>/<model_name>.zip

    Args:
        output_dir: Base output directory provided by the user.
        model_name: Model name used to create the subdirectory and zip filename.
        dfp_path: Path to the compiled .dfp file.
        chain_json_path: Path to chain.json to include in the archive.
        pre_onnx_path: Optional path to a pre-processing ONNX model.
        post_onnx_path: Optional path to a post-processing ONNX model.

    Returns:
        The path to the written zip file.
    """
    model_out_dir = join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    zip_path = join(model_out_dir, f"{model_name}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        if pre_onnx_path:
            zipf.write(pre_onnx_path, basename(pre_onnx_path))
        if post_onnx_path:
            zipf.write(post_onnx_path, basename(post_onnx_path))
        zipf.write(dfp_path, basename(dfp_path))
        zipf.write(chain_json_path, basename(chain_json_path))

    return zip_path


def onnx_to_mxa(model_path: str, output_dir: str) -> None:
    """Compile a model archive (.zip) into MemryX artifacts.

    The input archive must contain:
      - ncconfig.json (flat JSON object)
      - one or more .onnx files referenced by cfg["models"]

    Args:
        model_path: Path to the input zip archive.
        output_dir: Output directory where artifacts will be written.
    """
    plan = _plan_from_archive(model_path)

    logs.add_message(
        "Input archive extracted",
        {
            "Workspace": plan.work_dir,
            "Models": [basename(m) for m in plan.models],
        },
    )

    # Log the effective settings (config + toolchain overrides).
    effective_settings = _resolved_compiler_settings_for_logs(
        plan.cfg,
        num_processes=plan.num_processes,
        dfp_path=plan.dfp_path,
    )
    logs.add_message("Compiler settings (effective)", effective_settings)

    # Run the compiler with work_dir as CWD so all emitted files land there.
    cwd = os.getcwd()
    try:
        os.chdir(plan.work_dir)

        nc = NeuralCompiler()
        nc.reset_config()

        # Apply user config (flat) via set_config; NeuralCompiler owns validation.
        _apply_ncconfig(nc, plan.cfg)

        # Ensure models are set to the extracted workspace paths.
        # This allows co-mapping by listing multiple ONNX files in cfg["models"].
        nc.set_config(models=plan.models)

        # Toolchain overrides always win.
        _override_toolchain_config(
            nc,
            dfp_path=plan.dfp_path,
            num_processes=plan.num_processes,
        )

        # Compile.
        dfp = nc.run()
        logs.add_message("Compilation successful")

        # IO names from the compiled DFP object.
        dfp_io_names = _get_dfp_io_names(dfp)

        # Discover emitted pre/post ONNX files inside this workspace.
        pre_matches = sorted(glob(join(plan.work_dir, "*_pre.onnx")))
        post_matches = sorted(glob(join(plan.work_dir, "*_post.onnx")))

        pre_onnx_path = pre_matches[0] if pre_matches else None
        post_onnx_path = post_matches[0] if post_matches else None

        logs.add_data(
            **{
                "Found pre file": pre_onnx_path is not None,
                "Found post file": post_onnx_path is not None,
            }
        )

        # Create uniquely named copies of artifacts for packaging.
        pre_onnx_path, dfp_path, post_onnx_path = _rename_files(pre_onnx_path, plan.dfp_path, post_onnx_path)

        # Build chain.json describing the multi-stage execution chain.
        chain_json = build_chain_json(
            onnx_path=plan.models[0],  # metadata reference; first model is sufficient for naming purposes
            pre_path=pre_onnx_path,
            dfp_path=dfp_path,
            post_path=post_onnx_path,
            logs_dir=plan.work_dir,
            dfp_io_names=dfp_io_names,
        )
        with open(plan.chain_json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(chain_json, indent=3))

        # Package artifacts into a zip placed inside the model-specific directory.
        zip_path = _package_artifacts(
            output_dir=output_dir,
            model_name=plan.onnx_model_name,
            dfp_path=dfp_path,
            chain_json_path=plan.chain_json_path,
            pre_onnx_path=pre_onnx_path,
            post_onnx_path=post_onnx_path,
        )

        logs.add_message("Packaged artifacts", {"Zip Path": zip_path})

    finally:
        os.chdir(cwd)
        # Intentionally not deleting plan.work_dir to preserve artifacts for debugging.


def build_chain_json(
    onnx_path: str,
    pre_path: Optional[str],
    dfp_path: str,
    post_path: Optional[str],
    logs_dir: Optional[str] = None,
    dfp_io_names=None
) -> List[Dict[str, Any]]:
    """Build the chain.json structure from pre/post ONNX and the compiled DFP.

    Args:
        onnx_path: Path to the original ONNX model
        pre_path:  Path to the emitted pre.onnx (or None)
        dfp_path:  Path to the compiled .dfp
        post_path: Path to the emitted post.onnx (or None)
        logs_dir:  Directory to search for compiler log(s); defaults to CWD

    Returns:
        A list of stage objects suitable for JSON serialization.
    """
    all_io_info = _get_io_info(onnx_path)

    # DFP IO names are provided by the compiled DFP when available,
    # with compiler-log parsing as a fallback.

    if dfp_io_names is None:
        dfp_io_names = read_dfp_io_from_logs(logs_dir=logs_dir)

    # Pre/post IO info depends on whether those stages exist.
    pre_io_info = _get_io_info(pre_path)
    post_io_info = _get_io_info(post_path)

    # Resolve DFP input metadata from either original model inputs or pre-stage outputs.
    dfp_inputs_info: List[Dict[str, Any]] = []
    for dfp_input in dfp_io_names["Inputs"]:
        for onnx_input in all_io_info["Inputs"]:
            if dfp_input == onnx_input["Name"]:
                dfp_inputs_info.append(onnx_input)
                break
        for pre_output in pre_io_info["Outputs"]:
            if dfp_input == pre_output["Name"]:
                dfp_inputs_info.append(pre_output)
                break

    # Resolve DFP output metadata from either original model outputs or post-stage inputs.
    dfp_outputs_info: List[Dict[str, Any]] = []
    for dfp_output in dfp_io_names["Outputs"]:
        for onnx_output in all_io_info["Outputs"]:
            if dfp_output == onnx_output["Name"]:
                dfp_outputs_info.append(onnx_output)
                break
        for post_input in post_io_info["Inputs"]:
            if dfp_output == post_input["Name"]:
                dfp_outputs_info.append(post_input)
                break

    chain: List[Dict[str, Any]] = []

    if pre_path is not None:
        chain.append({
            "ModelPath": split(pre_path)[1],
            "RuntimePath": "",
            "InputNames": [i["Name"] for i in pre_io_info["Inputs"]],
        })

    chain.append({
        "ModelPath": split(dfp_path)[1],
        "RuntimePath": "",
        "InputNames": dfp_io_names["Inputs"],
        "RuntimeArgs": {
            "Inputs": dfp_inputs_info,
            "Outputs": dfp_outputs_info
        }
    })

    if post_path is not None:
        chain.append({
            "ModelPath": split(post_path)[1],
            "RuntimePath": "",
            "InputNames": [i["Name"] for i in post_io_info["Inputs"]],
        })

    return chain


def _rename_files(pre_path: Optional[str], dfp_path: str, post_path: Optional[str]):
    """Create uniquely named copies of compiler artifacts using a human-readable timestamp prefix.

    Artifacts are copied to:
        <YYYYMMDD_HHMMSS>_<original_filename>

    This preserves the original filenames while ensuring uniqueness and readability.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _prefixed(path: str) -> str:
        return join(dirname(path), f"{timestamp}_{basename(path)}")

    new_pre_path = None
    if pre_path is not None:
        new_pre_path = _prefixed(pre_path)
        shutil.copy(pre_path, new_pre_path)

    new_post_path = None
    if post_path is not None:
        new_post_path = _prefixed(post_path)
        shutil.copy(post_path, new_post_path)

    new_dfp_path = _prefixed(dfp_path)
    shutil.copy(dfp_path, new_dfp_path)

    return new_pre_path, new_dfp_path, new_post_path


def _get_io_info(onnx_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract input and output metadata from an ONNX model.

    This function reads the ONNX graph and extracts logical input/output
    names along with their tensor shapes and element data types.

    Args:
        onnx_path: Path to the ONNX model file. If None, an empty
                   input/output structure is returned.

    Returns:
        A dictionary with the following structure:
            {
                "Inputs": [
                    {
                        "Name": <input name>,
                        "Shape": <list of dimension sizes>,
                        "DataType": <ONNX element type enum>
                    },
                    ...
                ],
                "Outputs": [
                    {
                        "Name": <output name>,
                        "Shape": <list of dimension sizes>,
                        "DataType": <ONNX element type enum>
                    },
                    ...
                ]
            }

    Raises:
        onnx.onnx_cpp2py_export.checker.ValidationError:
            If the ONNX model is invalid or cannot be loaded.
        IOError:
            If the file cannot be read.
    """
    if onnx_path is None:
        return {'Inputs': [], 'Outputs': []}

    model = onnx.load(onnx_path)

    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    input_shapes = [i.type.tensor_type.shape.dim for i in model.graph.input]
    input_shapes = [[d.dim_value if d.dim_value > 0 else 1 for d in shape] for shape in input_shapes]

    output_shapes = [o.type.tensor_type.shape.dim for o in model.graph.output]
    output_shapes = [[d.dim_value if d.dim_value > 0 else 1 for d in shape] for shape in output_shapes]

    input_data_types = [i.type.tensor_type.elem_type for i in model.graph.input]
    output_data_types = [o.type.tensor_type.elem_type for o in model.graph.output]

    return {
        "Inputs": [
            {"Name": input_names[i], "Shape": input_shapes[i], "DataType": input_data_types[i]}
            for i in range(len(input_names))
        ],
        "Outputs": [
            {"Name": output_names[i], "Shape": output_shapes[i], "DataType": output_data_types[i]}
            for i in range(len(output_names))
        ]
    }


def _get_dfp_io_names(dfp):
    """Extract DFP input and output layer names from a compiled Dfp object.

    Args:
        dfp: Compiled Dfp object returned by NeuralCompiler.run().

    Returns:
        A dictionary with the following structure:
            {
                "Inputs":  [<input layer names>],
                "Outputs": [<output layer names>]
            }

    Raises:
        AttributeError: If the Dfp object does not expose the expected attributes.
    """
    out_names = list(getattr(dfp, "output_names", []) or [])

    in_names = []
    input_ports = getattr(dfp, "input_ports", None) or {}
    for _, port_info in input_ports.items():
        if isinstance(port_info, dict) and port_info.get("active") is False:
            continue
        if isinstance(port_info, dict) and "layer_name" in port_info:
            in_names.append(port_info["layer_name"])

    return {
        "Inputs": in_names,
        "Outputs": out_names,
    }


def read_dfp_io_from_logs(logs_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Parse compiler logs to extract DFP input/output layer names.

    Args:
        logs_dir: Directory to search for .log files. Defaults to CWD.

    Returns:
        {"Inputs": [...], "Outputs": [...]}

    Raises:
        RuntimeError if no suitable log is found or ports cannot be parsed.
    """
    search_dir = logs_dir or '.'
    candidates = glob(join(search_dir, '*.log'))

    if not candidates:
        raise RuntimeError(f"Couldn't find a .log file in: {search_dir}")

    # Choose the newest log deterministically (mtime).
    log_file = max(candidates, key=lambda p: os.path.getmtime(p))

    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        log = f.read()

    # Emit the log content for debugging (kept to preserve current behavior).
    print(log)

    input_port_pattern = re.compile(r'MPU \d+ input port \d+: ({.*})')
    output_port_pattern = re.compile(r'MPU \d+ output port \d+: ({.*})')

    input_port_matches = input_port_pattern.findall(log)
    output_port_matches = output_port_pattern.findall(log)

    input_ports = [json.loads(m.replace("'", '"')) for m in input_port_matches]
    output_ports = [json.loads(m.replace("'", '"')) for m in output_port_matches]

    if not input_ports or not output_ports:
        raise RuntimeError("Couldn't find the input and/or output ports in the compiler logs.")

    return {
        'Inputs': [i['layer_name'] for i in input_ports],
        'Outputs': [o['layer_name'] for o in output_ports]
    }

