import json
import os
import re
import shutil
import time
import zipfile
from glob import glob
from os.path import basename, dirname, join, split, splitext
import tempfile
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from typing import Iterable
from datetime import datetime

import onnx
from memryx import NeuralCompiler

from .logger import logs

# Default compiler configuration.
mxa_chip_gen = 'mx3'
mxa_auto_crop = True

@dataclass(frozen=True)
class CompilePaths:
    work_dir: str
    onnx_model_name: str
    dfp_path: str
    chain_json_path: str


def _plan_paths(onnx_path: str) -> CompilePaths:
    # Unique workspace per run avoids collisions across concurrent executions.
    work_dir = tempfile.mkdtemp(prefix="mxa_compile_")

    # Model name derived from ONNX filename (stable, human-readable).
    onnx_model_name = splitext(basename(onnx_path))[0]

    # All intermediate artifacts live in the workspace.
    dfp_path = join(work_dir, f"{onnx_model_name}.dfp")
    chain_json_path = join(work_dir, "chain.json")

    return CompilePaths(
        work_dir=work_dir,
        onnx_model_name=onnx_model_name,
        dfp_path=dfp_path,
        chain_json_path=chain_json_path,
    )



def onnx_to_mxa(onnx_path: str, output_dir: str, mxa_num_chips: int, mxa_extensions: Optional[str]) -> None:

    # Centralized path planning for an isolated, per-run workspace.
    paths = _plan_paths(onnx_path)
    work_dir = paths.work_dir
    onnx_model_name = paths.onnx_model_name
    dfp_path = paths.dfp_path


    # os.cpu_count() can return None; fall back to 1 to avoid TypeError.
    cpu_count = os.cpu_count() or 1
    mxa_compile_processes = max(1, cpu_count - 1)
    mxa_extensions_list = _normalize_extensions(mxa_extensions)


    logs.add_message(
        'Compilation configuration',
        {
            'Number of Chips': mxa_num_chips,
            'Chip Generation': mxa_chip_gen,
            'Auto Crop': mxa_auto_crop,
            'Compile Processes': mxa_compile_processes,
            'Extensions':mxa_extensions_list,
        }
    )

    # Run the compiler with work_dir as CWD so all emitted files land there.
    cwd = os.getcwd()
    try:
        os.chdir(work_dir)

        nc = NeuralCompiler(
            models=onnx_path,
            dfp_fname=dfp_path,
            num_chips=mxa_num_chips,
            chip_gen=mxa_chip_gen,
            autocrop=mxa_auto_crop,
            extensions=mxa_extensions_list,
            effort='normal',
            num_processes=mxa_compile_processes,
            verbose=1,
            show_optimization=True,
        )

        dfp = nc.run()
        logs.add_message('Compilation successful')

        #get dfp io names from DFP property itself 
        dfp_io_names = _get_dfp_io_names(dfp)

        # Discover emitted pre/post ONNX files inside this workspace.
        pre_matches = sorted(glob(join(work_dir, '*_pre.onnx')))
        post_matches = sorted(glob(join(work_dir, '*_post.onnx')))

        pre_onnx_path = pre_matches[0] if pre_matches else None
        post_onnx_path = post_matches[0] if post_matches else None

        logs.add_data(
            **{
                "Found pre file": pre_onnx_path is not None,
                "Found post file": post_onnx_path is not None
            }
        )

        # Rename artifacts to unique names to avoid collisions.
        pre_onnx_path, dfp_path, post_onnx_path = _rename_files(pre_onnx_path, dfp_path, post_onnx_path)

        # Build chain.json describing the multi-stage execution chain.
        json_path = paths.chain_json_path
        chain_json = build_chain_json(
            onnx_path=onnx_path,
            pre_path=pre_onnx_path,
            dfp_path=dfp_path,
            post_path=post_onnx_path,
            logs_dir=work_dir,
            dfp_io_names=dfp_io_names
        )
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(chain_json, indent=3))

        model_out_dir = join(output_dir, onnx_model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        # Package artifacts into a zip placed in output_dir.
        zip_path = join(model_out_dir, f'{onnx_model_name}.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            if pre_onnx_path:
                zipf.write(pre_onnx_path, basename(pre_onnx_path))
            if post_onnx_path:
                zipf.write(post_onnx_path, basename(post_onnx_path))
            zipf.write(dfp_path, basename(dfp_path))
            zipf.write(json_path, basename(json_path))

    finally:
        os.chdir(cwd)
        # Intentionally not deleting work_dir to preserve artifacts for debugging.
        # If desired later: add a flag to clean up temp output on success.


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




def _normalize_extensions(ext) -> List[str]:
    """Normalize extensions into a list[str] suitable for NeuralCompiler.

    Accepts:
      - None
      - string: "foo" or "foo,bar" or '["foo","bar"]'
      - list/tuple of strings (including from argparse action='append')
      - mixed list where elements may be comma-separated strings

    Returns:
      A list of non-empty strings.
    """
    if ext is None:
        return []

    # If a single string was provided, support JSON list or comma-separated values.
    if isinstance(ext, str):
        s = ext.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [t.strip() for t in s.split(",") if t.strip()]

    # If argparse used action='append', ext is typically a list[str].
    if isinstance(ext, (list, tuple)):
        out: List[str] = []
        for item in ext:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                # Allow each element to be comma-separated too.
                out.extend([t.strip() for t in s.split(",") if t.strip()])
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out

    # Last-resort: coerce anything else to string.
    s = str(ext).strip()
    return [s] if s else []
