import onnx
import re
import json
import shutil
import time
import zipfile
from glob import glob
from os.path import *

from memryx import NeuralCompiler

from .logger import logs

__here__ = dirname(__file__)

mxa_num_chips = 4
mxa_chip_gen = 'mx3'
mxa_auto_crop = True


def onnx_to_mxa(onnx_path, output_dir):
    tmp_dir = '/tmp'
    onnx_model_name = splitext(basename(onnx_path))[0]
    dfp_path = join(tmp_dir, f'{onnx_model_name}.dfp')

    # Compile MobileNet to 4 chips
    logs.add_message('Compilation configuration', 
                        {'Number of Chips': mxa_num_chips, 'Chip Generation': mxa_chip_gen, 'Auto Crop': mxa_auto_crop})
    nc = NeuralCompiler(models=onnx_path, dfp_fname=dfp_path,
                        num_chips=mxa_num_chips, chip_gen=mxa_chip_gen, autocrop=mxa_auto_crop,
                        show_optimization=False)

    # Run the compiler
    dfp = nc.run()

    logs.add_message('Compilation successful')
    # Get the path to the "*_pre.onnx" file
    pre_onnx_path = glob(join('.', '*_pre.onnx'))  # FIXME: Fix this dangerous glob syntax
    if not pre_onnx_path:
        pre_onnx_path = None
    else:
        pre_onnx_path = pre_onnx_path[0]

    # Get the path to the "*_post.onnx" file
    post_onnx_path = glob(join('.', '*_post.onnx'))  # FIXME: Fix this dangerous glob syntax
    if not post_onnx_path:
        post_onnx_path = None
    else:
        post_onnx_path = post_onnx_path[0]

    logs.add_data(**{"Found pre file": pre_onnx_path is not None, "Found post file": post_onnx_path is not None})

    # Rename the pre, post, and dfp files to unique names
    pre_onnx_path, dfp_path, post_onnx_path = _rename_files(pre_onnx_path, dfp_path, post_onnx_path)

    # Build the chain.json file
    json_path = join(tmp_dir, f'chain.json')
    chain_json = build_chain_json(onnx_path, pre_onnx_path, dfp_path, post_onnx_path)
    with open(json_path, 'w') as f:
        f.write(json.dumps(chain_json, indent=3))

    # Zip pre, post, dfp, and json files
    zip_path = join(output_dir, f'{onnx_model_name}.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if pre_onnx_path:
            zipf.write(pre_onnx_path, basename(pre_onnx_path))
        if post_onnx_path:
            zipf.write(post_onnx_path, basename(post_onnx_path))
        zipf.write(dfp_path, basename(dfp_path))
        zipf.write(json_path, basename(json_path))


def build_chain_json(onnx_path: str, pre_path: str, dfp_path: str, post_path: str):
    """ Build the chain.json file from the pre, post, and dfp files

    The chain.json file is a JSON file that has this structure:
        [
           {
              "ModelPath":"pre.onnx",
              "RuntimePath":"libRuntime.so",
               "InputNames":["sss", "ppp"],
              "RuntimeArgs":{} // Optional
           },
           {
              "ModelPath":"dfp.dfp",
              "RuntimePath":"libRuntime2.so",
              "InputNames":["sss", "ppp"],
              "RuntimeArgs":{"Inputs":[], "Outputs":[]} // Optional
           }
        ]

    Args:
        - onnx_path: The path to the original ONNX model
        - pre_path: The path to the pre.onnx file
        - dfp_path: The path to the dfp.dfp file
        - post_path: The path to the post.onnx file

    Returns:
        - The chain JSON object
    """
    all_io_info = _get_io_info(onnx_path)

    dfp_io_names = read_dfp_io_from_logs()

    # check if the pre file is not None
    pre_io_info = _get_io_info(pre_path)

    # Search for the DFP inputs in the pre and onnx files
    dfp_inputs_info = []
    for dfp_input in dfp_io_names["Inputs"]:
        # Look for the input in the onnx file
        for onnx_input in all_io_info["Inputs"]:
            if dfp_input == onnx_input["Name"]:
                dfp_inputs_info += [onnx_input]
                break
        # Look for the input in the pre file
        for pre_output in pre_io_info["Outputs"]:
            if dfp_input == pre_output["Name"]:
                dfp_inputs_info += [pre_output]
                break

    # check if the post file is not None
    post_io_info = _get_io_info(post_path)
    # Search for the DFP outputs in the post and onnx files
    dfp_outputs_info = []
    for dfp_output in dfp_io_names["Outputs"]:
        # Look for the output in the onnx file
        for onnx_output in all_io_info["Outputs"]:
            if dfp_output == onnx_output["Name"]:
                dfp_outputs_info += [onnx_output]
                break
        # Look for the output in the post file
        for post_input in post_io_info["Inputs"]:
            if dfp_output == post_input["Name"]:
                dfp_outputs_info += [post_input]
                break

    chain = []
    if pre_path is not None:
        chain.append({
            "ModelPath": split(pre_path)[1],
            "RuntimePath": "",
            "InputNames": [input_info["Name"] for input_info in pre_io_info["Inputs"]],
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
            "InputNames": [input_info["Name"] for input_info in post_io_info["Inputs"]],
        })

    return chain


def _rename_files(pre_path: str, dfp_path: str, post_path: str):
    """ Rename the pre, dfp, and post files to unique names

    Args:
        - pre_path: The path to the pre file
        - dfp_path: The path to the dfp file
        - post_path: The path to the post file

    Returns:
        - The new paths of the pre, dfp, and post files
    """
    # get current timestamp in nanoseconds
    timestamp = int(time.time_ns())

    # rename the pre file
    new_pre_path = None
    if pre_path is not None:
        new_pre_path = join(dirname(pre_path), f'{timestamp}_pre.onnx')
        shutil.copy(pre_path, new_pre_path)

    # rename the post file
    new_post_path = None
    if post_path is not None:
        new_post_path = join(dirname(post_path), f'{timestamp}_post.onnx')
        shutil.copy(post_path, new_post_path)

    # rename the dfp file
    new_dfp_path = join(dirname(dfp_path), f'{timestamp}.dfp')
    shutil.copy(dfp_path, new_dfp_path)

    return new_pre_path, new_dfp_path, new_post_path


def _get_io_info(onnx_path: str):
    """ Get the input and output names of the ONNX model

    Args:
        - onnx_path: The path to the ONNX model

    Returns:
        - A dictionary containing the input and output names of the ONNX model
    """
    if onnx_path is None:
        return {'Inputs': [], 'Outputs': []}

    model = onnx.load(onnx_path)
    # Get input names
    input_names = [input.name for input in model.graph.input]
    # Get output names
    output_names = [output.name for output in model.graph.output]
    # Get input shape
    input_shapes = [input.type.tensor_type.shape.dim for input in model.graph.input]
    input_shapes = [[dim.dim_value if dim.dim_value > 0 else 1 for dim in shape] for shape in input_shapes]
    # Get output shape
    output_shapes = [output.type.tensor_type.shape.dim for output in model.graph.output]
    output_shapes = [[dim.dim_value if dim.dim_value > 0 else 1 for dim in shape] for shape in output_shapes]
    # Get input data type
    input_data_types = [input.type.tensor_type.elem_type for input in model.graph.input]
    # Get output data type
    output_data_types = [output.type.tensor_type.elem_type for output in model.graph.output]

    io_info = {
        "Inputs": [
            {
                "Name": input_names[i],
                "Shape": input_shapes[i],
                "DataType": input_data_types[i]
            } for i in range(len(input_names))
        ],
        "Outputs": [
            {
                "Name": output_names[i],
                "Shape": output_shapes[i],
                "DataType": output_data_types[i]
            } for i in range(len(output_names))
        ]
    }

    return io_info

def read_dfp_io_from_logs():
    log_file = glob(join('.', '*.log'))  # FIXME: Fix this dangerous glob syntax
    if len(log_file) == 0:
        print("Couldn't find the logs file.")
        exit(1)
    
    log_file = log_file[0]
    with open(log_file, 'r') as f:
        log = f.read()

    print(log)

    # Define regular expressions for matching input and output port lines
    input_port_pattern = re.compile(r'MPU \d+ input port \d+: ({.*})')
    output_port_pattern = re.compile(r'MPU \d+ output port \d+: ({.*})')

    # Find all matches in the log
    input_port_matches = input_port_pattern.findall(log)
    output_port_matches = output_port_pattern.findall(log)

    # Convert the matched strings to JSON
    input_ports = [json.loads(match.replace("'", '"')) for match in input_port_matches]
    output_ports = [json.loads(match.replace("'", '"')) for match in output_port_matches]

    # check if both lists are not empty
    if len(input_ports) == 0 or len(output_ports) == 0:
        print("Couldn't find the input and output ports in the logs.")
        exit(1)

    # Combine input and output ports into a single JSON object
    mpu_ports_json = {
        'Inputs': [i['layer_name'] for i in input_ports],
        'Outputs': [o['layer_name'] for o in output_ports]
    }

    return mpu_ports_json