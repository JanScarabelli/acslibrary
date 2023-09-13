import subprocess
import shutil
from typing import Optional, List, Any

import numpy as np
import torch
import tensorflow

import onnx
from skl2onnx import to_onnx
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from xgboost import XGBClassifier, XGBRegressor
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert import convert_xgboost as convert_xgboost_booster
import onnxruntime as ort


def pytorch_to_onnx(
    torch_model: torch.nn,
    model_input: torch.tensor,
    save_onnx_filename: str,
    export_params: bool = True,
    opset_version: Optional[int] = None,
    input_names: Optional[List[str]] = ["input"],
    output_names: Optional[List[str]] = ["output"],
    dynamic_axes: Optional[dict] = {
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
) -> None:
    """Function that exports a pytorch model to onnx format.

    Parameters
    ----------
    torch_model: torch.nn
        Pytorch model to convert
    model_input: torch.tensor
        Model input (used for determining input types)
    save_onnx_filename: str
        Save onnx model location
    export_params: bool = True
        Store the trained parameter weights inside the model file, True by default
    opset_version: Optional[int] = None
        The version of the `default (ai.onnx) opset
        <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`
        to target. Must be >= 7 and <= 16.
    input_names: Optional[List[str]],
        The model's input names
        Ex: ['input']
    output_names: Optional[List[str]],
        The model's output names
        Ex: ['output']
    dynamic_axes: Optional[dict]
        Parameter to allow variable length axes
        Ex: {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'},}
    """

    # Trace the model to obtain a TorchScript representation
    traced_model = torch.jit.trace(torch_model, model_input)

    torch.onnx.export(
        traced_model,
        model_input,
        save_onnx_filename,
        export_params=export_params,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # variable length axes
    )


def sklearn_to_onnx(
    sklearn_model,
    X: Optional[torch.tensor] = None,
    initial_types=Optional[List[tuple]],
    save_onnx_filename: Optional[str] = None,
    target_opset: Optional[int] = None,
    model_optim: Optional[bool] = True,
) -> Optional[onnx.GraphProto]:
    """Function that exports an sklearn model to onnx format.

    Parameters
    ----------
    sklearn_model : Model / Pipeline [sklearn]
        Model or Pipeline created with scikit-learn, to be exported
    X: Optional[torch.tensor]
        training set, can be None, it is used to infered the input types (*initial_types*)
    initial_types: Optional[List[tuple]]
        if X is None, then *initial_types* must be
        defined
    save_onnx_filename: Optional[str]
        save onnx location, if None the function returns the converted model
    target_opset: Optional[int]
        conversion with a specific target opset
    model_optim: Optional[bool]
        enable or disable model optimisation after the model was converted into onnx, it reduces
        the number of identity nodes

    Returns
    -------
    converted model
    """
    if X is not None:
        onnx_model = to_onnx(
            sklearn_model, X=X, target_opset=target_opset, model_optim=model_optim
        )
    elif initial_types is not None:
        onnx_model = to_onnx(
            sklearn_model,
            initial_types=initial_types,
            target_opset=target_opset,
            model_optim=model_optim,
        )

    if save_onnx_filename:
        # Save ONNX model instance to a ONNX file
        with open(save_onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

    return onnx_model


def tensorflow_to_onnx(
    tf_model: tensorflow.nn, save_onnx_filename: Optional[str]
) -> None:
    """Function that exports a tensorflow model to onnx format.

    This function takes the tensorflow model in `tf_model`, it saves it to a temporal directory
    and then uses tf2onnx command tool to export the model to onnx in `save_onnx_filename`.

    Parameters
    ----------
    tf_model : Model [Tensorflow/Keras]
        Model created with Tensorflow/Keras, to be exported
    save_onnx_filename : Path [str]
        Directory where the exported ONNX model will be stored.
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model
    tmp_path_tf_model = "./tests/models/tf_model"
    # Save model in SavedModel format
    tf_model.save(tmp_path_tf_model, save_format="tf")

    bashCommand = f"python -m tf2onnx.convert --saved-model {tmp_path_tf_model} --output {save_onnx_filename}"
    p = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    p.communicate()
    shutil.rmtree(tmp_path_tf_model)


def xgboost_to_onnx(
    xgboost_pipeline,
    xgb_mode: str,
    save_onnx_filename: str,
    initial_type: Optional[np.array] = None,
    X_train: Optional[np.array] = None,
) -> None:
    if xgb_mode == "classifier":
        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        onnx_model = convert_sklearn(
            xgboost_pipeline,
            "pipeline_xgboost",
            initial_types=initial_type,
            target_opset={"": 12, "ai.onnx.ml": 2},
        )

    elif xgb_mode == "regressor":
        update_registered_converter(
            XGBRegressor,
            "XGBoostXGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
        )

        onnx_model = to_onnx(
            xgboost_pipeline,
            X_train.astype(np.float64),
            target_opset={"": 12, "ai.onnx.ml": 2},
        )

    elif xgb_mode == "boost":
        # A booster cannot be inserted in a pipeline. It requires a different conversion function
        # because it does not follow scikit-learn API.
        try:
            onnx_model = convert_xgboost_booster(
                xgboost_pipeline, "name", initial_types=initial_type
            )
        except AssertionError as e:
            raise f"XGBoost is too recent or onnxmltools too old. {e}"
    else:
        raise f"Wrong xgb mode {xgb_mode}, choose between: classifier | regressor | boost"

    with open(save_onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())


def onnx_predict(
    inputs: List[np.array],
    onnx_model_path: Optional[str] = None,
    onnx_model: Optional[onnx.ModelProto] = None,
) -> dict:
    """Given the path of a onnx model or the onnx model itself, create a
    running session and return the inference outputs.

    Parameters
    ----------
    inputs : List[np.array]
        list of input values for onnx prediction
    onnx_model_path : Optional[str]
        path to onnx model file
    onnx_model : Optional[str]
        onnx model already loaded

    Returns
    -------
        dict with inference results, where keys = output_graph_names and
        values= inference results for each output
    """
    # Create inference session
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Hide warning messages
    if onnx_model_path:
        sess = ort.InferenceSession(onnx_model_path, sess_options=sess_options)
    elif onnx_model:
        sess = ort.InferenceSession(
            onnx_model_path.SerializeToString(), sess_options=sess_options
        )
    else:
        raise ValueError("either onnx_model or onnx_model_path must be passed")

    # Run inference
    try:
        inference_data = {k.name: v for k, v in zip(sess.get_inputs(), inputs)}
        inference_result = sess.run(None, inference_data)
        output_names = [output.name for output in sess.get_outputs()]
        result_dict = {
            out_name: out_val
            for out_name, out_val in zip(output_names, inference_result)
        }
        return result_dict
    except Exception as e:
        print("ONNXRuntime error: ", e)
        print("------")
        # Get the names, shapes, and types of the model inputs
        input_names = [input.name for input in sess.get_inputs()]
        input_shapes = [input.shape for input in sess.get_inputs()]
        input_types = [input.type for input in sess.get_inputs()]

        print("Current input shape and type: ")
        for i, name in enumerate(inputs):
            print(
                f"Input {i}: {name} - Shape: {inputs[i].shape} - Type: {inputs[i].dtype}"
            )
        print("Desired inputs shape and types: ")
        for i, name in enumerate(input_names):
            print(
                f"Input {i}: {name} - Shape: {input_shapes[i]} - Type: {input_types[i]}"
            )
