from typing import Any, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import onnx
import onnxruntime as ort


def add_feature_concat(
    col_names: List[str],
    algorithm: Optional[Any] = None,
    pipeline: Optional[Pipeline] = None,
    algorithm_name: str = 'algorithm',
) -> Optional[Pipeline]:
    """Function that adds a column transformer to an algorithm so that the features are independent.

    This function either creates a sklearn pipeline with the column transformer and the algorithm
    (in that order), or inserts the column transformer at the beginning of an already existing
    pipeline. This way, when the model is converted to onnx, the inputs can be passed separated and
    the model compacts them into a single array. For that purpose, either `algorithm` is passed or
    `pipeline` is passed.

    Parameters
    ----------
    col_names : List[str]
        List of the column names to be passed to the column transformer. They must match the names
        in the dataset
    algorithm : Optional[Any]
        One of the sklearn many ML algorithms
    pipeline : Optional[Pipeline]
        A sklearn pipeline to which to insert the column transformer
    algorithm_name : Optional[str]
        A name for the algorithm step in the pipeline

    Returns
    -------
    Optional[Pipeline]
        If no pipeline is passed, the return will be a pipeline. Otherwise, the column transformer
        will be added to the pipeline itself and no return is necessary
    """
    if pipeline is None and algorithm is None:
        raise ValueError('either pipeline or algorithm must be passed')

    preprocessor = ColumnTransformer(
        transformers=[('featureConcat', 'passthrough', col_names)]
    )

    if pipeline is None:
        return Pipeline([('preprocessor', preprocessor), (algorithm_name, algorithm)])
    elif isinstance(pipeline, Pipeline):
        pipeline.steps.insert(0, ('preprocessor', preprocessor))
    else:
        raise ValueError(f'pipeline is not of the correct type ({type(Pipeline)})')


def optimize_graph(
    input_model_filename: str,
    optimized_model_filename: Optional[str] = None,
    op_level: Optional[str] = "ORT_ENABLE_EXTENDED"
) -> None:
    """    ONNX Runtime optimization

    This function optimizes the Graph in `input_model_filename` by removing unused nodes and
    simplifying the connectors, given the optimization level in `op_level`. The optimized graph
    is saved to `optimized_model_filename`.

    ONNX Runtime defines the GraphOptimizationLevel enum to determine which optimization levels
    will be enabled. Choosing a level enables the optimizations of that level, as well as the
    optimizations of all preceding levels. For example, enabling Extended optimizations, also
    enables Basic optimizations.

    Parameters
    ----------
    input_model_filename : str

    optimized_model_filename: Optional[str] = None,
    op_level: Optional[str]
     Optimization level, default = "ORT_ENABLE_EXTENDED"
        ORT_DISABLE_ALL -> Disables all optimizations
        ORT_ENABLE_BASIC -> Enables basic optimizations
        ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
        ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations
    """
    sess_options = ort.SessionOptions()

    # Set graph optimization level
    if op_level == "ORT_DISABLE_ALL":
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif op_level == "ORT_ENABLE_BASIC":
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif op_level == "ORT_ENABLE_EXTENDED":
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif op_level == "ORT_ENABLE_ALL":
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        raise ValueError("Unkown optimization level")

    if optimized_model_filename is None:
        optimized_model_filename = input_model_filename.replace(".onnx", "_optimized.onnx")

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = optimized_model_filename
    ort.InferenceSession(input_model_filename, sess_options)


def rename_onnx_outputs(
    onnx_model_path: str,
    input_mapping: List,
    output_mapping: List,
    save_onnx_filename: str,
    output_node_to_replace: str = "all",
) -> None:
    """Function that renames the output labels from an onnx format.

    The function maps the current output values `input_mapping`, into the new values as passed in
    `output_mapping` and saves the modified model in `save_onnx_filename`.

    Parameters
    ----------
    onnx_model_path : str
        Original ONNX model to be modified
    input_mapping : List[float]
        List of current values of the model in `onnx_model_path`
    output_mapping : List[int | float | str]
        List of new values of the model to be saved in `save_onnx_filename`
    save_onnx_filename: str
        Directory where the ONNX model with new outputs will be stored.
    """
    # Load onnx model
    onnx_model = onnx.load_model(onnx_model_path)

    # Make Label Encoder Node which will be our node for doing the mapping
    label_enc, output_value_info = _get_label_encoder_node(input_mapping, output_mapping)

    # Get Model Graph
    graph = onnx_model.graph

    # Attach this node to graph
    graph.node.append(label_enc)

    # Removing the previous outputs
    if output_node_to_replace == "all":
        while len(graph.output) > 0:
            graph.output.pop()
    else:
        # Find index of selected node to be replaced
        existing_output_index = -1
        for i, output in enumerate(graph.output):
            if output.name == output_node_to_replace:
                existing_output_index = i
                break
        if existing_output_index == -1:
            raise ValueError(f"{output_node_to_replace} is not an output of the ONNX model.")

        graph.output.pop(existing_output_index)

    # Add this new node as the output of the model - named as `new_label`
    graph.output.append(output_value_info)

    # We need to set the ai.onnx.ml opset version to 2 because we are using the new version of
    # LabelEncoder, probably there is a better way
    onnx_model.opset_import[1].version = 2

    # Export model to a file
    with open(save_onnx_filename, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def _get_label_encoder_node(input_mapping, output_mapping):

    if isinstance(output_mapping[0], int):
        encoder_params = {"keys_int64s": input_mapping, "values_int64s": output_mapping}
        output_value_info = onnx.helper.make_tensor_value_info(
            'new_label', onnx.TensorProto.INT64, shape=[None])
    elif isinstance(output_mapping[0], float):
        encoder_params = {"keys_int64s": input_mapping, "values_floats": output_mapping}
        output_value_info = onnx.helper.make_tensor_value_info(
            'new_label', onnx.TensorProto.FLOAT, shape=[None])
    elif isinstance(output_mapping[0], str):
        encoder_params = {"keys_int64s": input_mapping, "values_strings": output_mapping}
        output_value_info = onnx.helper.make_tensor_value_info(
            'new_label', onnx.TensorProto.STRING, shape=[None])
    else:
        raise ValueError(f'Incorrect output type {type(output_mapping[0])} cannot be mapped')

    label_encoder = onnx.helper.make_node(
            'LabelEncoder',         # node operator
            inputs=['label'],       # input name
            outputs=['new_label'],  # output name
            name='LabelEncoder',    # node name
            domain='ai.onnx.ml',    # need to set domain because it comes from
                                    # https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.FeatureVectorizer
            **encoder_params
        )
    return label_encoder, output_value_info
