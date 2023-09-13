from typing import Optional, Dict
from itertools import combinations
from pathlib import Path
import uuid

import numpy as np
import sklearn
import torch
import onnx
from onnx import GraphProto, ModelProto
from onnx import numpy_helper
from onnx import helper
import onnxruntime as ort

SHAP_SAMPLE_LIMIT = 500

dtype_mapping_torch = {
    'tensor(float16)': torch.float16,
    'tensor(float32)': torch.float32,
    'tensor(float64)': torch.float64,
    'tensor(float)': torch.float32,
    'tensor(double)': torch.float64,
    'tensor(int8)': torch.int8,
    'tensor(int16)': torch.int16,
    'tensor(int32)': torch.int32,
    'tensor(int64)': torch.int64,
    'tensor(bool)': torch.bool,

}

dtype_mapping_onnx = {
    'tensor(float16)': onnx.TensorProto.FLOAT16,
    'tensor(float32)': onnx.TensorProto.FLOAT,
    'tensor(float64)': onnx.TensorProto.DOUBLE,
    'tensor(float)': onnx.TensorProto.FLOAT,
    'tensor(double)': onnx.TensorProto.DOUBLE,
    'tensor(int8)': onnx.TensorProto.INT8,
    'tensor(int16)': onnx.TensorProto.INT16,
    'tensor(int32)': onnx.TensorProto.INT32,
    'tensor(int64)': onnx.TensorProto.INT64,
    'tensor(bool)': onnx.TensorProto.BOOL,
}


@torch.jit.script
def avg(a, weights, dim: int = 0):
    output = torch.sum(torch.mul(a, weights), dim=dim)
    output = torch.div(output, torch.sum(weights))
    return output


@torch.jit.script
def fact(n: int):
    out = 1
    for i in range(2, n + 1):
        out += i
    return out


@torch.jit.script
def comb(n: int, k: int):
    return fact(n) / (fact(k) * fact(n - k))


@torch.jit.script
def pi_x(num_features: int, num_avaialable_features: int):
    return torch.tensor(
        (num_features - 1)
        / (
            comb(num_features, num_avaialable_features)
            * num_avaialable_features
            * (num_features - num_avaialable_features)
        )
    )


@torch.jit.script
def non_zero(tensor: torch.Tensor):
    count = 0
    for i in tensor.flatten():
        if i > 0:
            count += 1
    return count


@torch.jit.script
def generate_pi_values(coalition_vectors):
    pi_values = torch.tensor([torch.inf])
    num_features = len(coalition_vectors[0])
    for coalition_vector in coalition_vectors[1:]:
        num_avaialable_features = non_zero(coalition_vector)
        pi_values = torch.cat(
            (pi_values, pi_x(num_features, num_avaialable_features).reshape(1))
        )
    return pi_values


@torch.jit.script
def diag(val):
    diag_mtx = torch.zeros((len(val), len(val)))
    for i in range(len(val)):
        diag_mtx[i][i] = val[i]
    return diag_mtx


@torch.jit.script
def invert_matrix(M):
    n = M.shape[0]
    I = torch.eye(n=n)
    M = torch.cat((M, I), dim=1)
    M = torch.cat((M[torch.any(M != 0, dim=1)], M[torch.all(M == 0, dim=1)]), dim=0)
    for i in range(0, n):
        j = 1
        pivot = M[i][i]
        while pivot == 0 and i + j < n:
            M[[i, i + j]] = M[[i + j, i]]
            j += 1
            pivot = M[i][i]
        if pivot != 0:
            row = M[i]
            M[i] = row / pivot
            for k in range(0, n):
                if k != i:
                    M[k] = M[k] - M[i] * M[k][i]
    return M[:, n:]


def modify_graph(model, n, onnx_input_type):
    graph = model.graph
    graph.output.pop(0)

    for i in range(n):
        output_name = "output_" + str(i)
        indice = onnx.helper.make_tensor(
            "indice" + str(i), onnx.TensorProto.INT32, dims=[], vals=[i]
        )
        gather_node = onnx.helper.make_node(
            "Gather",
            inputs=["filled_X", "indice" + str(i)],
            outputs=[output_name],
            name=f"Gather_{i}",
            axis=0,
        )
        output_value_info = onnx.helper.make_tensor_value_info(
            output_name, onnx_input_type, shape=[None, 1]
        )
        graph.node.append(gather_node)
        graph.output.append(output_value_info)
        graph.initializer.append(indice)
    return model


def build_shap_part_1(X_train, instances, coalition_vector):
    """
    create_dataset basically creates combinations of original dataset by changing indexes
    with instance based on coalition vector.
        e.g.
        create_dataset(X = [[1,2], [3,4]], instance=[0,0]) returns:
        [[1, 2], [3, 4],
         [0, 2], [0, 4],
         [1, 0], [3, 0],
         [0, 0],
         [0, 0]]
    """
    N = X_train.shape[0]
    num_coalition_vectors = len(coalition_vector)
    filled_X = X_train[:1]  # filled_X : [[1,2]] -> dummy
    for idx, ins in enumerate(instances):
        filled_X = torch.cat((filled_X, X_train))  # filled_x : [[1,2], [1,2], [3,4]]
        for i in range(num_coalition_vectors):
            if i != num_coalition_vectors - 1:
                filled_X = torch.cat((filled_X, X_train))
            mask = coalition_vector[i].type(torch.bool)
            str_idx = (idx * N * num_coalition_vectors) + (i * N + 1)
            end_idx = (idx * N * num_coalition_vectors) + ((i + 1) * N + 1)
            # Add print lines to enable onnx conversion of the following torch.where line, do not remove
            # Short explanation: torch->onnx conversion does not interpret well dynamic variables mask and filled_X
            if i == 0:
                print(f"mask : {mask}, {mask.shape}")
                print(
                    f"cropped x: {filled_X[str_idx: end_idx, :]}, {filled_X[str_idx: end_idx, :].shape}"
                )
            filled_X[str_idx:end_idx, :] = torch.where(
                mask, ins, filled_X[str_idx:end_idx, :]
            )
        # [[1,2], -> dummy
        #  [1,2], [3,4],  0/0
        #  [0,2], [0,4],  1/0
        #  [1,0], [3,0],  0/1
        #  [0,0]]         1/1

    # remove first dummy input and append instance at the end of the tensor
    filled_X = torch.cat((filled_X, instances))[1:]
    # reshape to : [number_of_inputs=1, batch_size, features]
    filled_X = torch.reshape(filled_X, [1, filled_X.shape[0], filled_X.shape[1]])
    return filled_X, instances


@torch.jit.script
def calculate_shap_values(instance, preds, X, coalition_vectors):
    N = X.shape[0]
    num_coalition_vectors = len(coalition_vectors)
    num_instances = instance.shape[0]
    num_features = instance.shape[-1]
    labels = preds[-num_instances:]
    preds = preds[:-num_instances]
    shap_vals = torch.zeros((num_instances, num_features))
    preds = preds.reshape(num_instances, -1)

    weights = torch.ones(N) / N
    phi0 = avg(torch.squeeze(preds[0][:N]), weights)  # phi0 = expected value
    for i in range(0, num_instances):
        X = coalition_vectors
        pi_values = generate_pi_values(coalition_vectors)
        W = diag(pi_values)
        W[0][0] = torch.tensor(1e9)
        W[-1][-1] = torch.tensor(1e9)
        R = invert_matrix(X.T @ W @ X) @ (X.T @ W)

        y = (
            avg(preds[i].reshape(num_coalition_vectors, N), weights=weights, dim=1)
            - phi0
        )
        phi = R @ y
        shap_vals[i, :] = phi
    return labels, shap_vals, phi0


def add_prefix_graph(  # pylint: disable=too-many-branches
    graph: GraphProto,
    prefix: str,
    rename_nodes: Optional[bool] = True,
    rename_edges: Optional[bool] = True,
    rename_inputs: Optional[bool] = True,
    rename_outputs: Optional[bool] = True,
    rename_initializers: Optional[bool] = True,
    rename_value_infos: Optional[bool] = True,
    inplace: Optional[bool] = False,
    name_map: Optional[Dict[str, str]] = None,
) -> GraphProto:
    """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
    initializers, sparse initializer, value infos.
    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not prefixed.
    Arguments:
        graph (GraphProto): Graph
        prefix (str): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        rename_initializers (bool): Whether to prefix initializer and sparse initializer names
        rename_value_infos (bool): Whether to prefix value info names
        inplace (bool): If True, mutates the graph directly.
                        Otherwise, a copy will be created
        name_map: (Dict): shared name_map in subgraph
    Returns:
        GraphProto
    """
    if type(graph) is not GraphProto:
        raise ValueError("graph argument is not an ONNX graph")

    if not inplace:
        g = GraphProto()
        g.CopyFrom(graph)
    else:
        g = graph

    def _prefixed(prefix: str, name: str) -> str:
        return prefix + name if len(name) > 0 else name

    if name_map is None:
        name_map = {}
    if rename_edges:
        for n in g.node:
            for e in n.input:
                name_map[e] = _prefixed(prefix, e)
            for e in n.output:
                name_map[e] = _prefixed(prefix, e)
    else:
        if rename_outputs:
            for entry in g.output:
                name_map[entry.name] = _prefixed(prefix, entry.name)
        if rename_inputs:
            for entry in g.input:
                name_map[entry.name] = _prefixed(prefix, entry.name)

    if rename_nodes:
        for n in g.node:
            n.name = _prefixed(prefix, n.name)
            for attribute in n.attribute:
                if attribute.g:
                    add_prefix_graph(
                        attribute.g, prefix, inplace=True, name_map=name_map
                    )

    if rename_initializers:
        for init in g.initializer:
            name_map[init.name] = _prefixed(prefix, init.name)
        for sparse_init in g.sparse_initializer:
            name_map[sparse_init.values.name] = _prefixed(
                prefix, sparse_init.values.name
            )
            name_map[sparse_init.indices.name] = _prefixed(
                prefix, sparse_init.indices.name
            )

    if rename_value_infos:
        for entry in g.value_info:
            name_map[entry.name] = _prefixed(prefix, entry.name)

    for n in g.node:
        for i, output in enumerate(n.output):
            if n.output[i] in name_map:
                n.output[i] = name_map[output]
        for i, input_ in enumerate(n.input):
            if n.input[i] in name_map:
                n.input[i] = name_map[input_]

    for in_desc in g.input:
        if in_desc.name in name_map:
            in_desc.name = name_map[in_desc.name]
    for out_desc in g.output:
        if out_desc.name in name_map:
            out_desc.name = name_map[out_desc.name]

    for initializer in g.initializer:
        if initializer.name in name_map:
            initializer.name = name_map[initializer.name]
    for sparse_initializer in g.sparse_initializer:
        if sparse_initializer.values.name in name_map:
            sparse_initializer.values.name = name_map[sparse_initializer.values.name]
        if sparse_initializer.indices.name in name_map:
            sparse_initializer.indices.name = name_map[sparse_initializer.indices.name]

    for value_info in g.value_info:
        if value_info.name in name_map:
            value_info.name = name_map[value_info.name]

    return g


def add_prefix(
    model: ModelProto,
    prefix: str,
    rename_nodes: Optional[bool] = True,
    rename_edges: Optional[bool] = True,
    rename_inputs: Optional[bool] = True,
    rename_outputs: Optional[bool] = True,
    rename_initializers: Optional[bool] = True,
    rename_value_infos: Optional[bool] = True,
    rename_functions: Optional[bool] = True,
    inplace: Optional[bool] = False,
) -> ModelProto:
    """Adds a prefix to names of elements in a graph: nodes, edges, inputs, outputs,
    initializers, sparse initializer, value infos, and local functions.

    It can be used as a utility before merging graphs that have overlapping names.
    Empty names are not _prefixed.

    Arguments:
        model (ModelProto): Model
        prefix (str): Prefix to be added to each name in the graph
        rename_nodes (bool): Whether to prefix node names
        rename_edges (bool): Whether to prefix node edge names
        rename_inputs (bool): Whether to prefix input names
        rename_outputs (bool): Whether to prefix output names
        rename_initializers (bool): Whether to prefix initializer and sparse initializer names
        rename_value_infos (bool): Whether to prefix value info nanes
        rename_functions (bool): Whether to prefix local function names
        inplace (bool): If True, mutates the model directly.
                        Otherwise, a copy will be created

    Returns:
        ModelProto
    """
    if type(model) is not ModelProto:
        raise ValueError("model argument is not an ONNX model")

    if not inplace:
        m = ModelProto()
        m.CopyFrom(model)
        model = m

    add_prefix_graph(
        model.graph,
        prefix,
        rename_nodes=rename_nodes,
        rename_edges=rename_edges,
        rename_inputs=rename_inputs,
        rename_outputs=rename_outputs,
        rename_initializers=rename_initializers,
        rename_value_infos=rename_value_infos,
        inplace=True,  # No need to create a copy, since it's a new model
    )

    if rename_functions:
        f_name_map = {}
        for f in model.functions:
            new_f_name = prefix + f.name
            f_name_map[f.name] = new_f_name
            f.name = new_f_name
        # Adjust references to local functions in other local function
        # definitions
        for f in model.functions:
            for n in f.node:
                if n.op_type in f_name_map:
                    n.op_type = f_name_map[n.op_type]
        # Adjust references to local functions in the graph
        for n in model.graph.node:
            if n.op_type in f_name_map:
                n.op_type = f_name_map[n.op_type]

    return model


def separate_graph_inputs(
    model: onnx.ModelProto,
    num_feats: int,
    input_node_name: str,
    onnx_input_type,
    new_input_names: list
):
    # Define the new inputs with their shapes
    new_input_shapes = [(1, 1)]*num_feats

    # Create new input tensors for each of the new inputs
    new_inputs = []

    for i in range(num_feats):
        input_name = new_input_names[i]
        tensor = numpy_helper.from_array(np.zeros(new_input_shapes[i]), name=input_name)
        new_input = helper.make_tensor_value_info(input_name, onnx_input_type, tensor.dims)
        new_inputs.append(new_input)

    # Create a Concat node that takes the new inputs as input
    concat_node = helper.make_node('Concat',
                                   inputs=[new_inputs[i].name for i in range(num_feats)],
                                   outputs=[input_node_name], axis=1)

    # Modify the input of the following node to be the output of the Concat node
    node1_input = model.graph.input[0]
    node1_input.type.tensor_type.shape.dim[0].dim_value = num_feats
    node1_input.name = input_node_name

    # Insert the new inputs and Concat node at the beginning of the graph
    model.graph.input.extend(new_inputs)
    model.graph.input.pop(0)  # remove the previous input node from inputs
    model.graph.node.insert(0, concat_node)

    return model


def generate_coalition_vectors(num_features):
    coalition_vectors = [[0.0] * num_features]
    for num_avaialable_features in range(1, num_features):
        for x in combinations(range(num_features), num_avaialable_features):
            coalition_vectors.append(
                [1.0 if i in x else 0.0 for i in range(num_features)]
            )
    coalition_vectors.append([1.0] * num_features)
    return torch.tensor(coalition_vectors)


# Create PyTorch Module for shap_part_1 and 2 to be able to convert ONNX
class Shap_part_1(torch.nn.Module):
    def __init__(self, X_train, coalition_vectors) -> None:
        super().__init__()
        self.X = X_train
        self.coalition_vectors = torch.nn.Parameter(coalition_vectors)
        self.create_dataset = build_shap_part_1

    def forward(self, instance):
        return self.create_dataset(
            X_train=self.X, instances=instance, coalition_vector=self.coalition_vectors
        )


class Shap_part_2(torch.nn.Module):
    def __init__(self, X_train, coalition_vectors) -> None:
        super().__init__()
        self.calculate_shap_values = calculate_shap_values
        self.coalition_vectors = torch.nn.Parameter(coalition_vectors)
        self.X = X_train

    def forward(self, instance, preds):
        return self.calculate_shap_values(
            instance, preds, self.X, self.coalition_vectors
        )


def concat_with_shap(
    base_model: onnx.ModelProto,
    X_train: np.ndarray,
    input_names: list[str],
    save_onnx_filename: Optional[str] = None
) -> onnx.ModelProto:
    """ Function that concatenates the shapley values concatenation module to an onnx model.
    For more information read the Method section in this docstring.

    Parameters
    ----------
    base_model: onnx.ModelProto
        onnx model to extend with shapley values, needs to have single input
    X_train: np.ndarray
        matrix of train data values used to fit base_model
    input_names: list[str]
        list of desired input names to be defined for the resulting model with shap values
    save_onnx_filename: Optional[str] = None
        if save_onnx_filename is not None, the concatenated model will be stored in the
        indicated directory

    Returns
    -------
    base_model concatenated with shapley values calculation
    when used for inference, the results are sorted as: prediction, shap_values and expected_score

    Example
    -------
        import onnxruntime as ort
        sess = ort.InferenceSession(save_onnx_filename)
        results_onnx = sess.run(None, inference_data)

        prediction = results_onnx[0][0].tolist()
        shap_values = results_onnx[1][-1].tolist()
        expected_score = float(results_onnx[2])

    Method:
    -------
    In short, Shap creates combinations of the dataset and using these dataset combinations, it
    gets prediction from IsolationForest model. It calculates shap values based on these
    predictions. Because ONNX doesn't support recursive functions and Shap calls the base_model
    for each combinations of dataset, it isn't feasible to merge Shap and base_model graph directly.

    The solution was dividing Shap module into 2 as 'create_dataset' and 'calculate_shap_values' and
    place the IForest between them. So that, It creates whole combinations of dataset as one
    dataset and it doesn't need to call base_model for each of them but instead it calls it ones for
    one aggregated dataset. Then it divides the predictions for each combinations and calculates
    shap values for each combinations.

    Shap_Part_1 (create_dataset) -> base_model -> Shap_part_2 (calculate_shap_values)
    """
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # hide warning messages

    save_path = Path(f'/tmp/{uuid.uuid4()}/models')
    save_path.mkdir(exist_ok=True, parents=True)

    base_model_opset = [i.version for i in base_model.opset_import if i.domain == ""][0]
    sess = ort.InferenceSession(base_model.SerializeToString())
    base_model_input_type = sess.get_inputs()[0].type

    X_train = torch.from_numpy(X_train).type(dtype_mapping_torch[base_model_input_type])

    if X_train.shape[0] > SHAP_SAMPLE_LIMIT:
        X_train = sklearn.utils.resample(
            X_train, n_samples=SHAP_SAMPLE_LIMIT, random_state=0
        )

    coalition_vectors = generate_coalition_vectors(X_train.shape[-1])

    # Convert shap_part_1 to ONNX
    scripted_model = torch.jit.script(
        Shap_part_1(X_train=X_train, coalition_vectors=coalition_vectors)
    )
    part_1_model_path = f"{save_path}/shap_part_1_test.onnx"
    torch.onnx.export(
        model=scripted_model,
        args=(X_train[:2]),
        f=part_1_model_path,
        opset_version=base_model_opset,
        input_names=["instance"],
        output_names=["filled_X", "instance_rep"],
        dynamic_axes={"instance": {0: "batch_size"}},
    )
    # Adding outputs to the shap_part_1 model
    part_1_model = onnx.load(part_1_model_path)
    part_1_model.ir_version = base_model.ir_version
    part_1_model = modify_graph(part_1_model,
                                len(base_model.graph.input),
                                dtype_mapping_onnx[base_model_input_type])
    onnx.save(part_1_model, part_1_model_path)

    # Merging Shap_part_1 + base_model models
    map = [
        (f"output_{i}", input.name) for i, input in enumerate(base_model.graph.input)
    ]
    merged_model = onnx.compose.merge_models(part_1_model, base_model, io_map=map)
    merged_model_path = f"{save_path}/merged_model_test.onnx"
    onnx.save(merged_model, merged_model_path)

    # Get the output of merged model to crete shap_part_2
    sess = ort.InferenceSession(merged_model_path, sess_options=sess_options)
    merged_out = sess.run([], {"instance": X_train[:2].numpy()})
    instances = torch.from_numpy(merged_out[0])
    predictions = torch.from_numpy(merged_out[1])

    # Convert Shap_part_2 to ONNX
    scripted_model = torch.jit.script(
        Shap_part_2(X_train=X_train, coalition_vectors=coalition_vectors)
    )
    part_2_model_path = f"{save_path}/shap_part_2_test.onnx"
    torch.onnx.export(
        model=scripted_model,
        args=(instances, predictions),
        f=part_2_model_path,
        opset_version=base_model_opset,
        input_names=["instance", "preds"],
        output_names=["label", "shap_values", "expected_value"],
        dynamic_axes={
            "instance": {0: "instance_size"},
            "preds": {0: "batch_size"},
        },
    )

    # Merging merdged model with shap_part_2
    part_2_model = onnx.load(part_2_model_path)
    part_2_model.ir_version = base_model.ir_version

    # Adding prefix to avoid overlaping names in both models
    part_2_model = add_prefix(part_2_model, prefix="_", inplace=True)
    map = [
        (base_model.graph.output[0].name, "_preds"),
        ("instance_rep", "_instance"),
    ]

    merged_model2 = onnx.compose.merge_models(merged_model, part_2_model, io_map=map)
    merged_model_path2 = f"{save_path}/merged_model_test2.onnx"
    onnx.save(merged_model2, merged_model_path2)

    # Replace single input by multiple inputs corresponding to every feature
    base_model_with_shap = separate_graph_inputs(merged_model2,
                                                 X_train.shape[-1],
                                                 'instance',
                                                 dtype_mapping_onnx[base_model_input_type],
                                                 input_names)
    if save_onnx_filename:
        onnx.save(base_model_with_shap, save_onnx_filename)

    return base_model_with_shap


if __name__ == "__main__":
    import pandas as pd

    kwargs = {'sep': '\t', 'header': 'infer', 'encoding': 'utf-8', 'index_col': [0]}
    file = "/Users/mariagil/repos/ACS/acs-library/models/training_data_X.csv"
    X_train = pd.read_csv(file, index_col=0).values

    # Convert model to onnx
    model_path = "/Users/mariagil/repos/ACS/acs-library/models/Frac1_pls.onnx"
    onnx_base_model = onnx.load(model_path)
    # onnx_base_mkodel = to_onnx(base_model, initial_types=initial_type, target_opset=12)
    import onnxruntime

    input_names = ['input_33', 'input_47', 'input_6']
    save_onnx_filename = "./models/sklearn_tree_model_with_shap.onnx"
    shap_onxx_model = concat_with_shap(onnx_base_model, X_train, input_names, save_onnx_filename)

    reshaped_test = np.array([X_train[2]])
    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3  # hide warning messages
    sess = onnxruntime.InferenceSession(shap_onxx_model.SerializeToString(),
                                        sess_options=sess_options)

    inference_data = {
        k.name: [[v]] for k, v in zip(sess.get_inputs(), reshaped_test[0])
    }
    results_onnx = sess.run(None, inference_data)
    result_label_onnx = results_onnx[0][0].tolist()
    result_scores_onnx = results_onnx[1][-1].tolist()
    result_expected_score_onnx = float(results_onnx[2])
    print(result_expected_score_onnx)