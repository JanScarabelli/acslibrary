import os
import torch
import torch.nn as nn
import onnx


# Creating Pytoch Module based on numpy version
class Resampler_nn(nn.Module):
    def __init__(self, n_chunks: int = 12):
        super(Resampler_nn, self).__init__()
        self.n_chunks = n_chunks

    def forward(self, x_in):
        x_in_dim = x_in.shape[0]
        x_in_mod = torch.remainder(torch.tensor(x_in_dim), torch.tensor(self.n_chunks))

        x_in_dim_a = x_in_dim - x_in_mod
        x_in_v = x_in[:x_in_dim_a]
        chunk_size = torch.divide(x_in_dim_a, self.n_chunks).type(torch.int64)
        chunk_indexes = torch.arange(0, x_in_dim_a+chunk_size, chunk_size)

        x_res = torch.empty(self.n_chunks, dtype=torch.float64)
        for chunk in range(self.n_chunks):
            chunk_idx_st = chunk_indexes[chunk]
            chunk_idx_end = chunk_indexes[chunk+1]
            x_in_v_res = x_in_v[chunk_idx_st:chunk_idx_end]
            x_res[chunk] = torch.mean(x_in_v_res)
        return x_res


class Resampler_nn_with_derivative(nn.Module):
    def __init__(self, n_chunks: int = 12):
        super(Resampler_nn_with_derivative, self).__init__()
        self.n_chunks = n_chunks

    def forward(self, x_in):
        x_in_dim = x_in.shape[1]
        x_in_mod = torch.remainder(torch.tensor(x_in_dim), torch.tensor(self.n_chunks))

        x_in_dim_a = x_in_dim - x_in_mod
        x_in_v = x_in[:, :x_in_dim_a]
        chunk_size = torch.divide(x_in_dim_a, self.n_chunks).type(torch.int64)
        chunk_indexes = torch.arange(0, x_in_dim_a+chunk_size, chunk_size)

        x_res = torch.empty(x_in.shape[0], self.n_chunks, dtype=torch.float64)
        df_res = torch.empty(x_in.shape[0], self.n_chunks-1, dtype=torch.float64)
        for chunk in range(self.n_chunks):
            chunk_idx_st = chunk_indexes[chunk]
            chunk_idx_end = chunk_indexes[chunk+1]
            x_in_v_res = x_in_v[:, chunk_idx_st:chunk_idx_end]
            x_res[:, chunk] = torch.mean(x_in_v_res, dim=1)
            if chunk > 0:
                df_res[:, chunk-1] = (x_res[:, chunk] - x_res[:, chunk-1]) / x_res[:, chunk-1]
        return x_res, df_res


def modify_nodes(model, derivatives=False, verbose=False):
    # Load original model and remove cast node for float31 -> int64
    graph = model.graph

    if verbose:
        # Observe the indexes of the nodes that need changes
        for i, j in enumerate(graph.node):
            if j.name == '/Cast_1':
                print('Cast_index: ', i)
            if j.name == '/Div':
                print('Div_index: ', i)
            if j.name == '/Add':
                print('Add: ', i)
            if j.name == "/Constant_5":
                print("Constant 5: ", i)
        # Results:
            # Resampler with derivatives model
                # Constant 5:  10
                # Add:  13
                # Cast_index:  20
                # Div_index:  22
            # Resampler model
                # Constant 5:  9
                # Add:  12
                # Cast_index:  19
                # Div_index:  21

    # /Div node is the output node from /Cast_1 node, which we want to remove
    # We therefore change its input nodes (/Cast_1, /Constant_10)
    # to the intended ones (/Add output, /Constant_5)
    if derivatives:
        graph.node[22].input[0] = graph.node[13].output[0]
        graph.node[22].input[1] = graph.node[10].output[0]
    else:
        graph.node[21].input[0] = graph.node[12].output[0]
        graph.node[21].input[1] = graph.node[9].output[0]

    # Create new graph without unused nodes
    inputs = graph.input
    outputs = graph.output
    new_nodes = graph.node

    # remove cast node (20 if derivative else 19) and its input node (21 if derivative else 20)
    # because it is not needed for the rest of graph.
    if derivatives:
        new_nodes.pop(21)
        new_nodes.pop(20)
    else:
        new_nodes.pop(20)
        new_nodes.pop(19)

    # Creating new graph with new nodes
    graph_name = "test_model_resampling"
    graph_ = onnx.helper.make_graph(new_nodes, graph_name, inputs, outputs)
    model_ = onnx.helper.make_model(graph_, opset_imports=[onnx.helper.make_opsetid('', 16)])
    model_.ir_version = model.ir_version
    onnx.checker.check_model(model_)

    return model_


def resampler_to_onnx(input_data, n_chunks, save_onnx_filename=None, derivative=False):
    if derivative:
        model = Resampler_nn_with_derivative(n_chunks=n_chunks)
    else:
        model = Resampler_nn(n_chunks=n_chunks)

    # Generate TorchScript module which is required to export model
    scripted_model = torch.jit.script(model)

    tmp_save_file_resampler = "tmp_resampler.onnx"
    # Exporting Pytorch module to ONNX model
    torch.onnx.export(
        scripted_model,
        input_data,
        tmp_save_file_resampler,
        opset_version=16,
        input_names=["input"],
        output_names=["resampler_output"],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'batch_size'},
            'resampler_output': {0: 'chunk_size'}
        })

    # Load original model and remove cast node for float64 -> float32
    onnx_model = onnx.load(tmp_save_file_resampler)
    modified_onnx_model = modify_nodes(onnx_model, derivatives=derivative)
    os.remove(tmp_save_file_resampler)

    if save_onnx_filename:
        onnx.save(modified_onnx_model, save_onnx_filename)
    else:
        return modified_onnx_model
