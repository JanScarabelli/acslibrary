import os
from typing import Optional

import torch
import onnx
from tslearn.clustering import KShape as tslearn_KShape


@torch.jit.script
def dft_mult(r1, i1, r2, i2):
    """ Multiplication of complex numbers

    Parameters:
    -----------
    r1 = real part of the first complex number
    i1 = imaginary part of the first complex number
    r2 = real part of the second complex number
    i2 = imaginary part of the second complex number

    Math
    -----
    (a, bi) * (c, di) = (ac - bd), i(ad + bc)
    """
    output_real = r1*r2 - i1*i2
    output_imag = r1*i2 + r2*i1
    return output_real, output_imag


@torch.jit.script
def matmul(r1, i1, r2, i2):
    """ Matrix multiplication

    Parameters:
    -----------
    r1 = real part of the first complex number

    i1 = imaginary part of the first complex number

    r2 = real part of the second complex number

    i2 = imaginary part of the second complex number
    """
    result_real = torch.zeros((r1.shape[0], r2.shape[1]), dtype=torch.float64)
    result_imag = torch.zeros((r1.shape[0], r2.shape[1]), dtype=torch.float64)
    for i in range(r1.shape[0]):
        for j in range(r2.shape[1]):
            x, y = dft_mult(r1[i], i1[i], r2[:, j], i2[:, j])
            result_real[i][j] = (torch.sum(x))
            result_imag[i][j] = (torch.sum(y))
    return result_real, result_imag


@torch.jit.script
def dft(x, dft_size: int = -1, y=torch.tensor([0.], dtype=torch.float64)):
    """ Fourier Transform of 2-dims complex numbers in axis = 0

    Parameters:
    -----------
    x = real part of the complext number

    y = imaginary part of the complex number

    dft_size = signal length
    """
    if dft_size > x.shape[0]:
        x = torch.cat((x, torch.zeros(dft_size-x.shape[0], x.shape[1], dtype=torch.float64)))
    elif dft_size == -1:
        dft_size = x.shape[0]
    x = x[:dft_size]
    N = x.shape[0]
    n = torch.arange(N, dtype=torch.float64)
    k = n.reshape((N, 1))
    exp = (-2 * torch.pi * k * n / N)
    cons = torch.tensor(90 * torch.pi / 180, dtype=torch.float64)
    cos_value = cons + exp
    R, I = torch.sin(cos_value), torch.sin(exp)
    if y.shape[0] != x.shape[0]:
        y = torch.zeros((x.shape), dtype=torch.float64)
    return matmul(R, I, x, y)


@torch.jit.script
def idft(x, y=None):
    """ Inverse Fourier Transform of 2-dims complex numbers

    Parameters:
    -----------
    x = real part of the complex number

    y = imaginary part of the complex number

    Math:
    -----
    idft(x) = conj(dft(conj(x))) / len(x)
    """
    dft_real, dft_imag = dft(x, y=-y)
    output_real = torch.divide(dft_real, x.shape[0])
    output_imag = -torch.divide(dft_imag, x.shape[0])
    return output_real, output_imag


@torch.jit.script
def cc(s1, s2, fft_sz):
    """ Cross-Correlation

    Parameters:
    -----------
    s1 : time series data

    s2 : time series data

    fft_sz : signal length
    """
    r1, i1 = dft(s1, fft_sz)
    r2, i2 = dft(s2, fft_sz)
    mult_r, mult_i = dft_mult(r1, i1, r2, -i2)
    cc = idft(mult_r, y=mult_i)
    return cc[0]


@torch.jit.script
def normalized_cc(s1, s2, norm1=torch.Tensor(), norm2=-torch.Tensor()):
    """ Normalized Cross-Correlation

    Parameters:
    -----------
    s1 : time series data

    s2 : time series data (cluster center)

    norm1 : matrix norm of s1

    norm2 : matrix norm of s2
    """
    sz = s1.shape[0]
    n_bits = (torch.tensor(2 * sz - 1, dtype=torch.float64).log2() + 1).type(torch.int64)
    fft_sz = 2**n_bits

    denom = (norm1 * norm2).type(torch.float64)
    if denom < torch.tensor(1e-9, dtype=torch.float64):  # To avoid NaNs
        denom = torch.tensor(torch.inf, dtype=torch.float64)

    cc_ = cc(s1, s2, fft_sz)
    cc_ = torch.cat((cc_[-(sz-1):], cc_[:sz]))
    output = cc_.sum(dim=-1) / denom
    return output


class KShape(torch.nn.Module):
    def __init__(self, cluster_centers):
        super(KShape, self).__init__()
        self.normalized_cc = normalized_cc
        self.cluster_centers = cluster_centers

    def forward(self, x):
        dists = torch.empty((x.shape[0], self.cluster_centers.shape[0]), dtype=torch.float64)
        norms1 = torch.linalg.norm(x, dim=(1, 2))
        # norms2 was defined in this way to make sure #cluster_centers.shape[0] nodes created in the
        # onnx graph. For some cases ,because the values are so close, onnx creates 1 node instead
        # of 2. The correct values of the nodes will be assigned after executing modify_graph().
        norms2 = torch.arange(self.cluster_centers.shape[0], dtype=torch.float64)

        for i in range(x.shape[0]):
            for j in range(self.cluster_centers.shape[0]):
                dists[i, j] = self.normalized_cc(x[i], self.cluster_centers[j], norm1=norms1[i],
                                                 norm2=norms2[j]).max()
        labels = torch.argmin(1 - dists, dim=1)
        return labels, dists


def kshape_to_onnx(
        train_data: torch.tensor,
        num_clusters: int,
        save_onnx_filename: Optional[str] = None,
        seed: Optional[int] = None,
        ) -> Optional[onnx.GraphProto]:
    """Function that trains kshape algorithm and saves it in onnx format.

    This function fits a tslearn k-shape algorithm with `train_data` and selected number of
    clusters `num_clusters`, then it fits the internal implementation of KShape in pytorch
    and exports it in ONNX format. The resulting model nodes are modified in order to allow
    float64 conversions. If `save_onnx_filename` is passed, the modified ONNX model is saved to
    the corresponding path; the function returns the modified model otherwise.

    Parameters
    ----------
    train_data : torch.tensor
        Train data to fit k-shape algorithm
    num_clusters : int
        Number of clusters to train the kshape algorithm
    save_onnx_filename : Optional[str]
        Path where the kshape onnx file will be stored
    seed : Optional[int64]
        Seed for development purposes, to initialize the centroids in the same p

    Returns
    -------
    Optional[ONNX model]
        If no save_onnx_filename is passed, the return will be a ONNX model. Otherwise, the model
        will be saved to the passed path and no return is necessary.
    """
    # Fit k-shape model to train data with original tslearn algorithm
    tslearn_model = tslearn_KShape(n_clusters=num_clusters, verbose=True, random_state=seed)
    tslearn_model.fit(train_data)
    cluster_centers = torch.from_numpy(tslearn_model.cluster_centers_)

    # Create KShape ONNX model
    tmp_kshape_model = "tmp_KShape.onnx"
    scripted_model = torch.jit.script(KShape(cluster_centers))
    torch.onnx.export(scripted_model,
                      (torch.randn(10, 10, 1).type(torch.float64)),
                      tmp_kshape_model,
                      input_names=["input_1"],
                      opset_version=16,
                      output_names=["labels_", "distances_"],
                      dynamic_axes={
                        "input_1": {0: "batch_size", 1: "time_series_size"},
                        "distances_": {0: "batch_size"}
                        })

    # Load original model and remove cast node for float64 -> float32
    onnx_model = onnx.load(tmp_kshape_model)
    modified_onnx_model = modify_nodes(onnx_model, cluster_centers)

    # Remove temporal onnx model without float64 conversion
    os.remove(tmp_kshape_model)

    if save_onnx_filename is None:
        return modified_onnx_model
    else:
        # Save kshape onnx model
        onnx.save(modified_onnx_model, save_onnx_filename)


def modify_nodes(model, cluster_centers):
    """ This function modifies the nodes from the original kshape model to remove float32 cast nodes
    """
    graph = model.graph

    # Create new nodes to replace with olders
    node_radian, node_norms = create_nodes(cluster_centers)
    # Get the indexes of the nodes should be changed
    cast_idxs, mult_idxs, add_idxs = get_indexes(graph, cluster_centers.shape[0])

    # Add new nodes to the graph
    for i, j in enumerate(node_norms):
        graph.node[20].attribute[0].g.node.insert(i, j)
    graph.node[20].attribute[0].g.node.insert(0, node_radian)
    extra_nodes = len(node_norms) + 1

    # Replace the olders with the news
    idx = 1
    for i, j in zip(mult_idxs, cast_idxs):
        graph.node[20].attribute[0].g.node[i+extra_nodes].input[0] = graph.node[20].attribute[0].g.node[j+extra_nodes].input[0]
        graph.node[20].attribute[0].g.node[i+extra_nodes].input[1] = graph.node[20].attribute[0].g.node[idx].output[0]
        idx += 1
    for i in add_idxs:
        graph.node[20].attribute[0].g.node[i+extra_nodes].input[0] = graph.node[20].attribute[0].g.node[0].output[0]

    # Creating new graph with new nodes
    graph_name = "test_model"
    graph_ = onnx.helper.make_graph(graph.node, graph_name, graph.input, graph.output)
    onnx_model = onnx.helper.make_model(graph_, opset_imports=[onnx.helper.make_opsetid('', 16)])
    onnx_model.ir_version = model.ir_version
    onnx.checker.check_model(onnx_model)
    return onnx_model


def get_indexes(graph, n):
    """Returns the indexes of the nodes that should be changed

    This function iterates the graph to find which nodes need to be changed.
    The nodes to be changed include:
        - both inputs of Mult nodes should be changed with new nodes
        - input of spicific Add nodes to be changed with its float64 values
        - float32 Cast nodes to be removed from the graph

    There is a pattern in Cast names and Mult names that allows us to delimit which nodes we want
    to modify: for each number of cluster n, next Cast and Mul node's names will be defined 3+n*18
    and 2+n*x39 consecutively.

    Example :
        for n = 2 : '/Cast_3','/Cast_21','/Mul_2','/Mul_41'

    """
    mults = []  # Both inputs of mult nodes should be changed with new nodes
    adds = []   # Input of spicific Add nodes should be changed with its float64 values
    casts = []  # All float32 Cast nodes should be removed from the graph

    cast_name_list = ['/Cast_'+str(3+i*18) for i in range(n)]
    mul_name_list = ['/Mul_'+str(2+i*39) for i in range(n)]

    # Adding indexes to the list it belongs
    for j, i in enumerate(graph.node[20].attribute[0].g.node):
        if i.name in cast_name_list:
            casts.append(j)
        if i.name in mul_name_list:
            mults.append(j)
        if i.name in ['/Add_1', '/Add_5', '/Add_10', '/Add_13', '/Add_17', '/Add_22', '/Add_25',
                      '/Add_29', '/Add_34']:
            adds.append(j)
    return casts, mults, adds


def create_nodes(cluster_centers):
    """Create new nodes with intended values
    """
    node_radian = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["radian_values"],
        value=onnx.helper.make_tensor(
            name="radian_const_tensor",
            data_type=onnx.TensorProto.DOUBLE,
            dims=[1],
            vals=torch.tensor([1.5707963267948966], dtype=torch.float64),
        ),
    )

    norms = torch.linalg.norm(cluster_centers, dim=(1, 2))
    node_norms = []
    for i, j in enumerate(norms):
        output_name = "norm_values_"+str(i)
        value_name = "norm_const_tensor_"+str(i)
        node_norms.append(onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[output_name],
            value=onnx.helper.make_tensor(
                name=value_name,
                data_type=onnx.TensorProto.DOUBLE,
                dims=[1],
                vals=torch.tensor([j], dtype=torch.float64),
            ),)
        )
    return node_radian, node_norms
