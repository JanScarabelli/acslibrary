import torch


class torchDTW(torch.nn.Module):
    def __init__(self):
        super(torchDTW, self).__init__()

    def forward(self, x, y):
        dtw = torch.empty(x.size(0) + 1, y.size(0) + 1).fill_(99999.0)
        dtw[0, 0] = 0.0
        for i in range(0, x.size(0)):
            for j in range(0, y.size(0)):
                cost = x[i] - y[j]
                cost = cost * cost
                dtw[i+1, j+1] = cost + min(
                                            dtw[i, j+1].item(),  # insertion
                                            dtw[i+1, j].item(),  # deletion
                                            dtw[i, j].item(),    # match
                                )
        return torch.sqrt(dtw)


def dtw_to_onnx(save_onnx_filename):
    """Function that exports Dynamic Time Warping (DTW) metric to onnx format.

    This function creates an instance of the custom Pytorch implementation of DTW and
      exports the model to onnx in `save_onnx_filename`.

    Parameters
    ----------
    save_onnx_filename : Path [str]
        Directory where the exported ONNX model will be stored.
    """

    # Instantiation and scripting
    model_scripted = torch.jit.script(torchDTW())
    dummy_input_one = torch.Tensor([1, 2, 3, 10, 8, 5, 4, 1, 2])
    dummy_input_two = torch.Tensor([4, 5, 6, 2, 2])

    # Export to onnx:
    torch.onnx.export(
        model_scripted,
        (dummy_input_one, dummy_input_two),
        save_onnx_filename,
        verbose=True,
        input_names=['input_data_1', 'input_data_2'],
        output_names=['dtw'],
        opset_version=14,
        do_constant_folding=False,
        dynamic_axes={
            # list value: automatic names
            "input_data_1": [0],
            "input_data_2": [0],
            "dtw": [0],
        },)
