#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn


default_config = {
    'descriptor_dim': 256,
    'nms_radius': 4,
}


class MaxPool(nn.Module):
    def __init__(self, nms_radius: int):
        super(MaxPool, self).__init__()
        self.block = nn.MaxPool2d(kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        return torch.squeeze(self.block(x), dim=1)


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    max_pool = MaxPool(nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class SuperPoint(nn.Module):

    # default_config = {
    #     'descriptor_dim': 256,
    #     'nms_radius': 4,
    #     'keypoint_threshold': 0.005,
    #     'max_keypoints': -1,
    #     'remove_borders': 4,
    # }

    def __init__(self):
        super().__init__()
        # self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, default_config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, default_config['nms_radius'])

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        return scores, descriptors


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_onnx():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="weights/superpoint_v1.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="weights", help="onnx model file output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    # load model
    superpoint_model = SuperPoint()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('total number ff params: ', pytorch_total_params)

    # initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    superpoint_model.load_state_dict(torch.load(weight_file, map_location=map_location))
    superpoint_model.eval()

    # create input to the model for onnx trace
    input = torch.randn(1, 1, 640, 640)

    torch_out = superpoint_model(input)
    onnx_filename = os.path.join(output_dir, weight_file.split("/")[-1].split(".")[0] + ".onnx")

    # export the model
    torch.onnx.export(superpoint_model,  # model being run
                    input,  # model input (or a tuple for multiple inputs)
                    onnx_filename,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=11,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model input names
                    output_names=['scores', 'descriptors'],  # the model output names
                    dynamic_axes={'input': {2: 'input_height', 3: "input_width"},
                                'scores': {1: 'scores_height', 2: "scores_width"},
                                'descriptors': {2: 'descriptors_height', 3: "descriptors_width"},  
                                }  # dynamic input names
                    )

    # check onnx conversion
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    onnxruntime_session = onnxruntime.InferenceSession(onnx_filename)

    # compute ONNX Runtime output prediction
    onnxruntime_inputs = {onnxruntime_session.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outs = onnxruntime_session.run(None, onnxruntime_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out[0]), onnxruntime_outs[0], rtol=1e-03,
                               atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), onnxruntime_outs[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good.")

    onnx_sim(onnx_filename)


def onnx_sim(output_path:str):
    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(output_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)


if __name__ == '__main__':
    export_onnx()


