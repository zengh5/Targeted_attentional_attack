from numbers import Number

import torch

from .hook import FeatureFetcher
from .base import _CAMBase
from torchvex.utils import index_to_onehot


class GradCAM(_CAMBase):
    def __init__(self, model, target_layer, create_graph=False, interpolate=True):
        if isinstance(target_layer, torch.nn.Module):
            target_layer = [target_layer]
        super().__init__(model, target_layer, create_graph, interpolate)

    @torch.enable_grad()
    def create_cam(self, inputs, target):
        self.model.zero_grad()
        inputs.requires_grad_(True)

        with FeatureFetcher(self.target_layer) as fetcher:
            output = self.model(inputs)

        onehot = index_to_onehot(target, output.shape[-1])
        loss = (output * onehot).sum()

        grad_cams = []
        for feature in fetcher.feature:
            grad = torch.autograd.grad(
                loss, feature, create_graph=self.create_graph, retain_graph=True
            )[0]
            weight = torch.nn.functional.adaptive_avg_pool2d(grad, 1)   # (1)
            grad_cam = feature.mul(weight)
            grad_cam = grad_cam.sum(dim=1, keepdim=True)
            if (grad_cam > 0).sum() == 0:
                print("All-negative attentional map")
            grad_cam = torch.nn.functional.relu(grad_cam)                # (2)
            grad_cams.append(grad_cam)
        return grad_cams
