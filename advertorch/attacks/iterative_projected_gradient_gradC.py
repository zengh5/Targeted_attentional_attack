# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp

from .base import Attack
from .base import LabelMixin
# from .utils import rand_init_delta
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from torchvex import GradCAM


def perturb_iterative_gradC(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                            delta_init=None, minimize=False, ord=np.inf,
                            clip_min=0.0, clip_max=1.0,
                            l1_sparsity=None, cam=torch.from_numpy(np.array([1])),
                            combine=False):
    """
    Iteratively maximize the loss over the input. Targeted attack guided by Grad-CAM.
    """
    grad_cam_generator = GradCAM(predict, target_layer=predict[1].layer4)
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)
        delta.requires_grad_()
    delta.requires_grad_()
    cam = cam.cuda()
    for ii in range(nb_iter):
        # delta -> delta_cam
        delta_cam = delta * cam
        x_new = xvar + delta_cam
        outputs = predict(x_new)
        print(outputs[0][2].data)
        print(outputs[0][7].data)
        loss = loss_fn(outputs, yvar)

        if minimize:
            loss = -loss
        loss.backward()

        if ii == nb_iter / 2:
            if combine:
                # get Grad_cam_t from the intermediate image
                grad_cam = grad_cam_generator(x_new, yvar)
                grad_cam = grad_cam.squeeze().cpu()
                grad_cam = grad_cam - grad_cam.min()
                grad_cam = grad_cam / grad_cam.max()
                # plt.imshow(grad_cam > (1. / 3), cmap='gray')
                # plt.show()

                newcam = grad_cam > (1. / 3)
                newcam = newcam.cuda()
                cam = (newcam | cam)

        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            # delta -> delta_cam
            delta_cam = delta * cam
            x_adv = xvar.data + delta_cam.data
            delta.data = clamp(x_adv, clip_min, clip_max) - xvar.data
        else:
            error = "Only ord = inf has been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta_cam, clip_min, clip_max)
    perturb_percentage = cam.sum() / (cam.shape[0] * cam.shape[1])
    # print(loss)
    return x_adv, perturb_percentage


class PGDAttack_gradC_t(Attack, LabelMixin):
    """
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.
        """
        super(PGDAttack_gradC_t, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None, cam=torch.from_numpy(np.array([1])), combine=False):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param cam:
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :param combine: whether use the CAM from the intermediate image
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)

        rval, perturb_percentage = perturb_iterative_gradC(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity, cam=cam, combine=combine,
        )
        return rval.data, perturb_percentage.cpu()


class LinfPGDAttack_gradC_t(PGDAttack_gradC_t):
    """
    PGD Attack with order=Linf
    """
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack_gradC_t, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted, ord=ord)
