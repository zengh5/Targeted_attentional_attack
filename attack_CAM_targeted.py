import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imread, imsave
from utils import bchw2bhwc, _show_images
from advertorch.attacks import LinfPGDAttack_cam_t, LinfPGDAttack_gradC_t
# Attack
from torchvision import datasets, transforms
import torchvision.models as models

torch.manual_seed(0)
device = torch.device("cuda")

# 1 Datasets
# Data transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      ])
valdir = 'images'
test_set = datasets.ImageFolder(valdir, test_transforms)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

# 2 Models
print("=> creating model '{}'".format('resnet18'))
model = models.__dict__['resnet18'](num_classes=20)
saved = torch.load('trained_models/best_res18_c20_847.pth')  # oracle
model.load_state_dict(saved)
# the model to attack include a pre-processing layer
model = nn.Sequential(
    normalize,
    model
)
model.eval()
model = model.to(device)

# 3 define the attack
epsinons = [2. / 255, 4. / 255, 6. / 255, 8./255,
            10./255, 12./ 255, 14./ 255, 16./ 255]

adversary = LinfPGDAttack_cam_t(
    predict=model, eps=4. / 255, eps_iter=4. / 255 * 4 / 40, nb_iter=80,
    rand_init=False, targeted=True)

# 4 attack
for epsinoni in range(0, 1):
    adversary.eps = epsinons[epsinoni]
    adversary.eps_iter = adversary.eps * 4 / 40

    for _, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()
        # define the target label as (original label+5)
        targetlabel = (target + 5) % 20

        #  Proposed method: combine the CAM of ori and CAM of target (from an intermediate image)
        # Attentional map
        cam_name = 'CAM/CAM_o.png'          # CAM of the original label
        cam_matrix_o = imread(cam_name)
        cam_matrix = cam_matrix_o.astype(np.float32) * 1.0
        cam = torch.from_numpy(cam_matrix > (255. / 3))
        # plt.imshow(cam_matrix > (255. / 3), cmap='gray')
        # plt.show()

        advimg, perturb_percentage1 = adversary.perturb(input, targetlabel, cam=cam, combine=True)
        # 8-bit quantization
        advuint8 = (advimg * 255).round()
        attacklogitQ = model(advuint8 / 255.)
        _, attacklabelQ = attacklogitQ.data.topk(1, dim=1)
        _show_images(input, advimg)
        ad_np = bchw2bhwc(advuint8[0].cpu().numpy())
        print('Baseline scheme, modified area:')
        print(perturb_percentage1.cpu().data)
        if attacklabelQ[0] == targetlabel[0]:
            print('Succeed')
        else:
            print('Fail')

        # Baseline, only use the CAM_t from the benign image
        # Attentional map
        cam_name = 'CAM/CAM_t.png'  # CAM of target label
        cam_matrix_t = imread(cam_name)
        cam_matrix = cam_matrix_t.astype(np.float32) * 1.0

        threshold = np.percentile(cam_matrix, (1 - perturb_percentage1.cpu()) * 100)
        cam = torch.from_numpy(cam_matrix > threshold)
        # plt.imshow(cam_matrix > (threshold), cmap='gray')
        # plt.show()

        advimg, perturb_percentage2 = adversary.perturb(input, targetlabel, cam=cam, combine=False)

        # 8-bit quantization
        advuint8 = (advimg * 255).round()
        attacklogitQ2 = model(advuint8 / 255.)
        _, attacklabelQ = attacklogitQ2.data.topk(1, dim=1)
        _show_images(input, advimg)
        ad_np = bchw2bhwc(advuint8[0].cpu().numpy())
        print('Baseline scheme, modified area:')
        print(perturb_percentage2.cpu().data)
        if attacklabelQ[0] == targetlabel[0]:
            print('Succeed')
        else:
            print('Fail')

done = 1
