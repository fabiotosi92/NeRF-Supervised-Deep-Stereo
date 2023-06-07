import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *

def trinocular_loss(disp, im1, im2, im3, uncertainty):
    im2_recons_from_1, mask_12 = disp_warp(im1, disp, r2l=True)
    im2_recons_from_3, mask_23 = disp_warp(im3, disp, r2l=False)

    photometric_loss_12 = photometric_loss(im2, mask_12 * im2_recons_from_1)
    photometric_loss_23 = photometric_loss(im2, mask_23 * im2_recons_from_3)
    loss_warp, _ = torch.min(torch.cat((photometric_loss_12, photometric_loss_23), dim=1), dim=1)

    photometric_loss_1 = photometric_loss(im2, im1)
    photometric_loss_3 = photometric_loss(im2, im3)
    loss_2, _ = torch.min(torch.cat((photometric_loss_1, photometric_loss_3), dim=1), dim=1)

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)[automask]

    return loss.mean()

def binocular_loss(disp, im1, im2, uncertainty):
    im1_recons, _ = disp_warp(im2, disp, r2l=False)

    loss_warp = photometric_loss(im1, im1_recons).squeeze()
    loss_2 = photometric_loss(im2, im1).squeeze()

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)[automask]

    return loss.mean()

def image_loss(disp, im1, im2, im3, uncertainty, trinocular=True):
    if trinocular:
        return trinocular_loss(disp, im1, im2, im3, uncertainty)
    else:
        return binocular_loss(disp, im2, im3, uncertainty)



## For training

# target_disp = data["label"]
# conf = data["conf"] * (target_disp > 0).float()

# n_predictions = len(pred_disps)
# loss_gamma = 0.9
# target_disp = target_disp.unsqueeze(1)

# disp_loss = 0.0
# photometric_loss = 0.0

# #  PSMNet & CFNet
# for i in range(len(pred_disps)):
#     disp_loss += (1 / 2 ** i) * (torch.abs(pred_disps[i] - target_disp) * conf * (target_disp > 0).float()).mean()
#     photometric_loss += (1 / 2 ** i) * image_loss(pred_disps[i].unsqueeze(1), data['im0'], data['im1'], data['im2'], 1 - conf, args.trinocular_loss)

# loss = args.alpha_disp_loss * disp_loss + args.alpha_photometric * photometric_loss

# # RAFT-Stereo
# for i in range(n_predictions):
#     adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
#     i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
    
#     disp_diff = torch.abs(pred_disps[i] - target_disp)
#     disp_loss += i_weight * (disp_diff * conf * (target_disp > 0).float()).mean()
    
#     if args.alpha_photometric != 0.:
#         photometric_loss += i_weight * image_loss(pred_disps[i], data['im0'], data['im1'], data['im2'], 1 - conf, args.trinocular_loss)
#     else:
#         photometric_loss = torch.zeros_like(disp_loss)

# loss = args.alpha_disp_loss * disp_loss + args.alpha_photometric * photometric_loss




