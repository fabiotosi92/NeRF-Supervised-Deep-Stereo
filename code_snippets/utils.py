import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision

def SSIM(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1) 

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2) 

    base_grid = torch.stack([x_base, y_base], 1)
    return base_grid

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def loss_smooth(disp, im1_scaled):
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])

def disp_warp(x, disp, r2l=False, pad='border', mode='bilinear', device='cuda'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)
    v_grid = norm_grid(base_grid + torch.cat((offset*disp,torch.zeros_like(disp)),1)) 
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    mask = torch.autograd.Variable(torch.ones(x_recons.size())).to(device)
    mask = nn.functional.grid_sample(mask, v_grid)
    return x_recons, mask

def photometric_loss(im1_scaled, im1_recons):
    loss = []
    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []