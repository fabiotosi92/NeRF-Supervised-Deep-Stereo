import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append('dataloaders')
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from models.psmnet import PSMNet
from models.raft_stereo import RAFTStereo
import datasets

def load_pretrained_model(args):
    # Load the pretrained model based on the selected model type
    print('Load pretrained model')
    model = None
    if args.model == 'raft-stereo':
        model = RAFTStereo(args)
    elif args.model == 'psmnet':
        model = PSMNet(args.maxdisp)
    else:
        print('Invalid model selected.')
        exit()

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        pretrain_dict = torch.load(args.loadmodel, torch.device('cuda:0'))
        model.load_state_dict(pretrain_dict)
    else:
        print('A pretrained model is required!')

    return model

def eval(pred_disp, gt_disp, valid):
    # Calculate evaluation metrics for the predicted disparity map
    gt_disp = gt_disp.squeeze()
    gt_disp[gt_disp == np.inf] = 0
    mask = gt_disp > 0
    abs_diff = np.abs(gt_disp[mask] - pred_disp[mask])
    EPE = abs_diff.mean()

    d1 = (abs_diff > 1).sum() / mask.sum()
    d2 = (abs_diff > 2).sum() / mask.sum()
    d3 = (abs_diff > 3).sum() / mask.sum()

    return {'EPE': EPE, 'bad 1.0': d1, 'bad 2.0': d2, 'bad 3.0': d3}

@torch.no_grad()
def run(model, data, args):
    # Run the model on input data and calculate the disparity map
    model.eval()

    if args.cuda:
        data['im2'], data['im3'] = data['im2'].cuda(), data['im3'].cuda()

    ht, wt = data['im2'].shape[-2], data['im2'].shape[-1]

    pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
    pad_wd = (((wt // 32) + 1) * 32 - wt) % 32

    _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    if args.model == 'raft-stereo':
        data['im2'] = F.pad(data['im2'], _pad, mode='replicate')
        data['im3'] = F.pad(data['im3'], _pad, mode='replicate')
    else:
        data['im2'] = F.pad(data['im2'], _pad)
        data['im3'] = F.pad(data['im3'], _pad)

    pred_disps = model(data['im2'], data['im3'])

    if args.model == 'psmnet':
        pred_disp = pred_disps[0]
    elif args.model == 'raft-stereo':
        pred_disp = pred_disps[-1].squeeze()

    ht, wd = pred_disp.shape[-2:]
    c = [_pad[2], ht - _pad[3], _pad[0], wd - _pad[1]]
    pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

    result = {}

    if 'gt' in data:
        scaling = data['gt'].shape[3] / pred_disp.shape[-1]
        pred_disp = F.interpolate(pred_disp.unsqueeze(0).unsqueeze(0),
                                  size=(data['gt'].shape[2], data['gt'].shape[3]),
                                  mode='nearest').squeeze() * scaling
        result = eval(pred_disp.cpu().numpy(), data['gt'].numpy(), data['validgt'].numpy())

    result['disp'] = pred_disp

    return result

def process_batch(model, data, args, batch_idx, outdir=None):
    # Process a batch of data using the model and save the output if specified
    result = run(model, data, args)

    if outdir is not None:
        disp_path = os.path.join(outdir, f"{batch_idx}.jpg")
        npy_path = os.path.join(outdir, f"{batch_idx}.npy")
        plt.imsave(disp_path, result['disp'].detach().cpu().numpy(), cmap="magma")
        np.save(npy_path, result['disp'].detach().cpu().numpy())

    return result

def main():

    parser = argparse.ArgumentParser(description='PSMNet')
    parser.add_argument('--maxdisp', type=int, default=256, help='maximum disparity')
    parser.add_argument('--datapath', default='dataset/', help='datapath')
    parser.add_argument('--dataset', choices=['middlebury', 'kitti', '3nerf'], default='middlebury', help='dataset type')
    parser.add_argument('--version', default='training', help='version')
    parser.add_argument('--model', default='raft-stereo', choices=['psmnet', 'raft-stereo'], help='select model')
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--loadmodel', default='./weights/raftstereo-NS.tar', help='load model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    parser.add_argument('--occ', action='store_true', default=False, help='occluded regions are included in the eval process')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.test = True
    args.batch_size = 1

    model = load_pretrained_model(args)
    demo_loader = datasets.fetch_dataloader(args)

    if args.outdir is not None and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    results = {}
    progress_bar = tqdm.tqdm(total=len(demo_loader))

    for batch_idx, data in enumerate(demo_loader):
        result = process_batch(model, data, args, batch_idx, args.outdir)

        for key in result:
            if key != 'disp' and key != 'errormap':
                results.setdefault(key, []).append(result[key])

        progress_bar.update(1)

    progress_bar.close()

    print("Average results:")
    for key in results:
        mean_value = np.array(results[key]).mean()
        if 'bad' not in key:
            print(f"{key}: {mean_value:.4f}")
        else:
            print(f"{key}: {mean_value * 100:.2f}%")

if __name__ == '__main__':
    main()

