
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import cv2
from utils import *
#from augmentor import DisparityAugmentor

class Middlebury(data.Dataset):
    def __init__(self, datapath, version="training", occ=False, test=True):
        self.is_test = test
        self.disp_list = []
        self.image_list = []

        self.version = version
        self.occ = occ

        self.gt_name = "disp0" if "2021" in self.version else "disp0GT"
        self.mask_name = "mask0nocc"

        image_list = sorted(glob(osp.join(datapath, version, '*/im0.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1')] ]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            im2_path, im3_path = self.image_list[index][1], self.image_list[index][2]
            data['im2'] = np.array(frame_utils.read_gen(im2_path), dtype=np.uint8)
            data['im3'] = np.array(frame_utils.read_gen(im3_path), dtype=np.uint8)

            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][..., None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][..., None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            gt_path = self.image_list[index][0].replace('im0.png', f"{self.gt_name}.pfm")
            data['gt'] = np.expand_dims(frame_utils.readPFM(gt_path), -1)
            data['validgt'] = data['gt'] < 5000

            if not self.occ:
                mask_path = self.image_list[index][0].replace('im0.png', f"{self.mask_name}.png")
                mask = np.expand_dims(cv2.imread(mask_path, -1), -1)
                mask = (mask == 255).astype(np.float32)
                data['gt'] *= mask

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()

            return data

    def __len__(self):
        return len(self.image_list)

class KITTI(data.Dataset):
    def __init__(self, datapath, version="KITTI/2015/training/", occ=False, test=True):
        self.is_test = test
        self.version = version
        self.occ = occ

        self.disp_list = []
        self.image_list = []

        if "2015" in self.version:
            self.gt_name = "disp_occ_0" if self.occ else "disp_noc_0"
            self.im0 = "image_2"
            self.im1 = "image_3"
        else: 
            self.gt_name = "disp_occ" if self.occ else "disp_noc"
            self.im0 = "colored_0"
            self.im1 = "colored_1"

        image_list = sorted(glob(osp.join(datapath, version, self.im0, '*_10.png')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace(self.im0, self.im1)] ]

    def __getitem__(self, index):

        data = {}
        if self.is_test:

            im2_path, im3_path = self.image_list[index][1], self.image_list[index][2]
            data['im2'] = np.array(frame_utils.read_gen(im2_path), dtype=np.uint8)
            data['im3'] = np.array(frame_utils.read_gen(im3_path), dtype=np.uint8)


            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][..., None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][..., None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace(self.im0, self.gt_name))

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            return data

    def __len__(self):
        return len(self.image_list)

# class NS_Data(data.Dataset):
#     def __init__(self, datapath, training_file, scalor_file, conf_threshold=0.5, disp_threshold=512., aug_params=None, scale=1):
#         self.augmentor = DisparityAugmentor(**aug_params)
#         self.scale=scale
#         self.disp_threshold = disp_threshold
#         self.conf_threshold = conf_threshold
#         self.disp_list = []
#         self.image_list = []

#         training_file = open(training_file, 'r')

#         for line in training_file.readlines():
#             left, center, right, disp, confidence = line.split()
#             self.image_list += [[os.path.join(datapath,left), 
#                                  os.path.join(datapath,center), 
#                                  os.path.join(datapath,right), 
#                                  os.path.join(datapath,disp),
#                                  os.path.join(datapath,confidence)]]

#     def __getitem__(self, index):

#         data = {}

#         index = index % len(self.image_list)

#         data['im0'] = frame_utils.read_gen(self.image_list[index][0])
#         data['im1'] = frame_utils.read_gen(self.image_list[index][1])
#         data['im2'] = frame_utils.read_gen(self.image_list[index][2])
#         data['disp'] = cv2.imread(self.image_list[index][3], -1) / 64.
#         data['conf'] = cv2.imread(self.image_list[index][4], -1) / 65536.

#         split = self.image_list[index][1].split('/')
#         seq = split[-4]
#         img = split[-1]

#         data['im0'] = np.array(data['im0']).astype(np.uint8)
#         data['im1'] = np.array(data['im1']).astype(np.uint8)
#         data['im2'] = np.array(data['im2']).astype(np.uint8)
        
#         data['disp'] = np.squeeze(np.array(data['disp']).astype(np.float32))
#         data['conf'] = np.squeeze(np.array(data['conf']).astype(np.float32))
#         data['disp'] = data['disp'] * np.squeeze((data['conf'] > self.conf_threshold))
#         data['disp'][np.isinf(data['disp'])] = 0  
#         data['disp'][data['disp']> self.disp_threshold ] = 0
        
#         if self.scale != 1:
#             h, w = data['im2'].shape[0]//self.scale, data['im2'].shape[1]//self.scale
#             data['im0'] = cv2.resize(data['im0'], (w, h), interpolation=cv2.INTER_NEAREST)
#             data['im1'] = cv2.resize(data['im1'], (w, h), interpolation=cv2.INTER_NEAREST)
#             data['im2'] = cv2.resize(data['im2'], (w, h), interpolation=cv2.INTER_NEAREST)
#             data['disp'] = cv2.resize(data['disp'], (w, h), interpolation=cv2.INTER_NEAREST)
#             data['conf'] = cv2.resize(data['conf'], (w, h), interpolation=cv2.INTER_NEAREST)

#         # grayscale images
#         if len(data['im0'].shape) == 2:
#             data['im0'] = np.tile(data['im0'][...,None], (1, 1, 3))
#         else:
#             data['im0'] = data['im0'][..., :3]                

#         if len(data['im1'].shape) == 2:
#             data['im1'] = np.tile(data['im1'][...,None], (1, 1, 3))
#         else:
#             data['im1'] = data['im1'][..., :3]  

#         augm_data = self.augmentor(data['im0'], data['im1'], data['im2'], data['disp'], data['conf'])
#         augm_data['im0f'] = np.ascontiguousarray(augm_data['im0'][:,::-1])
#         augm_data['im1f'] = np.ascontiguousarray(augm_data['im1'][:,::-1])
#         augm_data['im2f'] = np.ascontiguousarray(augm_data['im2'][:,::-1])

#         for k in augm_data:
#             if augm_data[k] is not None:
#                 if len(augm_data[k].shape) == 3:
#                     augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
#                 else:
#                     augm_data[k] = torch.from_numpy(augm_data[k].copy()).float() 

#         # augm_data <- data['im0'], data['im1'], data['im2'], data['im0_aug'], data['im1_aug'], data['im2_aug'], data['label'], data['conf']
#         return augm_data

#     def __len__(self):
#         return len(self.image_list)


def fetch_dataloader(args):

    if args.dataset == 'kitti':
        if args.test:
            dataset = KITTI(args.datapath, version=args.version, occ=args.occ, test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=4, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))

    elif args.dataset == 'middlebury':
        if args.test:
            dataset = Middlebury(args.datapath, version=args.version, occ=args.occ, test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=4, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))

    # elif args.dataset == '3nerf':
    #     aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.5, 'do_flip': True}
    #     dataset = NS_Data(
    #         args.datapath, args.training_file, conf_threshold=args.conf_threshold, disp_threshold=args.disp_threshold,
    #         aug_params=aug_params
    #     )
    #     loader = data.DataLoader(
    #         dataset, batch_size=args.batch_size, persistent_workers=True,
    #         pin_memory=False, shuffle=True, num_workers=8, drop_last=True
    #     )

    return loader

