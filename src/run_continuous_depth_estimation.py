# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image

import src.models as models
import cv2
import numpy as np
from src.util import disp2rgb, str2bool
import random
import torch.nn.functional as F
from natsort import natsorted
from src.read_pfm import readPFM
import math
from tifffile import imread

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

# Parse Arguments
parser = argparse.ArgumentParser(allow_abbrev=False)

# Experiment Type
parser.add_argument("--arch", type=str, default="bi3dnet_continuous_depth_2D")

parser.add_argument("--bi3dnet_featnet_arch", type=str, default="featextractnetspp")
parser.add_argument("--bi3dnet_segnet_arch", type=str, default="segnet2d")
parser.add_argument("--bi3dnet_refinenet_arch", type=str, default="disprefinenet")
parser.add_argument("--bi3dnet_regnet_arch", type=str, default="segregnet3d")
parser.add_argument("--bi3dnet_max_disparity", type=int, default=192)
parser.add_argument("--regnet_out_planes", type=int, default=16)
parser.add_argument("--disprefinenet_out_planes", type=int, default=32)
parser.add_argument("--bi3dnet_disps_per_example_true", type=str2bool, default=True)

# Input
parser.add_argument("--pretrained", type=str)
parser.add_argument("--disp_range_min", type=int)
parser.add_argument("--disp_range_max", type=int)

parser.add_argument("--dataset", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--threshold", type=int, default=3)
parser.add_argument('--within_max_disp', action='store_true')

args, unknown = parser.parse_known_args()


##############################################################################################################
def main():
    options = vars(args)
    # print("==> ALL PARAMETERS")
    # for key in options:
    #     print("{} : {}".format(key, options[key]))

    out_dir = "out"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # base_name = os.path.splitext(os.path.basename(args.img_left))[0]

    # Model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
    else:
        print("Need an input model")
        exit()

    # print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](options, network_data).cuda()
    model.eval()
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # Inputs
    if args.dataset == 'sceneflow':
        file_path = args.data_path
        directory = os.path.join(file_path, 'frame_finalpass', 'TEST')
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                            os.path.isdir(os.path.join(sub_folder, seq))]

        left_data = []
        for seq_folder in seq_folders:
            left_data += [os.path.join(seq_folder, 'left', img) for img in
                          os.listdir(os.path.join(seq_folder, 'left'))]

        left_data = natsorted(left_data)
        right_data = [left.replace('left', 'right') for left in left_data]
        disp_data = [left.replace('frame_finalpass', 'disparity').replace('.png', '.pfm') for left in left_data]

        directory = os.path.join(file_path, 'occlusion', 'TEST', 'left')
        occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        occ_data = natsorted(occ_data)
    elif args.dataset == 'kitti2015':
        file_path = args.data_path
        left_data = natsorted(
            [os.path.join(file_path, 'image_2/', img) for img in os.listdir(os.path.join(file_path, 'image_2/')) if
             img.find('_10') > -1])
        right_data = [img.replace('image_2/', 'image_3/') for img in left_data]
        disp_data = [img.replace('image_2/', 'disp_noc_0/') for img in left_data]
        occ_data = None
    elif args.dataset == 'middlebury':
        file_path = args.data_path
        left_data = [os.path.join(file_path, obj, 'im0.png') for obj in os.listdir(file_path)]
        right_data = [os.path.join(file_path, obj, 'im1.png') for obj in os.listdir(file_path)]
        disp_data = [os.path.join(file_path, obj, 'disp0GT.pfm') for obj in os.listdir(file_path)]
        occ_data = [os.path.join(file_path, obj, 'mask0nocc.png') for obj in os.listdir(file_path)]

        left_data = natsorted(left_data)
        right_data = natsorted(right_data)
        disp_data = natsorted(disp_data)
        occ_data = natsorted(occ_data)

    elif args.dataset == 'scared':
        file_path = args.data_path
        left_data = natsorted(
            [os.path.join(file_path, 'left/', img) for img in os.listdir(os.path.join(file_path, 'left'))])
        right_data = [img.replace('left/', 'right/') for img in left_data]
        disp_data = [img.replace('left/', 'disparity/').replace('.png', '.tiff') for img in left_data]
        occ_data = [img.replace('left/', 'occlusion/') for img in left_data]

    avg_error = 0.0
    avg_wrong = 0.0
    avg_total = 0.0
    for idx in range(len(left_data)):
        # read data
        img_left = Image.open(left_data[idx]).convert("RGB")
        img_right = Image.open(right_data[idx]).convert("RGB")
        img_left = transforms.functional.to_tensor(img_left)
        img_right = transforms.functional.to_tensor(img_right)
        img_left = transforms.functional.normalize(img_left, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_right = transforms.functional.normalize(img_right, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_left = img_left.type(torch.cuda.FloatTensor)[None, :, :, :]
        img_right = img_right.type(torch.cuda.FloatTensor)[None, :, :, :]

        # disp
        if args.dataset == 'sceneflow' or args.dataset == 'middlebury':
            disp_left, _ = readPFM(disp_data[idx])
        elif args.dataset == 'kitti2015':
            disp_left = np.array(Image.open(disp_data[idx])).astype(np.float) / 256.
        elif args.dataset == 'scared':
            disp_left = imread(disp_data[idx]).squeeze(0)

        # occ
        if occ_data is not None:
            if args.dataset == 'sceneflow':
                occ = np.array(Image.open(occ_data[idx])).astype(np.bool)
            elif args.dataset == 'middlebury' or args.dataset == 'scared':
                occ = np.array(Image.open(occ_data[idx])) != 255
        else:
            occ = disp_left <= 0.0

        disp_left[occ] = 0.0
        disp_left = np.ascontiguousarray(disp_left)
        disp_left = transforms.functional.to_tensor(disp_left)
        disp_left = disp_left.cuda()[None,]

        # Prepare Disparities
        max_disparity = args.disp_range_max
        min_disparity = args.disp_range_min

        assert max_disparity % 3 == 0 and min_disparity % 3 == 0, "disparities should be divisible by 3"

        if args.arch == "bi3dnet_continuous_depth_3D":
            assert (max_disparity - min_disparity) % 48 == 0, \
                "for 3D regularization the difference in disparities should be divisible by 48"

        max_disp_levels = (max_disparity - min_disparity) + 1

        max_disparity_3x = int(max_disparity / 3)
        min_disparity_3x = int(min_disparity / 3)
        max_disp_levels_3x = (max_disparity_3x - min_disparity_3x) + 1
        disp_3x = np.linspace(min_disparity_3x, max_disparity_3x, max_disp_levels_3x, dtype=np.int32)
        disp_long_3x_main = torch.from_numpy(disp_3x).type(torch.LongTensor).cuda()
        disp_float_main = np.linspace(min_disparity, max_disparity, max_disp_levels, dtype=np.float32)
        disp_float_main = torch.from_numpy(disp_float_main).type(torch.float32).cuda()
        delta = 1
        d_min_GT = min_disparity - 0.5 * delta
        d_max_GT = max_disparity + 0.5 * delta
        disp_long_3x = disp_long_3x_main[None, :].expand(img_left.shape[0], -1)
        disp_float = disp_float_main[None, :].expand(img_left.shape[0], -1)

        # Pad Inputs
        h = img_left.shape[2]
        w = img_left.shape[3]
        tw = math.ceil(w / 96) * 96
        th = math.ceil(h / 96) * 96
        assert tw % 96 == 0, "image dimensions should be multiple of 96"
        assert th % 96 == 0, "image dimensions should be multiple of 96"
        x1 = random.randint(0, max(0, w - tw))
        y1 = random.randint(0, max(0, h - th))
        pad_w = tw - w if tw - w > 0 else 0
        pad_h = th - h if th - h > 0 else 0
        pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        img_left = img_left[:, :, y1: y1 + min(th, h), x1: x1 + min(tw, w)]
        img_right = img_right[:, :, y1: y1 + min(th, h), x1: x1 + min(tw, w)]
        img_left_pad = pad_opr(img_left)
        img_right_pad = pad_opr(img_right)
        disp_left = disp_left[:, y1: y1 + min(th, h), x1: x1 + min(tw, w)]
        disp_left_pad = pad_opr(disp_left)

        # Inference
        with torch.no_grad():
            if args.arch == "bi3dnet_continuous_depth_2D":
                output_seg_low_res_upsample, output_disp_normalized = model(
                    img_left_pad, img_right_pad, disp_long_3x
                )
                output_seg = output_seg_low_res_upsample
            else:
                (
                    output_seg_low_res_upsample,
                    output_seg_low_res_upsample_refined,
                    output_disp_normalized_no_reg,
                    output_disp_normalized,
                ) = model(img_left_pad, img_right_pad, disp_long_3x)
                output_seg = output_seg_low_res_upsample_refined

            output_seg = output_seg[:, :, pad_h:, pad_w:]
            output_disp_normalized = output_disp_normalized[:, :, pad_h:, pad_w:]
            output_disp = torch.clamp(
                output_disp_normalized * delta * max_disp_levels + d_min_GT, min=d_min_GT, max=d_max_GT
            )

            if args.within_max_disp:
                mask = torch.logical_and(disp_left > 0.0, disp_left < args.bi3dnet_max_disparity)
            else:
                mask = disp_left > 0.0
            error = F.l1_loss(output_disp[mask], disp_left[mask], reduction='none')
            wrong = torch.sum(error > args.threshold).item()
            total = error.numel()

            avg_error += torch.mean(error)
            avg_wrong += float(wrong)
            avg_total += float(total)

            print('Index:', idx, 'EPE:', torch.mean(error).item(), 'px error', float(wrong) / float(total) * 100)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(disp_left[0, 0].data.cpu())
            # plt.figure()
            # plt.imshow(output_disp[0, 0].data.cpu())
            # plt.show()
            # asdf

        # # Write Results
        # max_disparity_color = 192
        # output_disp_clamp = output_disp[0, 0, :, :].cpu().clone().numpy()
        # output_disp_clamp[output_disp_clamp < min_disparity] = 0
        # output_disp_clamp[output_disp_clamp > max_disparity] = max_disparity_color
        # disp_np_ours_color = disp2rgb(output_disp_clamp / max_disparity_color) * 255.0
        # cv2.imwrite(
        #     os.path.join(out_dir, "%s_%s_%d_%d.png" % (base_name, args.arch, min_disparity, max_disparity)),
        #     disp_np_ours_color,
        # )
    print('Total EPE', avg_error.item() / len(left_data), 'Total px error', float(avg_wrong) / float(avg_total) * 100)

    return


if __name__ == "__main__":
    main()
