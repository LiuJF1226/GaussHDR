#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import collections

def readImages(renders_dir, gt_dir, mode="LDR"):
    renders = []
    gts = []
    image_names = []
    files = os.listdir(renders_dir)
    if mode=="HDR":
        sorted_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        sorted_files = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[-2]))
    for fname in sorted_files:
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, name="test", iteration=None):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir+"/{}".format(name))
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / name

        for method in os.listdir(test_dir):
            if iteration is not None and str(iteration) not in method:
                continue   ## only evaluate on the given iteration
            
            print("Iteration:", method)

            full_dict[scene_dir][method+"_LDR_from3d_loc"] = {}
            per_view_dict[scene_dir][method+"_LDR_from3d_loc"] = {}
            full_dict[scene_dir][method+"_LDR_from3d_glo"] = {}
            per_view_dict[scene_dir][method+"_LDR_from3d_glo"] = {}
            full_dict[scene_dir][method+"_LDR_from2d_loc"] = {}
            per_view_dict[scene_dir][method+"_LDR_from2d_loc"] = {}
            full_dict[scene_dir][method+"_LDR_from2d_glo"] = {}
            per_view_dict[scene_dir][method+"_LDR_from2d_glo"] = {}
            full_dict[scene_dir][method+"_LDR_merge"] = {}
            per_view_dict[scene_dir][method+"_LDR_merge"] = {}
            full_dict[scene_dir][method+"_HDR"] = {}
            per_view_dict[scene_dir][method+"_HDR"] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            LDR_from3d_loc_dir = method_dir / "LDR_from3d_loc"
            LDR_from3d_glo_dir = method_dir / "LDR_from3d_glo"
            LDR_from2d_loc_dir = method_dir / "LDR_from2d_loc"
            LDR_from2d_glo_dir = method_dir / "LDR_from2d_glo"
            LDR_merge_dir = method_dir / "LDR_merge"
            HDR_tm_gt_dir = method_dir / "HDR_tm_gt"
            HDR_tm_dir = method_dir / "HDR_tm"
            
            ## LDR evaluation
            # evaluate_LDR(per_view_dict, full_dict, scene_dir, LDR_from3d_glo_dir, gt_dir, method+"_LDR_from3d_glo")
            # evaluate_LDR(per_view_dict, full_dict, scene_dir, LDR_from2d_glo_dir, gt_dir, method+"_LDR_from2d_glo")
            evaluate_LDR(per_view_dict, full_dict, scene_dir, LDR_from3d_loc_dir, gt_dir, method+"_LDR_from3d_loc")
            evaluate_LDR(per_view_dict, full_dict, scene_dir, LDR_from2d_loc_dir, gt_dir, method+"_LDR_from2d_loc")
            evaluate_LDR(per_view_dict, full_dict, scene_dir, LDR_merge_dir, gt_dir, method+"_LDR_merge")         
            
            ## HDR evaluation
            try:
                ssims = []; psnrs = []; lpipss = []
                renders, gts, image_names = readImages(HDR_tm_dir, HDR_tm_gt_dir, mode="HDR")
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress ({})".format(method+"_HDR")):
                    ssims.append(ssim(renders[idx], gts[idx]).mean())
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='alex'))
                print("\nMethod:", method+"_HDR")
                print("  PSNR_HDR : {:.4f}".format(torch.tensor(psnrs).mean()))
                print("  SSIM_HDR : {:.4f}".format(torch.tensor(ssims).mean()))
                print("  LPIPS_HDR: {:.4f}".format(torch.tensor(lpipss).mean()))
                print("")

                full_dict[scene_dir][method+"_HDR"].update({"SSIM_HDR": torch.tensor(ssims).mean().item(),
                                                        "PSNR_HDR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS_HDR": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method+"_HDR"].update({"SSIM_HDR": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)}, "PSNR_HDR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}, "LPIPS_HDR": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            except:
                print("No HDR GT images for model", scene_dir)

        with open(scene_dir + "/results_{}.json".format(name), 'a') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view_{}.json".format(name), 'a') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


def evaluate_LDR(per_view_dict, full_dict, scene_dir, render_dir, gt_dir, method):
    renders, gts, image_names = readImages(render_dir, gt_dir)
    ssims = []; psnrs = []; lpipss = []
    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress ({})".format(method)):
        ssims.append(ssim(renders[idx], gts[idx]).mean())
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='alex'))

    per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)}, "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}, "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
    
    psnrs = collections.defaultdict(list)
    ssims = collections.defaultdict(list)
    lpipss = collections.defaultdict(list)
    for k, v in per_view_dict[scene_dir][method]["PSNR"].items():
        expo_id = int(k.split(".")[0].split("_")[-1])
        psnrs[expo_id].append(v)
    for k, v in per_view_dict[scene_dir][method]["SSIM"].items():
        expo_id = int(k.split(".")[0].split("_")[-1])
        ssims[expo_id].append(v)
    for k, v in per_view_dict[scene_dir][method]["LPIPS"].items():
        expo_id = int(k.split(".")[0].split("_")[-1])
        lpipss[expo_id].append(v)
    for expo_id, psnr_list in psnrs.items():
        ssim_list = ssims[expo_id]
        lpips_list = lpipss[expo_id]
        psnrs[expo_id] = torch.tensor(psnr_list).mean().item()
        ssims[expo_id] = torch.tensor(ssim_list).mean().item()
        lpipss[expo_id] = torch.tensor(lpips_list).mean().item()
    for expo_id in [0, 1, 2, 3, 4]:
        if expo_id not in psnrs:
            psnrs[expo_id] = float('nan')
            ssims[expo_id] = float('nan')
            lpipss[expo_id] = float('nan')

    per_view_dict[scene_dir][method].update({"SSIM_0": ssims[0], "SSIM_1": ssims[1], "SSIM_2": ssims[2], "SSIM_3": ssims[3], "SSIM_4": ssims[4]})
    per_view_dict[scene_dir][method].update({"PSNR_0": psnrs[0], "PSNR_1": psnrs[1], "PSNR_2": psnrs[2], "PSNR_3": psnrs[3], "PSNR_4": psnrs[4]})
    per_view_dict[scene_dir][method].update({"LPIPS_0": lpipss[0], "LPIPS_1": lpipss[1], "LPIPS_2": lpipss[2], "LPIPS_3": lpipss[3], "LPIPS_4": lpipss[4]})

    full_dict[scene_dir][method].update({"SSIM_024": (ssims[0]+ssims[2]+ssims[4])/3, "SSIM_13": (ssims[1]+ssims[3])/2, "SSIM": (ssims[0]+ssims[2]+ssims[4]+ssims[1]+ssims[3])/5, "PSNR_024": (psnrs[0]+psnrs[2]+psnrs[4])/3, "PSNR_13": (psnrs[1]+psnrs[3])/2, "PSNR": (psnrs[0]+psnrs[2]+psnrs[4]+psnrs[1]+psnrs[3])/5, "LPIPS_024": (lpipss[0]+lpipss[2]+lpipss[4])/3, "LPIPS_13": (lpipss[1]+lpipss[3])/2, "LPIPS": (lpipss[0]+lpipss[2]+lpipss[4]+lpipss[1]+lpipss[3])/5})

    print("\nMethod:", method)  
    print("  PSNR_0: {:.4f}   PSNR_1: {:.4f}   PSNR_2: {:.4f}   PSNR_3: {:.4f}   PSNR_4: {:.4f}".format(psnrs[0], psnrs[1], psnrs[2], psnrs[3], psnrs[4]))
    print("      PSNR: {:.4f}   PSNR_024: {:.4f}   PSNR_13: {:.4f}".format(full_dict[scene_dir][method]["PSNR"], full_dict[scene_dir][method]["PSNR_024"], full_dict[scene_dir][method]["PSNR_13"]))
    print("  SSIM_0: {:.4f}   SSIM_1: {:.4f}   SSIM_2: {:.4f}   SSIM_3: {:.4f}   SSIM_4: {:.4f}".format(ssims[0], ssims[1], ssims[2], ssims[3], ssims[4]))
    print("      SSIM: {:.4f}   SSIM_024: {:.4f}   SSIM_13: {:.4f}".format(full_dict[scene_dir][method]["SSIM"], full_dict[scene_dir][method]["SSIM_024"], full_dict[scene_dir][method]["SSIM_13"]))
    print("  LPIPS_0: {:.4f}   LPIPS_1: {:.4f}   LPIPS_2: {:.4f}   LPIPS_3: {:.4f}   LPIPS_4: {:.4f}".format(lpipss[0], lpipss[1], lpipss[2], lpipss[3], lpipss[4]))
    print("      LPIPS: {:.4f}   LPIPS_024: {:.4f}   LPIPS_13: {:.4f}".format(full_dict[scene_dir][method]["LPIPS"], full_dict[scene_dir][method]["LPIPS_024"], full_dict[scene_dir][method]["LPIPS_13"]))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.system("echo $CUDA_VISIBLE_DEVICES")
    print(f'using GPU {args.gpu}')

    evaluate(args.model_paths, "test")
    # evaluate(args.model_paths, "train")

