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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.pose_utils import render_path_spheric, render_path_spiral
from utils.general_utils import vis_depth
import numpy as np
import imageio
import math
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import matplotlib.cm as cm
import matplotlib as mpl
from PIL import Image 
tonemap_mu = lambda x : (torch.log(torch.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    LDR_from3d_glo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LDR_from3d_glo")
    LDR_from3d_loc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LDR_from3d_loc")
    LDR_from2d_glo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LDR_from2d_glo")
    LDR_from2d_loc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LDR_from2d_loc")
    LDR_merge_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LDR_merge")
    HDR_tm_path = os.path.join(model_path, name, "ours_{}".format(iteration), "HDR_tm")
    HDR_exr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "HDR_exr")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(LDR_from3d_glo_path, exist_ok=True)
    makedirs(LDR_from3d_loc_path, exist_ok=True)
    makedirs(LDR_from2d_glo_path, exist_ok=True)
    makedirs(LDR_from2d_loc_path, exist_ok=True)
    makedirs(LDR_merge_path, exist_ok=True)
    makedirs(HDR_tm_path, exist_ok=True)
    makedirs(HDR_exr_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    if views[0].original_image_HDR is not None:
        HDR_tm_gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "HDR_tm_gt")
        makedirs(HDR_tm_gt_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for exp_id in list(view.expos.keys()):
            expos = view.expos[exp_id]
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)            
            results = render(view, gaussians, pipeline, background, expos, visible_mask=voxel_visible_mask, ret_pts=True)
            LDR_from3d_glo = results["image_LDR_from3d_glo"]
            LDR_from3d_loc = results["image_LDR_from3d_loc"]
            LDR_from2d_glo = results["image_LDR_from2d_glo"]
            LDR_from2d_loc = results["image_LDR_from2d_loc"]
            LDR_merge = results["render_merge"]
            gt = view.original_image[exp_id][0:3, :, :]

            torchvision.utils.save_image(LDR_from3d_glo, os.path.join(LDR_from3d_glo_path, '{}'.format(view.image_name[exp_id]) + ".png"))
            torchvision.utils.save_image(LDR_from3d_loc, os.path.join(LDR_from3d_loc_path, '{}'.format(view.image_name[exp_id]) + ".png"))
            torchvision.utils.save_image(LDR_from2d_glo, os.path.join(LDR_from2d_glo_path, '{}'.format(view.image_name[exp_id]) + ".png"))
            torchvision.utils.save_image(LDR_from2d_loc, os.path.join(LDR_from2d_loc_path, '{}'.format(view.image_name[exp_id]) + ".png"))
            torchvision.utils.save_image(LDR_merge, os.path.join(LDR_merge_path, '{}'.format(view.image_name[exp_id]) + ".png"))      
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name[exp_id]) + ".png"))

            ### plot the teaser figure in the paper
            # if exp_id==0:
            #     # print(view.image_name[0])
            #     pixel_x = 498  
            #     pixel_y = 235
            #     # print(gt[:,pixel_y,pixel_x])
            #     # print(LDR_from3d_glo[:,pixel_y,pixel_x])
            #     # print(LDR_from2d_glo[:,pixel_y,pixel_x])
            #     # print(results["image_HDR"][:,pixel_y,pixel_x], view.original_image_HDR[0:3, pixel_y,pixel_x])
            #     query_pixel_gaussian(results["point_list"], results["means2D"], results["conic_opacity"], results["start_end"], results["gauss_depth"], results["gauss_rgb_h"], results["gauss_rgb_l"], pixel_x, pixel_y)
            #     exit()

        rendering_h = results["image_HDR"]
        if view.original_image_HDR is not None:  ## synthetic scenes
            gt_HDR = view.original_image_HDR[0:3, :, :]
            gt_HDR_tm = tonemap_mu(gt_HDR / gt_HDR.max())
            gt_HDR_tm = torch.nn.functional.interpolate(gt_HDR_tm.unsqueeze(0), rendering_h.shape[1:], mode='bilinear', align_corners=True)[0]
            torchvision.utils.save_image(gt_HDR_tm, os.path.join(HDR_tm_gt_path, '{}'.format(view.image_name[exp_id][:-2]) + ".png"))
            rendering_h_tm = tonemap_mu(rendering_h / gt_HDR.max())
            torchvision.utils.save_image(rendering_h_tm, os.path.join(HDR_tm_path, '{}'.format(view.image_name[exp_id][:-2]) + ".png"))
        else:    ## real scenes
            rendering_h_tm = tonemap_mu(rendering_h / rendering_h.max())
            torchvision.utils.save_image(rendering_h_tm, os.path.join(HDR_tm_path, '{}'.format(view.image_name[exp_id][:-2]) + ".png"))
        HDR_exr = rendering_h.clone().permute(1,2,0).cpu().numpy()[:, :, ::-1]
        cv2.imwrite(os.path.join(HDR_exr_path, '{}'.format(view.image_name[exp_id][:-2]) + ".exr"), HDR_exr)

        depth = results["depth"]
        depth = vis_depth(depth[0].cpu().numpy())
        cv2.imwrite(os.path.join(depth_path, '{}'.format(view.image_name[exp_id][:-2]) + ".png"), depth)


def query_pixel_gaussian(point_list, means2D, conic_opacity, start_end, gauss_depth, gauss_rgb_h, gauss_rgb_l, pixel_x, pixel_y):
    start = start_end[0, pixel_y, pixel_x]
    end = start_end[1, pixel_y, pixel_x]
    gausses_id = point_list[start:end].type(torch.long)
    pixel_conic_opa = conic_opacity[gausses_id]
    pixel_means2D = means2D[gausses_id]
    pixel_gauss_depth = gauss_depth[gausses_id]
    pixel_gauss_rgb_h = gauss_rgb_h[gausses_id]
    pixel_gauss_rgb_l = gauss_rgb_l[gausses_id]

    T = 1
    weight = []
    select_id = []
    for i in range(len(gausses_id)):
        dx = pixel_x - pixel_means2D[i, 0]
        dy = pixel_y - pixel_means2D[i, 1]
        power = -0.5*(pixel_conic_opa[i,0]*(dx*dx) + pixel_conic_opa[i,2]*(dy*dy)) - pixel_conic_opa[i,1]*dx*dy
        if power > 0:
            continue
        alpha = torch.clamp_max(pixel_conic_opa[i,3] * torch.exp(power), 0.99)
        if alpha < 1 / 255:
            continue
        test_T = T * (1 - alpha)
        if test_T < 0.0001:
            break
        weight.append(alpha*T)
        select_id.append(i)
        T = test_T
    weight = torch.tensor(weight).cuda()


    z = pixel_gauss_depth[select_id]
    rgb_h = pixel_gauss_rgb_h[select_id]
    rgb_l = pixel_gauss_rgb_l[select_id]
    
    # z = torch.linspace(z.min(), z.max(), z.shape[0])

    plt.xlabel("depth")
    plt.ylabel("weight")
    plt.plot(z.cpu().numpy(), weight.cpu().numpy())


    plt.grid()
    plt.savefig(os.path.join("./", 'weight.png'))

    plt.clf()

    fig, axs = plt.subplots(2, 2)

    # axs[0, 0].plot(z.cpu().numpy(), rgb_h[:,0].cpu().numpy(), color='r', label='red')
    axs[0, 0].plot(z.cpu().numpy(), rgb_h[:,1].cpu().numpy(), color='g', label='green')
    axs[0, 0].plot(z.cpu().numpy(), rgb_h[:,2].cpu().numpy(), color='b', label='color')
    axs[0, 0].plot(z.cpu().numpy(), rgb_h[:,0].cpu().numpy(), color='r', label='red')
    axs[0, 0].legend(['red', 'green', 'blue'])
    axs[0, 0].set_title('color_HDR')

    # axs[0, 1].plot(z.cpu().numpy(), (weight*rgb_h[:,0]).cpu().numpy(), color='r', label='red')
    axs[0, 1].plot(z.cpu().numpy(), (weight*rgb_h[:,1]).cpu().numpy(), color='g', label='green')
    axs[0, 1].plot(z.cpu().numpy(), (weight*rgb_h[:,2]).cpu().numpy(), color='b', label='color')
    axs[0, 1].plot(z.cpu().numpy(), (weight*rgb_h[:,0]).cpu().numpy(), color='r', label='red')
    axs[0, 1].legend(['red', 'green', 'blue'])
    axs[0, 1].set_title('weighted color_HDR')

    # axs[1, 0].plot(z.cpu().numpy(), rgb_l[:,0].cpu().numpy(), color='r', label='red')
    axs[1, 0].plot(z.cpu().numpy(), rgb_l[:,1].cpu().numpy(), color='g', label='green')
    axs[1, 0].plot(z.cpu().numpy(), rgb_l[:,2].cpu().numpy(), color='b', label='color')
    axs[1, 0].plot(z.cpu().numpy(), rgb_l[:,0].cpu().numpy(), color='r', label='red')
    axs[1, 0].legend(['red', 'green', 'blue'])
    axs[1, 0].set_title('color_LDR')

    # axs[1, 1].plot(z.cpu().numpy(), (weight*rgb_l[:,0]).cpu().numpy(), color='r', label='red')
    axs[1, 1].plot(z.cpu().numpy(), (weight*rgb_l[:,1]).cpu().numpy(), color='g', label='green')
    axs[1, 1].plot(z.cpu().numpy(), (weight*rgb_l[:,2]).cpu().numpy(), color='b', label='color')
    axs[1, 1].plot(z.cpu().numpy(), (weight*rgb_l[:,0]).cpu().numpy(), color='r', label='red')
    axs[1, 1].legend(['red', 'green', 'blue'])
    axs[1, 1].set_title('weighted color_LDR')

    plt.tight_layout()
    plt.savefig(os.path.join("./", 'weighted_color.png'))


def render_video(model_path, iteration, views, gaussians, pipeline, background, N_views=80, spherify=False):
    poses = {}
    for c in views:
        if c.image_HDR_name in poses:
            continue
        R = np.transpose(c.R)
        T = c.T
        w2c = np.concatenate([R, T.reshape([-1, 1])], 1)
        w2c = np.concatenate([w2c , np.array([0,0,0,1]).reshape([1, -1])], 0)
        c2w = np.linalg.inv(w2c)
        poses[c.image_HDR_name] = c2w
    poses = list(poses.values())
    poses = np.stack(poses, 0)

    if spherify:
        ## For 360 scene, zh and radcircle will be set automatically if not provided (None)
        ## The z-axis of spheric cameras point to the scene center
        ## zh controls camera height, negative to be higher than scene center while positive to be lower
        ## radcircle controls the horizontal distance of the camera from scene center
        render_poses = render_path_spheric(poses, zh=None, radcircle=None, N_views=N_views)
    else:
        ## For forward-facing scene
        render_poses = render_path_spiral(poses, focal=100, N_views=N_views)

    min_exp = min(views[0].expos.values())
    max_exp = max(views[0].expos.values())
    ev_min_exp = np.log(min_exp) / np.log(2)
    ev_max_exp = np.log(max_exp) / np.log(2)
    render_exps = np.linspace(ev_min_exp, ev_max_exp, N_views//2) # the exposure denotes exposure value (EV) 
    render_exps = 2 ** render_exps
    render_exps = np.concatenate([render_exps, render_exps[::-1]])

    render_cameras = []
    for i, p in enumerate(render_poses):
        R = np.transpose(p[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = p[:3, 3]
        img = torch.zeros((3, views[0].image_height, views[0].image_width))
        cam = Camera(colmap_id=None, R=R, T=T, FoVx=views[0].FoVx, FoVy=views[0].FoVy, image={0:img}, gt_alpha_mask={0:None}, image_name=None, image_HDR=None, image_HDR_name=None, uid=None, expos=None)
        render_cameras.append(cam)
    
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
    tonemap_mu = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1) * 255).astype(np.uint8)
    tonemapReinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0)

    rgbs = [];  rgbs_h = []
    for idx, view in enumerate(tqdm(render_cameras, desc="Rendering progress")):
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)            
        results = render(view, gaussians, pipeline, background, render_exps[idx], visible_mask=voxel_visible_mask)
        rendering = results["render_merge"]
        rgbs.append(rendering.permute(1,2,0).cpu().numpy())
        rendering_h = results["image_HDR"]
        rgbs_h.append(rendering_h.permute(1,2,0).cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    rgbs_h = np.stack(rgbs_h, 0)
    rgbs = to8b(rgbs)
    rgbs_h = rgbs_h/(np.max(rgbs_h))
    # rgbs_h = np.clip(rgbs_h/(np.mean(np.max(rgbs_h, 0))), 0, 1)


    rgbs_h = tonemap_mu(rgbs_h)  ## use mu-law

    # Reinhards = [] 
    # for h in rgbs_h:
    #     h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
    #     Reinhards.append(h)
    # rgbs_h = np.stack(Reinhards, 0)

    render_path = os.path.join(model_path)
    makedirs(render_path, exist_ok=True)
    if spherify:
        moviebase = os.path.join(render_path, 'spheric_{}_'.format(iteration))
    else:
        moviebase = os.path.join(render_path, 'spiral_{}_'.format(iteration))
    imageio.mimwrite(moviebase + 'rgb_l.mp4', rgbs, fps=15, quality=8)
    imageio.mimwrite(moviebase + 'rgb_h.mp4', rgbs_h, fps=15, quality=8)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video : bool, spherify : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.data_type)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_video:
            render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras()+scene.getTestCameras(), gaussians, pipeline, background, spherify=spherify)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--spherify", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default='0')
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.system("echo $CUDA_VISIBLE_DEVICES")
    print(f'using GPU {args.gpu}')

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.spherify)