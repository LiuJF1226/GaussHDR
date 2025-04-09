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

import os
import random
import numpy.random
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, unit_expos_loss, draw_CRF
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
import shutil, pathlib
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_set, render_video
from metrics import evaluate
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# torch.set_num_threads(32)

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.data_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    gaussians.train()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    ## set for dataset.exp_mode == 1
    if len(scene.getTrainCameras()) % 3 == 0:
        pool = [0,2,4] * (len(scene.getTrainCameras()) // 3)
    else:
        pool = [0,2,4] * (len(scene.getTrainCameras()) // 3 + 1) 
    exp_id_pool = {}
    for c in scene.getTrainCameras():
        exp_id_pool[c.colmap_id] = pool[c.colmap_id]

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # if iteration >= 6000:
        #     gaussians.tone_mapper.freeze_global_tm()
 
        if dataset.exp_mode == 1:
            exp_id = exp_id_pool[viewpoint_cam.colmap_id]
        if dataset.exp_mode == 3:
            exp_id = random.choice([0, 2, 4])
            
        expos = viewpoint_cam.expos[exp_id]
        gt_image = viewpoint_cam.original_image[exp_id].cuda()

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, expos, visible_mask=voxel_visible_mask, retain_grad=retain_grad, training_iter=iteration)

        image_LDR_from3d_loc, image_LDR_from2d_glo, image_LDR_from2d_loc, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["image_LDR_from3d_loc"], render_pkg["image_LDR_from2d_glo"], render_pkg["image_LDR_from2d_loc"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

        image_LDR_from3d_glo = render_pkg["image_LDR_from3d_glo"]
        offset_selection_mask, scaling, opacity = render_pkg["selection_mask"], render_pkg["scaling"], render_pkg["neural_opacity"]
        image_LDR_from2d_loc_uncert = render_pkg["image_LDR_from2d_loc_uncert"]
        image_LDR_from3d_loc_uncert = render_pkg["image_LDR_from3d_loc_uncert"]


        ## scaling regularization loss from Scaffold-GS
        scaling_reg = scaling.prod(dim=1).mean()

        ## lossed from 3d tone mapping
        Ll1_from3d_glo = l1_loss(image_LDR_from3d_glo, gt_image)
        DSSIM_from3d_glo = 1.0 - ssim(image_LDR_from3d_glo, gt_image)
        Ll1_from3d_loc = l1_loss(image_LDR_from3d_loc, gt_image)
        DSSIM_from3d_loc = 1.0 - ssim(image_LDR_from3d_loc, gt_image)
        loss_from3d_glo = (1.0 - opt.lambda_dssim) * Ll1_from3d_glo + opt.lambda_dssim * DSSIM_from3d_glo
        loss_from3d_loc = (1.0 - opt.lambda_dssim) * Ll1_from3d_loc + opt.lambda_dssim * DSSIM_from3d_loc

        ## lossed from 2d tone mapping
        Ll1_from2d_glo = l1_loss(image_LDR_from2d_glo, gt_image)
        DSSIM_from2d_glo = 1.0 - ssim(image_LDR_from2d_glo, gt_image)
        Ll1_from2d_loc = l1_loss(image_LDR_from2d_loc, gt_image)  
        DSSIM_from2d_loc = 1.0 - ssim(image_LDR_from2d_loc, gt_image)    # [3,h,w]
        loss_from2d_glo = (1.0 - opt.lambda_dssim) * Ll1_from2d_glo + opt.lambda_dssim * DSSIM_from2d_glo
        loss_from2d_loc = (1.0 - opt.lambda_dssim) * Ll1_from2d_loc + opt.lambda_dssim * DSSIM_from2d_loc    

        ## uncertainty losses
        loss_3d_uncert = DSSIM_from3d_loc.detach() / (2*image_LDR_from3d_loc_uncert.pow(2)) + 0.5 * torch.log(image_LDR_from3d_loc_uncert)
        loss_2d_uncert = DSSIM_from2d_loc.detach() / (2*image_LDR_from2d_loc_uncert.pow(2)) + 0.5 * torch.log(image_LDR_from2d_loc_uncert)

        ## joint loss via uncertainty
        loss_gs = (image_LDR_from2d_loc_uncert.detach().pow(2) * loss_from3d_loc + image_LDR_from3d_loc_uncert.detach().pow(2) * loss_from2d_loc) / (image_LDR_from2d_loc_uncert.detach().pow(2) + image_LDR_from3d_loc_uncert.detach().pow(2)) 
       
        ## Different from that described in paper, we always include global tone-mapping losses during training.
        ## only loss_from2d_glo is enough
        loss_glo = dataset.gamma * loss_from2d_glo.mean() + 0.0 * loss_from3d_glo.mean()

        ## overall loss
        loss = loss_3d_uncert.mean() + loss_2d_uncert.mean() + loss_gs.mean() + loss_glo + 0.01*scaling_reg
        
        ## Similar to HDR-NeRF, we enforce unit exposure loss (only for synthetic scenes).
        if dataset.data_type == "synthetic":
            loss = loss + 0.5 * unit_expos_loss(gaussians.tone_mapper, 0.73)
            
        loss.backward()
        iter_end.record()   

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % 500 == 1:
                draw_CRF(scene.model_path, gaussians.tone_mapper, dataset.data_type)

            # Log and save
            training_report(tb_writer, iteration, loss, loss_3d_uncert.mean(), loss_2d_uncert.mean(), loss_gs.mean(), loss_glo, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if gaussians.tone_mapper_optimizer is not None:
                    gaussians.tone_mapper_optimizer.step()
                    gaussians.tone_mapper_optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration in testing_iterations:
                with torch.no_grad():
                    gaussians.eval()
                    render_video(dataset.model_path, iteration, scene.getTrainCameras()+scene.getTestCameras(), gaussians, pipe, background)
                    render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, background)
                    evaluate([dataset.model_path], "test", iteration)
                    render_set(dataset.model_path, "train", iteration, scene.getTestCameras(), gaussians, pipe, background)
                    # evaluate([dataset.model_path], "train", iteration)
                    gaussians.train()
                    
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, loss_3d_uncert, loss_2d_uncert, loss_gs, loss_glo, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/loss_3d_uncert', loss_3d_uncert.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_2d_uncert', loss_2d_uncert.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_gs', loss_gs.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/loss_glo', loss_glo.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.system("echo $CUDA_VISIBLE_DEVICES")
    print(f'using GPU {args.gpu}')

    # saveRuntimeCode(os.path.join(args.model_path, 'backup'))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    
    # All done
    print("\nTraining complete.")
