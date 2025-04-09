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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    feat_crf = pc.mlp_proj(feat)
    feat_crf = repeat(feat_crf, 'n (c) -> (n k) (c)', k=pc.n_offsets)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, feat_crf], dim=-1)
    masked = concatenated_all[mask]

    scaling_repeat, repeat_anchor, color, scale_rot, offsets, feat_crf = masked.split([6, 3, 3, 7, 3, pc.feat_crf_dim], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, feat_crf
    else:
        return xyz, color, opacity, scaling, rot, feat_crf


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, exposure, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, training_iter=None, ret_pts=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # is_training = pc.get_color_mlp.training
        
    if training_iter is not None:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, feat_crf_3d = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=True)
    else:
        xyz, color, opacity, scaling, rot, feat_crf_3d = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=False)
    
    log_gauss_rgb_h = color
    gauss_rgb_h = torch.exp(log_gauss_rgb_h)
    if torch.isinf(gauss_rgb_h.mean()).sum() > 0:
        print("inf", log_gauss_rgb_h.max(), gauss_rgb_h.mean())
        exit()

    expos = torch.tensor(float(exposure)).cuda()
    gauss_lnx = log_gauss_rgb_h + torch.log(expos)

    gauss_rgb_l_glo, gauss_rgb_l_loc, gauss_rgb_l_uncertainty = pc.tone_mapper(gauss_lnx, feat_crf_3d, training_iter)   

    colors_precomp = torch.cat([gauss_rgb_l_glo, gauss_rgb_l_loc, gauss_rgb_h, feat_crf_3d, gauss_rgb_l_uncertainty], dim=-1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg_color = torch.cat([bg_color, bg_color, bg_color, torch.tensor([0]*(3+pc.feat_crf_dim)).cuda()])
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        ret_pts=ret_pts,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, alpha, point_list, means2D, conic_opacity, gauss_depth, start_end, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    depth = depth / (alpha + 1e-6)
    image_LDR_from3d_glo = rendered_image[0:3]
    image_LDR_from3d_loc = rendered_image[3:6]
    image_HDR = rendered_image[6:9]
    image_LDR_from3d_loc_uncert = rendered_image[13:].clamp_min(0.1)
    feat_crf_2d = rendered_image[9:13]
    fmap = feat_crf_2d.clone()
    feat_crf_2d = feat_crf_2d.reshape([pc.feat_crf_dim, -1]).transpose(0, 1)

    tmp = image_HDR.clone().reshape([3, -1]).transpose(0, 1)
    pixel_lnx = torch.log(tmp*expos + 1e-5)
  
    image_LDR_from2d_glo, image_LDR_from2d_loc, image_LDR_from2d_loc_uncert = pc.tone_mapper(pixel_lnx, feat_crf_2d, training_iter)
    image_LDR_from2d_glo = image_LDR_from2d_glo.transpose(0, 1).reshape(image_HDR.shape)  
    image_LDR_from2d_loc = image_LDR_from2d_loc.transpose(0, 1).reshape(image_HDR.shape)  
    image_LDR_from2d_loc_uncert = image_LDR_from2d_loc_uncert.transpose(0, 1).reshape(3, image_HDR.shape[1], image_HDR.shape[2])

    u2d = image_LDR_from2d_loc_uncert.clone()
    u3d = image_LDR_from3d_loc_uncert.clone()
    merge_mask = image_LDR_from2d_loc_uncert.pow(2) / (image_LDR_from2d_loc_uncert.pow(2) + image_LDR_from3d_loc_uncert.pow(2)) 
    merge_mask = merge_mask.detach()
    merge = merge_mask * image_LDR_from3d_loc + (1-merge_mask) * image_LDR_from2d_loc


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if training_iter is not None:
        return {"render": image_LDR_from3d_loc,
                "render_merge": merge,
                "image_LDR_from2d_loc_uncert": image_LDR_from2d_loc_uncert,
                "image_LDR_from3d_loc_uncert": image_LDR_from3d_loc_uncert,                
                "image_LDR_from3d_glo": image_LDR_from3d_glo,
                "image_LDR_from3d_loc": image_LDR_from3d_loc,
                "image_LDR_from2d_glo": image_LDR_from2d_glo,
                "image_LDR_from2d_loc": image_LDR_from2d_loc,
                "image_HDR": image_HDR,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "depth": depth,
                "weight_sum": alpha,
                }
    else:
        return {"render": image_LDR_from3d_loc,
                "fmap": fmap,
                "u3d": u3d,
                "u2d": u2d,
                "render_merge": merge,
                "merge_mask": merge_mask,
                "image_LDR_from2d_loc_uncert": image_LDR_from2d_loc_uncert,
                "image_LDR_from3d_loc_uncert": image_LDR_from3d_loc_uncert, 
                "image_LDR_from3d_glo": image_LDR_from3d_glo,
                "image_LDR_from3d_loc": image_LDR_from3d_loc,
                "image_LDR_from2d_glo": image_LDR_from2d_glo,
                "image_LDR_from2d_loc": image_LDR_from2d_loc,
                "image_HDR": image_HDR,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": depth,
                "weight_sum": alpha,
                "point_list": point_list,
                "means2D": means2D,
                "conic_opacity": conic_opacity,
                "start_end": start_end,
                "gauss_depth": gauss_depth,
                "gauss_rgb_h": gauss_rgb_h,
                "gauss_rgb_l": gauss_rgb_l_loc
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        ret_pts=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    return radii_pure > 0
