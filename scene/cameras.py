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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, image_HDR, image_HDR_name, uid, expos, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_HDR_name = image_HDR_name
        self.expos = expos
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = {}

        self.original_image_neighbors = []
        self.pose_to_neighbors = []
        self.id_neighbors = []
        
        for i in image.keys():
            self.original_image[i] = image[i].clamp(0.0, 1.0).to(self.data_device) if image[i] is not None else None
            self.image_width = self.original_image[i].shape[2]
            self.image_height = self.original_image[i].shape[1]

            if gt_alpha_mask[i] is not None:
                self.original_image[i] *= gt_alpha_mask[i].to(self.data_device)
            else:
                self.original_image[i] *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.original_image_HDR = torch.tensor(image_HDR.copy()).to(self.data_device).permute(2, 0, 1) if image_HDR is not None else None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        fx = self.image_width / (2 * np.tan(FoVx / 2))
        fy = self.image_height / (2 * np.tan(FoVy / 2))

        self.K = torch.tensor([[fx, 0, self.image_width/2, 0],
                [0, fy, self.image_height/2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).float().cuda()
        self.inv_K = self.K.clone().inverse()
        self.backproject_depth = BackprojectDepth(1, self.image_height, self.image_width).cuda()
        self.project_3d = Project3D(1, self.image_height, self.image_width).cuda()


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        points_warp = torch.matmul(T.unsqueeze(0), points)[:, :3, :]
        points_warp = points_warp.view(self.batch_size, 3, self.height, self.width)
        src_depth_warp = points_warp[:, 2:]

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)

        proj_mask_x = (pix_coords[:, 0:1] >= 0) & (pix_coords[:, 0:1] <= self.width-1)
        proj_mask_y = (pix_coords[:, 1:2] >= 0) & (pix_coords[:, 1:2] <= self.height-1)
        inside_mask = proj_mask_x & proj_mask_y

        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, src_depth_warp, inside_mask


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

