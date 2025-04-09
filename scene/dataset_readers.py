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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: dict
    image_name: dict
    image_HDR: np.array
    image_HDR_name: str
    width: int
    height: int
    expos: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)  # 3*N
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, white_background):
    cam_infos = []

    ejson_file = os.path.join(os.path.dirname(images_folder), "exposure.json")
    with open(ejson_file) as f:
        exps_dict = json.load(f)

    # cam_extrinsics_sorted = sorted(list(cam_extrinsics.values()), key = lambda x : int(x.name.split('_')[-2]))
    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        # uid = intr.id

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        exp_idxs = [0,1,2,3,4]
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        image_dict = {}
        image_name_dict = {}
        expos_dict = {}
        for i, exp_id in enumerate(exp_idxs):
            image_name = os.path.basename(extr.name).split(".")[0]
            extension = os.path.basename(extr.name).split(".")[-1]
            file_path = image_name[:-1] + "{}.".format(exp_id) + extension
            try:
                expos = exps_dict[file_path] 
            except:
                continue
            image_path = os.path.join(images_folder, file_path)
            image_name = Path(file_path).stem   
            image = Image.open(image_path)
            mode = image.mode
            if mode == "RGBA":
                if white_background:
                    norm_data = np.array(image) / 255.0
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                else:
                    image = image.convert("RGB")
            image_dict[exp_id] = image
            image_name_dict[exp_id] = image_name
            expos_dict[exp_id] = expos
            
        ### load hdr gt images if available
        images_HDR_folder = images_folder + "_hdr"
        image_HDR_name = image_name.split('_')[0] + "_hdr_{:03d}.exr".format(int(image_name.split('_')[-2]))
        image_HDR_path = os.path.join(images_HDR_folder, image_HDR_name)
        if os.path.exists(image_HDR_path):
            image_HDR = cv2.imread(image_HDR_path, cv2.IMREAD_UNCHANGED)
            if mode == "RGBA" and white_background:
                norm_data = image_HDR
                arr =  norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            else:
                arr = image_HDR[:,:,:3]
            image_HDR = arr[:, :, ::-1]
            image_HDR_name = Path(image_HDR_name).stem   
        else:
            image_HDR = None
            image_HDR_name = Path(image_HDR_name).stem   

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image_dict,
                            image_name=image_name_dict, image_HDR=image_HDR, image_HDR_name=image_HDR_name, width=width, height=height, expos=expos_dict)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')

    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, white_background, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), white_background=white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name[0])

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if "train" in c.image_name[0]]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if "test" in c.image_name[0]]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    exposurefile = transformsfile.replace("transforms", "exposure")

    with open(os.path.join(path, transformsfile)) as json_file, open(os.path.join(path, exposurefile)) as ejson_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        exps_dict = json.load(ejson_file)

        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            exp_idxs = [0,2,4] if "train" in transformsfile else [0,1,2,3,4]
            pre = "train/" if "train" in transformsfile else "test/"
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            for i, exp_id in enumerate(exp_idxs):
                if "train" in transformsfile and idx % len(exp_idxs) != i:
                    continue
                file_path = frame["file_path"] + "_{}".format(exp_id) + extension
                expos = exps_dict[file_path]
                image_path = os.path.join(path, file_path)
                image_name = Path(file_path).stem
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                if white_background:
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                else:
                    arr = norm_data[:,:,:3]
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

                ### load hdr gt images if available
                a = frame["file_path"].split('/')[-2] + "_hdr"
                b = int(frame["file_path"].split('/')[-1].split('_')[-1])
                image_HDR_path = os.path.join(path, a, "hdr_{:03d}.exr".format(b))
                if os.path.exists(image_HDR_path):
                    norm_data = np.array(cv2.imread(image_HDR_path, cv2.IMREAD_UNCHANGED))
                    if white_background:
                        arr =  norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    else:
                        arr = norm_data[:,:,:3]
                    image_HDR = arr[:, :, ::-1]
                    image_HDR_name = pre + "hdr_{:03d}".format(b)
                else:
                    image_HDR = None
                    image_HDR_name = pre + "hdr_{:03d}".format(b)

                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, image_HDR=image_HDR, image_HDR_name=image_HDR_name, width=image.size[0], height=image.size[1], expos_id=exp_id, expos=expos))
 
    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}