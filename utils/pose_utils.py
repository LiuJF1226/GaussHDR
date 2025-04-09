import numpy as np

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    bottom = np.reshape([0,0,0,1.], [1,4])
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), bottom], 0)
    return c2w

def render_path_spiral(poses, focal=100, zrate=0.5, rots=2, N_views=120):
    c2w_avg = poses_avg(poses)
    # tt = poses[:,:3,3]
    tt = ptstocam(poses.transpose(1,2,0)[:3,3,:].T, c2w_avg)
    rads = np.percentile(np.abs(tt), 30, 0)
    rads = np.array(list(rads) + [1.])

    bottom = np.reshape([0,0,0,1.], [1,4])
    up = normalize(poses[:, :3, 1].sum(0))
    render_poses = []
    
    for theta in np.linspace(0., 2. * np.pi * rots, N_views+1)[:-1]:
        c = np.dot(c2w_avg[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads/1.5) 
        z = normalize(np.dot(c2w_avg[:3,:4], np.array([0,0,focal, 1.])) - c)
        pose = np.concatenate([viewmatrix(z, up, c), bottom], 0)  ## camera to world
        pose = np.linalg.inv(pose)  ## world to camera
        render_poses.append(pose)

    return render_poses

def render_path_spheric(poses, zh=None, radcircle=None, N_views=120):
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    # up = (poses[:,:3,3] - center).mean(0)
    up = poses[:,:3,1].mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_recenter = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_recenter[:,:3,3]), -1)))
    centroid = np.mean(poses_recenter[:,:3,3], 0)
    if zh is None or radcircle is None:
        zh = centroid[2]
        radcircle = np.sqrt(rad**2-zh**2)

    bottom = np.reshape([0,0,0,1.], [1,4])
    render_poses = []
    for th in np.linspace(0.,2.*np.pi, N_views):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(-camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        p = np.concatenate([p, bottom], 0)  ## camera to world
        p = np.concatenate([c2w, bottom], 0) @ p  ## correct recentered pose to the original
        p = np.linalg.inv(p)          ## world to camera
        render_poses.append(p)

    render_poses = np.stack(render_poses, 0)

    return render_poses