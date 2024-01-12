import os
import numpy as np
import random
import torch


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x,
                        dtype=dtype,
                        device=device,
                        requires_grad=requires_grad)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float32, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def slope_platform_stairs_terrain(terrain,
                                  slope=1,
                                  step_width=0.2,
                                  step_height=0.1,
                                  num_steps=5):
    """
    Generate a sloped followed by a platform and a downstair

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
        step_width (float): the width of the step [meters]
        step_height (float): the height of the step [meters]
        num_steps (int): number of the step of the stair
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = round(step_width / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)

    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width:(i + 1) *
                                                step_width, :] += height
        height += step_height

    platform_height = height
    slope_width = int(platform_height /
                      (np.abs(slope) *
                       (terrain.horizontal_scale / terrain.vertical_scale)))
    x = np.arange(0, slope_width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(slope_width, 1)
    platform_width = int(terrain.width - num_steps * step_width - slope_width)
    terrain.height_field_raw[num_steps * step_width:,
    np.arange(terrain.length)] = platform_height
    terrain.height_field_raw[-slope_width:,
    np.arange(terrain.length)] -= (
            platform_height * xx / slope_width).astype(
        terrain.height_field_raw.dtype)
    terrain.height_field_raw = terrain.height_field_raw[::-1]

    return terrain


def stairs_platform_slope_terrain(terrain,
                                  step_width=0.2,
                                  step_height=0.1,
                                  num_steps=10,
                                  slope=0.26):
    """
    Generate a stairs followed with a flat plane and a downslope

    Parameters:
        terrain (terrain): the terrain
        step_width (float): the width of the step [meters]
        step_height (float): the height of the step [meters]
        num_steps (int): number of the step of the stair
        slope (float): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = round(step_width / terrain.horizontal_scale)
    step_height = round(step_height / terrain.vertical_scale)

    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width:(i + 1) *
                                                step_width, :] += height
        height += step_height

    platform_height = height
    slope_width = int(platform_height /
                      (np.abs(slope) *
                       (terrain.horizontal_scale / terrain.vertical_scale)))
    x = np.arange(0, slope_width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(slope_width, 1)
    platform_width = int(terrain.width - num_steps * step_width - slope_width)
    terrain.height_field_raw[num_steps * step_width:,
    np.arange(terrain.length)] = platform_height
    terrain.height_field_raw[-slope_width:,
    np.arange(terrain.length)] -= (
            platform_height * xx / slope_width).astype(
        terrain.height_field_raw.dtype)

    return terrain


if torch.cuda.is_available():
    import string
    import cupy as cp


    def compute_meshes_normals_kernel():
        compute_meshes_normals_kernel = cp.ElementwiseKernel(
            in_params="raw U vertices, raw T triangles",
            out_params="raw U output",
            preamble=string.Template("""
                __device__ bool cross(float p1x, float p1y, float p1z,
                                        float p2x, float p2y, float p2z,
                                        float& normalx,float& normaly,float& normalz)
                {
                    float nx = p1y*p2z - p1z*p2y;
                    float ny = p1z*p2x - p1x*p2z;
                    float nz = p1x*p2y - p1y*p2x;
                    float l = sqrt(nx*nx + ny*ny + nz*nz);
                    if (l <1e-4) {
                        l = 1;
                    }
                    if (nz < 0.) {
                        nx *=-1;
                        ny *=-1;
                        nz *=-1;
                    }
                    atomicAdd(&normalx, nx/l);
                    atomicAdd(&normaly, ny/l);
                    atomicAdd(&normalz, nz/l);

                    return true;
                }
                """).substitute(),
            operation=string.Template("""
                int idx = i*2;
                
                T mesh_1_0 = triangles[idx*3];
                T mesh_1_1 = triangles[idx*3+1];
                T mesh_1_2 = triangles[idx*3+2];
                T mesh_2_0 = triangles[idx*3+3];
                T mesh_2_1 = triangles[idx*3+4];
                T mesh_2_2 = triangles[idx*3+5];

                U p01x = vertices[mesh_1_1*3] - vertices[mesh_1_0*3];
                U p01y = vertices[mesh_1_1*3+1] - vertices[mesh_1_0*3+1];
                U p01z = vertices[mesh_1_1*3+2] - vertices[mesh_1_0*3+2];
                U p02x = vertices[mesh_1_2*3] - vertices[mesh_1_0*3];
                U p02y = vertices[mesh_1_2*3+1] - vertices[mesh_1_0*3+1];
                U p02z = vertices[mesh_1_2*3+2] - vertices[mesh_1_0*3+2];
                cross(p01x, p01y, p01z, p02x, p02y, p02z, output[3*i], output[3*i+1],output[3*i+2]);
                
                p01x = vertices[mesh_2_1*3] - vertices[mesh_2_0*3];
                p01y = vertices[mesh_2_1*3+1] - vertices[mesh_2_0*3+1];
                p01z = vertices[mesh_2_1*3+2] - vertices[mesh_2_0*3+2];
                p02x = vertices[mesh_2_2*3] - vertices[mesh_2_0*3];
                p02y = vertices[mesh_2_2*3+1] - vertices[mesh_2_0*3+1];
                p02z = vertices[mesh_2_2*3+2] - vertices[mesh_2_0*3+2];
                cross(p01x, p01y, p01z, p02x, p02y, p02z, output[3*i], output[3*i+1],output[3*i+2]);
                
                output[3*i] /= 2.0;
                output[3*i+1] /= 2.0;
                output[3*i+2] /= 2.0;
                
                """).substitute(),
            name="compute_meshes_normals_kernel",
        )
        return compute_meshes_normals_kernel


    compute_meshes_normals_gpu_ = compute_meshes_normals_kernel()


    def compute_meshes_normals_gpu(num_rows, num_cols, vertices, triangles):
        vertices_gpu = cp.array(vertices, dtype=np.float32)
        triangles_gpu = cp.array(triangles, dtype=np.int32)
        output = cp.zeros((num_rows - 1, num_cols - 1, 3), dtype=np.float32)
        compute_meshes_normals_gpu_(vertices_gpu,
                                    triangles_gpu,
                                    output,
                                    size=((num_rows - 1) * (num_cols - 1)))
        output_torch = torch.as_tensor(output)
        return output_torch


def compute_meshes_normals(num_rows, num_cols, vertices, triangles):
    """
    Compute normal vectors for all trimesh of the designed terrain

    Parameters:
        num_rows (int): Number of the row of the terrain
        num_cols (int): Number of the column of the terrain
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    Returns:
        normals (np.array(float)): array of shape (num_rows-1, num_cols-1, 3). Normals of the terrain
    """
    normals = np.zeros((num_rows - 1, num_cols - 1, 3), dtype=np.float32)
    for i in range(num_rows - 1):
        start = 2 * i * (num_cols - 1)
        for j in range(num_cols - 1):
            idx_1 = start + 2 * j
            mesh_1, mesh_2 = triangles[idx_1], triangles[idx_1 + 1]
            normal1 = compute_triangle_normal(vertices[mesh_1[0]],
                                              vertices[mesh_1[1]],
                                              vertices[mesh_1[2]])
            normal2 = compute_triangle_normal(vertices[mesh_2[0]],
                                              vertices[mesh_2[1]],
                                              vertices[mesh_2[2]])
            new_normal = (
                                 normal1 +
                                 normal2) / 2.0  # np.mean(np.stack((normal1, normal2)), axis=0)
            if new_normal[2] < 0.:
                new_normal *= -1
            normals[i][j] = new_normal
    return torch.from_numpy(normals)


def compute_triangle_normal(p0, p1, p2):
    """
    Compute the normal vector of a trimesh

    Parameters:
        p0 (np.array(float)): array of shape (3,). Position of the trimesh's first vertice
        p1 (np.array(float)): array of shape (3,). Position of the trimesh's second vertice
        p2 (np.array(float)): array of shape (3,). Position of the trimesh's third vertice
    Returns:
        normal (np.array(float)): array of shape (3,). Normal Vector of the given trimesh
    """
    p0p1 = p1 - p0
    p0p2 = p2 - p0
    n = np.cross(p0p1, p0p2)
    l = np.linalg.norm(n)
    if l != 0.0:
        normal = n / l
    else:
        normal = n
    return normal


class Point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


@torch.jit.script
def coordinate_rotation(axis: int, angle):
    s = torch.sin(angle)
    c = torch.cos(angle)

    R = torch.eye(3, dtype=torch.float, device=angle.device)
    R = R.reshape((1, 3, 3)).repeat(angle.size(0), 1, 1)

    if axis == 0:
        R[:, 1, 1] = c
        R[:, 2, 2] = c
        R[:, 1, 2] = s
        R[:, 2, 1] = -s
    elif axis == 1:
        R[:, 0, 0] = c
        R[:, 0, 2] = -s
        R[:, 2, 0] = s
        R[:, 2, 2] = c
    elif axis == 2:
        R[:, 0, 0] = c
        R[:, 0, 1] = s
        R[:, 1, 0] = -s
        R[:, 1, 1] = c

    return R


@torch.jit.script
def euler_xyz_to_rotation_matrix(rpy):
    N = rpy.size(0)
    R = torch.tensor((N, 3, 3), dtype=torch.float, device=rpy.device)
    R = torch.matmul(
        torch.matmul(coordinate_rotation(0, rpy[:, 0]),
                     coordinate_rotation(1, rpy[:, 1])),
        coordinate_rotation(2, rpy[:, 2]))
    return R.transpose(1, 2)


def get_contact_normals(robots_feet_pos, mesh_normals, contact_forces):
    pos_to_idx = robots_feet_pos[:, :, :2].to(torch.int)

    pos_to_idx = torch.permute(pos_to_idx, (2, 0, 1))
    pos_to_idx = torch.flatten(pos_to_idx, 1)
    contact_normals = mesh_normals[pos_to_idx.tolist()]
    contact_normals = contact_normals.view(-1, 4, 3)

    filter = torch.norm(contact_forces[:, :, :], dim=2) > 1.
    filter = torch.repeat_interleave(filter, repeats=3,
                                     dim=1).view(-1, robots_feet_pos.shape[1],
                                                 3)
    contact_normals *= filter

    return contact_normals.flatten(1)


@torch.jit.script
def vector_2_skewmat(v):
    N = v.size(0)
    a = torch.zeros((N, 3, 3), device=v.device, dtype=torch.float)
    a[:, 2, 1] = v[:, 0]
    a[:, 2, 0] = -v[:, 1]
    a[:, 1, 0] = v[:, 2]
    a -= a.clone().transpose(1, 2)
    return a


def parse_sim_params(args, cfg):
    if 'real' in args.task:
        return cfg
    from isaacgym import gymapi, gymutil
    # initialize sim params
    sim_params = gymapi.SimParams()
    if 'flex' in args.physics_engine:
        args.physics_engine = gymapi.SIM_FLEX
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    else:
        args.physics_engine = gymapi.SIM_PHYSX
        sim_params.physx.use_gpu = True if 'cuda' in args.sim_device else False
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = True if 'cuda' in args.rl_device else False
    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params
