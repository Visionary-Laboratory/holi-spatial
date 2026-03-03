import torch
import numpy as np
import trimesh
import nvdiffrast.torch as dr

import math

# ---------- Camera helpers ----------
def opengl_proj_from_intrinsics(fx, fy, cx, cy, w, h, znear, zfar, device):
    proj = torch.zeros((4, 4), dtype=torch.float32, device=device)
    proj[0, 0] =  2.0 * fx / w
    proj[1, 1] =  2.0 * fy / h
    proj[0, 2] =  1.0 - 2.0 * (cx / w)
    proj[1, 2] =  2.0 * (cy / h) - 1.0
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    proj[3, 2] = -1.0
    return proj


def world_to_view_rt(R, t, device):
    M = torch.eye(4, dtype=torch.float32, device=device)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def compose_mvp(proj, view, model=None):
    if model is None:
        model = torch.eye(4, dtype=proj.dtype, device=proj.device)
    return proj @ view @ model


def load_mesh_ply_to_torch(ply_path, device="cuda:0"):
    m = trimesh.load(ply_path, process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump().sum()
    verts = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(m.faces,    dtype=np.int32)).to(device)
    return verts.contiguous(), faces.contiguous()


def quat_to_R(q, device):
    # q = (qw, qx, qy, qz)
    qw, qx, qy, qz = q
    R = torch.empty((3,3), dtype=torch.float32, device=device)
    R[0,0] = 1 - 2*(qy*qy + qz*qz); R[0,1] = 2*(qx*qy - qz*qw);   R[0,2] = 2*(qx*qz + qy*qw)
    R[1,0] = 2*(qx*qy + qz*qw);     R[1,1] = 1 - 2*(qx*qx + qz*qz); R[1,2] = 2*(qy*qz - qx*qw)
    R[2,0] = 2*(qx*qz - qy*qw);     R[2,1] = 2*(qy*qz + qx*qw);     R[2,2] = 1 - 2*(qx*qx + qy*qy)
    return R

def rt_to_view(R, t, device):
    # world -> camera（视图矩阵）
    M = torch.eye(4, dtype=torch.float32, device=device)
    M[:3, :3] = R
    M[:3, 3]  = t
    return M

def c2w_to_view(c2w):
    # camera->world 转 world->camera
    w2c = torch.linalg.inv(c2w)
    return w2c


class DepthRenderer:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

        # === CHECK: 绑定目标设备并创建上下文 ===
        torch.cuda.set_device(self.device.index or 0)
        # print(f"[info] visible device count: {torch.cuda.device_count()}")
        # print(f"[info] current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")

        try:
            self.glctx = dr.RasterizeGLContext(
            output_db=False,     # 不需要 image-space 导数就关掉
            mode='automatic',    # 默认即可；也可用 'manual' 自己 set/release
            device=self.device   # 放在同一块 GPU 上
        )
        except TypeError:
            print("[error] nvdiffrast.torch.RasterizeGLContext failed. Ensure you have the correct version of nvdiffrast installed.")




    @torch.inference_mode()
    def render_depth_batched(
        self, mesh, camera_parameters,
        flip_y=True, max_tris_per_pass=1_000_000_0, verbose=True, linear_depth = False
    ):
        with torch.no_grad():
            """对超大三角网（例如 1000 万面）进行分批渲染并合并深度。
            - max_tris_per_pass: 每批最多投多少个三角形（可按显存调整 2e5
            """


            H, W = camera_parameters["H"],camera_parameters["W"]
            znear, zfar = camera_parameters["near"],camera_parameters["far"]
            mvp = camera_parameters["mvp"]
            faces_world = mesh["faces"]
            verts_world = mesh["verts"]
            dev = self.device
            V = verts_world.shape[0]

            # 预计算 clip 坐标（1, V, 4），一次即可在各批复用
            ones = torch.ones((V, 1), dtype=torch.float32, device=dev)
            verts_h = torch.cat([verts_world.to(torch.float32), ones], dim=1).contiguous()
            pos_clip = (mvp.to(torch.float32) @ verts_h.T).T.contiguous()[None, ...]  # (1,V,4)

            # 全局深度初始化为 +inf，mask 初始化为 False
            depth_global = torch.full((H, W), float("inf"), dtype=torch.float32, device=dev)
            mask_global  = torch.zeros((H, W), dtype=torch.bool, device=dev)

            n = torch.tensor(znear, device=dev, dtype=torch.float32)
            f = torch.tensor(zfar,  device=dev, dtype=torch.float32)

            F = faces_world.shape[0]
            num_chunks = math.ceil(F / max_tris_per_pass)

            for ci in range(num_chunks):
                s = ci * max_tris_per_pass
                e = min(F, (ci + 1) * max_tris_per_pass)
                tri_chunk = faces_world[s:e]  # (K,3) int32 CUDA

                # 光栅化当前批
                rast, _ = dr.rasterize(self.glctx, pos_clip, tri_chunk, (H, W),ranges=None, grad_db=False )
                tri_id  = rast[0, :, :, 3]
                hit     = tri_id > 0
                if not hit.any():
                    if verbose: print(f"[chunk {ci+1}/{num_chunks}] no hits")
                    continue

                z_over_w = rast[0, :, :, 2].clamp(-1.0, 1.0)
                if linear_depth:
                    # NDC -> 线性视空间深度（正距离）
                    z_over_w = (2.0 * n * f) / (f + n - z_over_w * (f - n))
                    depth_lin = (z_over_w)  # 正
                else:
                    depth_lin = (z_over_w)  # 正

                # 将未命中的像素置为 +inf，便于做逐像素最小
                depth_lin = torch.where(hit, depth_lin, torch.full_like(depth_lin, float("inf")))

                # 合并到全局 z-buffer：取更近的深度
                depth_global = torch.minimum(depth_global, depth_lin)
                mask_global  = mask_global | hit

                # 可选：清理临时张量以减小峰值显存
                del rast, tri_id, hit, z_over_w, depth_lin
                torch.cuda.synchronize()

            # 对没有命中的像素，depth 仍是 inf；保持 mask 返回即可
            if flip_y:
                depth_global = torch.flip(depth_global, dims=[0])
                mask_global  = torch.flip(mask_global,  dims=[0])

            return depth_global.contiguous(), mask_global

# ---------- Simple loader ----------



def Build_Ply_Render_Camera_Parameters_default(device):
    H, W = 720, 1280
    znear, zfar = 0.01, 100.0
    fx, fy = 1000.0, 1000.0
    cx, cy = W / 2.0, H / 2.0
    proj = opengl_proj_from_intrinsics(fx, fy, cx, cy, W, H, znear, zfar, device=torch.device(device))

    R = torch.eye(3, dtype=torch.float32, device=device)
    t = torch.tensor([0.0, 0.0, -5.0], dtype=torch.float32, device=device)
    view = world_to_view_rt(R, t, device=torch.device(device))
    mvp = compose_mvp(proj, view)
    return {"mvp":mvp,"far":zfar,"near":znear,"H":H,"W":W}

def Build_Ply_Render_Camera_Parameters_colmap(fx, fy, cx, cy, W, H, znear, zfar,R,T, device):
    # H, W = 720, 1280
    # znear, zfar = 0.01, 100.0
    # fx, fy = 1000.0, 1000.0
    # cx, cy = W / 2.0, H / 2.0
    proj = opengl_proj_from_intrinsics(fx, fy, cx, cy, W, H, znear, zfar, device=torch.device(device))

    R = R
    t = T
    view = world_to_view_rt(R, t, device=torch.device(device))
    mvp = compose_mvp(proj, view)
    return {"mvp":mvp,"far":zfar,"near":znear,"H":H,"W":W}


# def Build_Ply_Render_Camera_Parameters_colmap(
#     fx, fy, cx, cy, W, H,
#     qvec=None, tvec=None, Rcw=None, tcw=None,
#     znear=0.01, zfar=100.0, device="cuda:0"
# ):
#     device = torch.device(device)
#     # 允许传四元数(qvec,tvec) 或 直接传 Rcw,tcw（world->camera）
#     if Rcw is None:
#         assert qvec is not None and tvec is not None
#         if not torch.is_tensor(qvec):
#             qvec = torch.tensor(qvec, dtype=torch.float32, device=device)
#         if not torch.is_tensor(tvec):
#             tvec = torch.tensor(tvec, dtype=torch.float32, device=device)
#         Rcw = quat_to_R(qvec, device)
#         tcw = tvec.to(dtype=torch.float32, device=device)
#     else:
#         Rcw = Rcw.to(dtype=torch.float32, device=device)
#         tcw = tcw.to(dtype=torch.float32, device=device)

#     proj = opengl_proj_from_intrinsics(fx, fy, cx, cy, W, H, znear, zfar, device)
#     view = rt_to_view(Rcw, tcw, device)
#     mvp  = compose_mvp(proj, view)
#     return {"mvp": mvp, "far": zfar, "near": znear, "H": H, "W": W}


def Load_ply_resource(ply_path, device="cuda:0"):
    verts, faces = load_mesh_ply_to_torch(ply_path, device=device)
    # === CHECK: 基本一致性 ===
    assert verts.is_cuda and faces.is_cuda
    assert faces.dtype == torch.int32 and verts.dtype == torch.float32
    V = verts.shape[0]
    assert faces.min() >= 0 and faces.max() < V
    faces = faces.to(torch.int32).contiguous().to(verts.device)
    return {"verts":verts,"faces":faces}


import torch

def Build_Ply_Render_Camera_Parameters_colmap_correct(
    fx, fy, cx, cy, W, H, znear, zfar, R, T, device
):
    device = torch.device(device)

    # 1) 投影矩阵（保持你原来的）
    proj = opengl_proj_from_intrinsics(fx, fy, cx, cy, W, H, znear, zfar, device=device)

    # 2) 直接使用 COLMAP 的 world->camera 外参：x_cam = R x_world + T
    R = torch.as_tensor(R, dtype=torch.float32, device=device)        # (3,3)
    t = torch.as_tensor(T, dtype=torch.float32, device=device).view(3, 1)  # (3,1)

    # 3) OpenCV/Colmap -> OpenGL（y 取反、z 取反）
    S = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))    # (3,3)
    R_gl = S @ R
    t_gl = S @ t

    # 4) 组装 4x4 view 矩阵（world->camera，在 OpenGL 相机坐标系下）
    view = torch.eye(4, dtype=torch.float32, device=device)
    view[:3, :3] = R_gl
    view[:3, 3:4] = t_gl

    # 5) mvp
    mvp = compose_mvp(proj, view)

    return {"mvp": mvp, "far": zfar, "near": znear, "H": H, "W": W}



def show_depth_preview(depth_m, mask, path, invert=True, q=(0.02, 0.98)):
    import torch, numpy as np
    import matplotlib.pyplot as plt
    """把有效深度按分位数拉伸到[0,1]并预览。"""
    if not mask.any():
        print("No visible pixels.")
        return
    valid = depth_m[mask]

    # 选一个稳定的显示区间（去掉极端值）
    lo, hi = torch.quantile(valid, torch.tensor([q[0], q[1]], device=valid.device))
    dep = depth_m.clamp(min=float(lo), max=float(hi))
    # dep = depth_m
    if invert:                     # 近处亮、远处暗（更符合人眼）
        dep = hi - (dep - lo)      # 也可以用 1/(dep+eps)，但这个更线性

    # 归一化到[0,1]，无效处设为0（黑）
    dep01 = (dep - dep[mask].min()) / (dep[mask].max() - dep[mask].min() + 1e-12)
    dep01 = torch.where(mask, dep01, torch.zeros_like(dep01))

    plt.figure(figsize=(17,15))
    plt.imshow(dep01.detach().cpu().numpy(), cmap="gray")  # 或 "gray"
    plt.title("Depth preview (normalized)")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(path)
    # plt.show()