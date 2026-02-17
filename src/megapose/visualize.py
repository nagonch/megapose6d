import viser
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from dataloader import YCBV_LF


def create_viser_server() -> viser.ViserServer:
    server = viser.ViserServer(verbose=False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    return server


def backproject_depth_to_pointcloud(depths, camera_matrix, return_scales=False):
    depths = torch.from_numpy(depths).double().cuda()
    camera_matrix = torch.from_numpy(camera_matrix).double().cuda()
    uu, vv = torch.meshgrid(
        (
            torch.arange(depths.shape[0], device=depths.device),
            torch.arange(depths.shape[1], device=depths.device),
        )
    )
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    pixel_indices = torch.stack((vv, uu), dim=0).T
    depths = depths.reshape(-1)
    inv_camera_matrix = torch.linalg.inv(camera_matrix).double()
    ones = torch.ones(
        (pixel_indices.shape[0], 1),
        device=pixel_indices.device,
        dtype=pixel_indices.dtype,
    ).double()
    uv1 = torch.cat([pixel_indices, ones], dim=1).T  # Shape: [3, N]
    xyz_camera = (inv_camera_matrix @ uv1) * depths
    xyz_camera = xyz_camera.T
    if return_scales:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        scales_x = xyz_camera[:, 2] / fx
        scales_y = xyz_camera[:, 2] / fy
        scales = torch.zeros(xyz_camera.shape[0], 3, device=xyz_camera.device)
        scales[:, 0] = scales_x
        scales[:, 1] = scales_y
        scales[:, 2] = (scales_x + scales_y) / 2
        return xyz_camera, scales
    else:
        return xyz_camera.cpu().numpy()


def run_viser_server(server: viser.ViserServer):
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        server.scene.reset


class Visualizer:
    def __init__(self):
        self.server = create_viser_server()
        self.scene = self.server.scene

    def run(self):
        run_viser_server(self.server)

    def add_point_cloud(self, name, points, colors=None, point_size=1e-4):
        if colors is None:
            colors = np.array([255, 0, 0])
        self.scene.add_point_cloud(name, points, colors=colors, point_size=point_size)

    def add_frame(self, name, frame_T, frames_scale=0.05):
        if not isinstance(frame_T, np.ndarray):
            frame_T = np.array(frame_T)
        position = frame_T[:3, 3]
        rotation = frame_T[:3, :3]
        xyzw = R.from_matrix(rotation).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        self.scene.add_frame(
            name=name,
            position=position,
            wxyz=wxyz,
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
        )

    def add_camera_frustum(self, name, camera_T, camera_matrix, image, scale=0.1):
        image_height, image_width = image.shape[:2]
        if not isinstance(camera_T, np.ndarray):
            camera_T = np.array(camera_T)
        position = camera_T[:3, 3]
        rotation = camera_T[:3, :3]
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        fov = np.arctan2(image_width / 2, camera_matrix[0, 0]) * 2
        self.scene.add_camera_frustum(
            name=name,
            aspect=image_width / image_height,
            fov=fov.item(),
            scale=scale,
            line_width=0.5,
            image=image,
            wxyz=wxyz,
            position=position,
        )

    def add_mesh(self, name, mesh, pose):
        vertices = mesh.verts_list()[0].cpu().numpy()
        faces = mesh.faces_list()[0].cpu().numpy()
        position = pose[:3, 3].cpu().numpy()
        rotation = pose[:3, :3].cpu().numpy()
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        self.scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            wxyz=wxyz,
            position=position,
        )


if __name__ == "__main__":
    dataset = YCBV_LF("/home/ngoncharov/cvpr2026/megapose6d/datasets/ycbv_lf/bleach0")
    poses = np.load("/home/ngoncharov/cvpr2026/megapose6d/results_megapose6d/ycbv_lf/bleach0.npy")
    visulizer = Visualizer()
    for i in np.linspace(0, len(dataset) - 1, 10).astype(int):
        sample = dataset[i]
        depth = sample["depth"]
        mask = sample["mask"].astype(bool)
        camera_matrix = dataset.camera_intrinsics
        pose_gt = sample["pose"]
        pose = poses[i]
        color = sample["rgb"].reshape(-1, 3)
        points_camera = backproject_depth_to_pointcloud(depth, camera_matrix)

        color = color[mask.reshape(-1)]
        points_camera = points_camera[mask.reshape(-1)]
        points_camera_np = points_camera
        visulizer.add_point_cloud(
            name=f"pointcloud_{i}", points=points_camera_np, colors=color, point_size=1e-3
        )
        visulizer.add_frame(
            name=f"frame_{i}",
            frame_T=pose,
        )
        visulizer.add_frame(
            name=f"frame_gt_{i}",
            frame_T=pose_gt,
        )
    visulizer.run()
    visualizer = Visualizer()
    visualizer.run()
