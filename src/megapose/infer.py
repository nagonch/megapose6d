from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import (
    ObservationTensor,
)
import numpy as np
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from dataloader import YCBV_LF
from tqdm import tqdm
import os


def bounding_box_from_mask(binary_mask: np.ndarray):
    object_rows, object_cols = np.where(binary_mask)
    if object_rows.size == 0:
        return None
    ymin = object_rows.min()
    ymax = object_rows.max() + 1
    xmin = object_cols.min()
    xmax = object_cols.max() + 1

    return np.array([xmin, ymin, xmax, ymax])


def make_object_dataset(example_dir):
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def get_poses(dataset):
    poses = []
    model_name = "megapose-1.0-RGBD"
    model_info = NAMED_MODELS[model_name]
    rigid_object_dataset = RigidObjectDataset(
        [RigidObject(label=dataset.model_name, mesh_path=dataset.model_path, mesh_units="mm")]
    )
    pose_estimator = load_named_model(model_name, rigid_object_dataset).cuda()
    for sample in tqdm(dataset):
        rgb = sample["rgb"]
        depth = sample["depth"]
        bounding_box = bounding_box_from_mask(sample["mask"])
        observation = ObservationTensor.from_numpy(rgb, depth, dataset.camera_intrinsics).cuda()
        detections = make_detections_from_object_data(
            [ObjectData(label=dataset.model_name, bbox_modal=bounding_box)]
        ).cuda()
        output, _ = pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **model_info["inference_parameters"],
            cuda_timer=True,
        )
        pose = output.poses.cpu().numpy()[0]
        poses.append(pose)

    return np.stack(poses)


if __name__ == "__main__":
    data_path = "/home/ngoncharov/cvpr2026/megapose6d/datasets/ycbv_lf/bleach_hard_00_03_chaitanya"
    dataset = YCBV_LF(data_path)
    results_path = "results_megapose6d/" + data_path.split("/")[-2]
    print(results_path)
    os.makedirs(results_path, exist_ok=True)
    poses = get_poses(dataset)
    print(poses)
    np.save(f"{results_path}/{data_path.split('/')[-1]}.npy", poses)
