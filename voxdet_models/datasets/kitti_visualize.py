import os
import numpy as np
import open3d as o3d


def read_semantickitti_scan(bin_path, label_path=None):
    """
    Read one SemanticKITTI scan (.bin) and its labels (.label).

    bin_path: path to velodyne/*.bin
    label_path: path to labels/*.label (optional)
    """
    # points: [N, 4] = x, y, z, intensity
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    if label_path is not None and os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)

        # SemanticKITTI encoding:
        # lower 16 bits: semantic label
        # upper 16 bits: instance id
        semantic_labels = labels & 0xFFFF
        instance_labels = labels >> 16
    else:
        semantic_labels = None
        instance_labels = None

    return xyz, semantic_labels, instance_labels


def build_color_map(num_classes=256, seed=0):
    """
    Simple deterministic color map for semantic labels.
    For real use, replace with SemanticKITTI official color map.
    """
    rng = np.random.default_rng(seed)
    cmap = rng.random((num_classes, 3), dtype=np.float32)
    cmap[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # class 0 = black
    return cmap


def visualize_semantickitti_pointcloud(
    bin_path,
    label_path=None,
    use_instance_colors=False,
    max_points=None,
):
    """
    Visualize a single SemanticKITTI frame with Open3D.

    bin_path: path to velodyne/*.bin file
    label_path: path to labels/*.label file
    use_instance_colors: False = color by semantic, True = color by instance
    max_points: randomly subsample to this many points (for speed), or None
    """
    xyz, semantic_labels, instance_labels = read_semantickitti_scan(
        bin_path, label_path
    )

    if max_points is not None and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        if semantic_labels is not None:
            semantic_labels = semantic_labels[idx]
        if instance_labels is not None:
            instance_labels = instance_labels[idx]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    if semantic_labels is not None:
        if use_instance_colors and instance_labels is not None:
            labels_for_color = instance_labels
        else:
            labels_for_color = semantic_labels

        max_label = int(labels_for_color.max()) if labels_for_color.size > 0 else 0
        cmap = build_color_map(num_classes=max(256, max_label + 1))

        labels_for_color = labels_for_color.astype(np.int32)
        colors = cmap[labels_for_color]
        pc.colors = o3d.utility.Vector3dVector(colors)
    else:
        # gray if no labels
        gray = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.5
        pc.colors = o3d.utility.Vector3dVector(gray)

    o3d.visualization.draw_geometries([pc])


if __name__ == "__main__":
    # Example paths for sequence 00, frame 000000
    root = "data/kitti/dataset"  # change this
    seq = "00"
    frame = "000000"

    bin_path = os.path.join(root, "sequences", seq, "velodyne", frame + ".bin")
    label_path = os.path.join(root, "sequences", seq, "labels", frame + ".label")

    visualize_semantickitti_pointcloud(
        bin_path,
        label_path,
        use_instance_colors=False,  # set True to color by instance id
        max_points=100000,          # or None
    )







