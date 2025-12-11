import open3d as o3d
import time
import glob

def inspect_pcd_sequence(pcd_files, fps=5):
    stop = {"flag": False}   # mutable container for closure

    def esc_callback(vis):
        stop["flag"] = True
        return False

    # Load first cloud
    pcd0 = o3d.io.read_point_cloud(pcd_files[0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("PCD Viewer")
    vis.register_key_callback(256, esc_callback)

    vis.add_geometry(pcd0)
    dt = 1.0 / fps

    for f in pcd_files:
        if stop["flag"]:
            break

        pcd = o3d.io.read_point_cloud(f)

        pcd0.points = pcd.points
        if pcd.has_colors():
            pcd0.colors = pcd.colors
        if pcd.has_normals():
            pcd0.normals = pcd.normals

        vis.update_geometry(pcd0)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(dt)

        if stop["flag"]:
            break

    vis.destroy_window()
