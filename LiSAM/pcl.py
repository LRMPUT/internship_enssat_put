import open3d as o3d
import numpy as np
import laspy

def las_to_open3d(las_file: str):
    las_pcl = laspy.read(las_file)
    pcl = o3d.t.geometry.PointCloud()

    points = np.vstack((las_pcl.x, las_pcl.y, las_pcl.z)).T
    pcl.point.positions = o3d.core.Tensor(points)

    points = np.vstack((las_pcl.red / 255.0, las_pcl.green / 255.0, las_pcl.blue / 255.0)).T
    pcl.point.colors = o3d.core.Tensor(points)

    return pcl

def open3d_to_numpy(pcl):
    return np.hstack((pcl.point.positions.numpy(), pcl.point.colors.numpy()))

def downsample(pcl, parameter: float, method: str = "voxel"):
    if method == "voxel":
        # Voxel downsampling
        return pcl.voxel_down_sample(voxel_size=parameter)
    elif method == "uniform":
        # Uniform downsampling
        return pcl.uniform_down_sample(int(parameter))
    elif method == "random":
        # Random downsampling
        return pcl.random_down_sample(parameter)
    else:
        raise Exception(f"downsampling method {method} unknown")
