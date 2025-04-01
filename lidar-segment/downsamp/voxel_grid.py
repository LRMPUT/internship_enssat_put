import open3d as o3d
import numpy as np

def nb_points(pcd):
    return len(pcd.points)

def voxel_downsample(input_file):
    ply_point_cloud = input_file

    pcd = o3d.io.read_point_cloud(filename=ply_point_cloud)

    voxel = 0.03
    downsample = pcd.voxel_down_sample(voxel)
    print("before : ",pcd, ", after : ",downsample)
    return downsample

def downsample(input_file, reduc_percent):
    print("start downsampling")
    ply_point_cloud = input_file

    pcd = o3d.io.read_point_cloud(filename=ply_point_cloud)
    print(nb_points(pcd))
    print(int(round((100-reduc_percent)*nb_points(pcd))/100))
    downsample = pcd.farthest_point_down_sample(int(round((100-reduc_percent)*nb_points(pcd))/100))
    print("before : ",pcd, ", after : ",downsample)
    return downsample
'''
If your pointcloud is expressed in meters, voxel_size=0.02 will mean the voxel grid will be made up of voxels measuring 2 x 2 x 2 cm. 
After filtering the pointcloud density is reducted so that you have only one point per such voxel. 
'''
down_file = downsample("table_scene_lms400.pcd", 50)
print(down_file)
o3d.io.write_point_cloud("./sampled.pcd", down_file)
down_file = downsample("./sampled.pcd", 50)
print(down_file)
#o3d.visualization.draw_geometries([down_file])