import open3d as o3d
input_file="../scan_PUT/block_0_pcl.pcd"
pcd = o3d.io.read_point_cloud(input_file)
voxel_down_pcd = pcd.voxel_down_sample(pcd, voxel_size=0.02)