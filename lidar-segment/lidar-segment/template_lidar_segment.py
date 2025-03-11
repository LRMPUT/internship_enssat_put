
from segment_lidar import samlidar, view

viewpoint = view.TopView()

model = samlidar.SamLidar(ckpt_path="sam_vit_b_01ec64.pth")
points = model.read("../lazfiles/pointcloud.las")
labels, *_ = model.segment(points=points, view=viewpoint, labels_path="/home/savinien/Documents/Polonie/lidar-segment/labeled.tif")
model.write(points=points, segment_ids=labels, save_path="segmented.las")