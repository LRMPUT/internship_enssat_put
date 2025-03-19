import os, logging, queue, laspy
import PIL.Image as Image
import open3d as o3d
import numpy as np

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

def run(
        pointclouds_folder: str,
        result_path: str,
        result_folder_name: str,
        model_path: str,
        resolution: float = 0.25,
        subsampling: float = 0,
        subsampling_method: str = "voxel",
        model_type: str = None,
        no_confirm: bool = True,
        confirm: bool = False,
    ) -> dict:
    # Check if model type is correct
    if model_type and model_type not in ["vit_b", "vit_h", "vit_l"]:
        logging.error("model type is not valid, should be vit_b, vit_h or vit_l")
        exit(1)

    # Check if the path to the model is correct
    if model_path and not os.path.exists(model_path):
        logging.error(f"the path to the model {model_path} does not exists")
        exit(1)
    
    # If model type not specified find it with the name of the file
    if model_path and not model_type:
        logging.debug("the type of the model is not precised, trying to find it")
        if "vit_b" in model_path:
            model_type = "vit_b"
        elif "vit_h" in model_path:
            model_type = "vit_h"
        elif "vit_l" in model_path:
            model_type = "vit_l"
        
        if model_type:
            logging.info(f"model type found is {model_type}")
        else:
            logging.error("model type has not been found, please add the argument --model-type with the corresponding model (vit_h, vit_b, vit_l)")
            exit(1)
        
    # Check if subsampling method is correct
    if subsampling_method and subsampling_method not in ["random", "voxel", "uniform"]:
        logging.error("subsampling method is not valid, should be random, voxel or uniform")
        exit(1)

    # Create masks folder
    if not os.path.exists(os.path.join(result_path, result_folder_name)):
        logging.info(f"creating folder {result_folder_name}...")
        os.mkdir(os.path.join(result_path, result_folder_name))
    else:
        logging.info(f"folder {result_folder_name} already exists")

    # Check if results already presents in the folder and remove contents
    skip_computing = False
    if len(os.listdir(os.path.join(result_path, result_folder_name))):
        logging.warning(f"files are already presents in {os.path.join(result_path, result_folder_name)} folder, they will be deleted are you sure you want to continue [Y/N] ?")
        if no_confirm:
            logging.warning("(--no-confirm) enabled, say no and skip the question")
            answer = "n"
        elif confirm:
            logging.warning("(--confirm) enabled, say yes and skip the question")
            answer = "y"
        else:
            answer = input("")
        if answer.lower() == "y":
            files = os.listdir(os.path.join(result_path, result_folder_name))
            for file in files:
                os.unlink(os.path.join(result_path, result_folder_name, file))
        else:
            skip_computing = True

    # Create a queue of files
    pointclouds_files = os.listdir(pointclouds_folder)
    logging.info(f"find {len(pointclouds_files)} pointclouds files")
    file_queue = queue.Queue(maxsize=len(pointclouds_files))
    logging.debug(f"file_queue created with size of {len(pointclouds_files)}")
    for file in pointclouds_files:
        file_queue.put(os.path.join(pointclouds_folder, file))
        logging.debug(f"file_queue enqueued {file}")

    # Load the model
    from segment_lidar import samlidar, view

    viewpoint = view.TopView()
    model = samlidar.SamLidar(
        ckpt_path=model_path,
        model_type=model_type,
        resolution=resolution,
    )

    # Processing the queue
    count_mask_by_pcl = dict()
    while not file_queue.empty():
        pointclouds_file: str = file_queue.get()
        segmented_image, rgb_image = None, None

        if not skip_computing:
            logging.info(f"processing {pointclouds_file}...")
            points = las_to_open3d(pointclouds_file)

            if subsampling > 0:
                logging.info(f"downsampling to {subsampling} (method: {subsampling_method})")
                try:
                    points = downsample(points, subsampling, method=subsampling_method)
                except Exception as err:
                    logging.error(f"Exception raised when downsampling: {err}")
                    exit(1)

            _, segmented_image, rgb_image = model.segment(
                points=open3d_to_numpy(points),
                view=viewpoint,
                image_path=os.path.join(result_path, result_folder_name, f"raster_{pointclouds_file.replace('/', '_')}.tif"),
                labels_path=os.path.join(result_path, result_folder_name, f"label_{pointclouds_file.replace('/', '_')}.tif")
            )
            rgb_image = rgb_image.transpose(1, 2, 0) # [channels, height, width] --> [height, width, channels]  (=🝦 ﻌ 🝦=)
        else:
            logging.info(f"skip processing of {pointclouds_file}")
            rgb_image = np.array(Image.open(os.path.join(result_path, result_folder_name, f"raster_{pointclouds_file.replace('/', '_')}.tif")))
            segmented_image = np.array(Image.open(os.path.join(result_path, result_folder_name, f"label_{pointclouds_file.replace('/', '_')}.tif")))

        count_mask_by_pcl.update({pointclouds_file: np.max(segmented_image)})

    logging.info(f"All files have been processed in {os.path.join(result_path, result_folder_name)}")

    return count_mask_by_pcl
