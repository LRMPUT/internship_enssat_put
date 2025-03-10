import os, logging, argparse, queue
import numpy as np
from LiSAM.pcl import downsample, las_to_open3d, open3d_to_numpy

def run(
        pointclouds_folder: str,
        result_path: str,
        result_folder_name: str,
        model_path: str,
        resolution: float = 0.25,
        subsampling: float = 0,
        subsampling_method: str = "voxel",
        model_type: str = None,
        no_confirm: bool = True
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
    if len(os.listdir(os.path.join(result_path, result_folder_name))):
        logging.warning(f"files are already presents in {os.path.join(result_path, result_folder_name)} folder, they will be deleted are you sure you want to continue [Y/N] ?")
        if no_confirm:
            logging.warning("(--no-confirm) enabled, skip the question")
            answer = "y"
        else:
            answer = input("")
        if answer.lower() == "y":
            files = os.listdir(os.path.join(result_path, result_folder_name))
            for file in files:
                os.unlink(os.path.join(result_path, result_folder_name, file))
        else:
            exit(1)

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
        logging.info(f"processing {pointclouds_file}...")
        points = las_to_open3d(pointclouds_file)

        if subsampling > 0:
            logging.info(f"downsampling to {subsampling} (method: {subsampling_method})")
            try:
                points = downsample(points, subsampling, method=subsampling_method)
            except Exception as err:
                logging.error(f"Execption raised when downsampling: {err}")
                exit(1)

        _, segmented_image, _ = model.segment(
            points=open3d_to_numpy(points),
            view=viewpoint,
            image_path=os.path.join(result_path, result_folder_name, f"raster_{pointclouds_file.replace('/', '_')}.tif"),
            labels_path=os.path.join(result_path, result_folder_name, f"label_{pointclouds_file.replace('/', '_')}.tif")
        )
        count_mask_by_pcl.update({pointclouds_file: np.max(segmented_image)})

    logging.info(f"All files have been processed in {os.path.join(result_path, result_folder_name)}")

    return count_mask_by_pcl

if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog='LiSAM',
        description='Generate segmentation from points clouds using SegmentAnything'
    )

    # Mandatory arguments
    parser.add_argument("pointclouds_folder")
    parser.add_argument("result_folder")

    # CLI options
    parser.add_argument("--debug", "--log-level", default="info")
    parser.add_argument("--result-name", "--name", "-n", default="results")
    parser.add_argument("--no-confirm", action='store_true', default=False)

    # Model parameters
    parser.add_argument("--model-path", "--model", required=False)
    parser.add_argument("--model-type", required=False)

    # Preprocess parameters
    parser.add_argument("--resolution", default=0.25)
    parser.add_argument("--subsampling", default=-1.0)
    parser.add_argument("--subsampling-method", default="voxel")

    args = parser.parse_args()

    pointclouds_folder: str = args.pointclouds_folder
    result_path: str = args.result_folder

    result_folder_name: str = args.result_name
    no_confirm: bool = bool(args.no_confirm)

    model_path: str = args.model_path
    model_type: str = args.model_type

    resolution: float = float(args.resolution)
    subsampling: float = float(args.subsampling)
    subsampling_method: str = args.subsampling_method

    count = run(
        pointclouds_folder = pointclouds_folder,
        result_path = result_path,
        result_folder_name = result_folder_name,
        model_path = model_path,
        resolution = resolution,
        subsampling = subsampling,
        subsampling_method = subsampling_method,
        model_type = model_type,
        no_confirm = no_confirm
    )

    logging.info(f"Masks count by PCL file")
    for k, v in count.items():
        logging.info(f"|> {k} => {v} masks found")
