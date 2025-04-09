import argparse, logging
from lisam import run

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
    parser.add_argument("--confirm", action='store_true', default=False)

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
    confirm: bool = bool(args.confirm)

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
        default_answer = "y" if confirm else "n" if no_confirm else ""
    )

    logging.debug(f"Masks count by PCL file")
    for k, v in count.items():
        logging.debug(f"|> {k} => {v} masks found")
