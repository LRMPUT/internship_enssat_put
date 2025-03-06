import os, logging, argparse

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(
    prog='LiSAM',
    description='Generate segmentation from points clouds using SegmentAnything'
)
parser.add_argument("pointclouds_folder")
parser.add_argument("result_folder")
parser.add_argument("--debug", "--log-level", default="debug")

args = parser.parse_args()
result_path = args.result_folder

# Create masks folder
if not os.path.exists(os.path.join(result_path, "results")):
    logging.info("creating folders...")
    os.mkdir(os.path.join(result_path, "results"))
else:
    logging.info("folder already exists")

# 