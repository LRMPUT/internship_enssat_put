# LiSAM

## What is LiSAM?

LiSAM is a tool to extract segmentations from multiple point clouds (.las/.laz) using SegmentAnything model.

## How to use LiSAM?

Install the required conda environment using the following command:

```bash
conda env create -f SegmentAnything/conda_env.yml
```

Activate the conda environment:

```bash
conda activate sam-lidar
```

Run the following command to extract segmentations from multiple point clouds:

```bash
python LiSAM/main.py <pointclouds_folder> <result_folder> --model_path <path_to_model>
```
