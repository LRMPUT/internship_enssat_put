# SegmentAnything

This project contains Jupter notebooks to segmente .las file using **Segment Lidar** library, classify the segments between **building** and **vegetation**, segment road, and compare segmented road with a ground truth.

## Installation

To install the required packages, run the following command:

```bash
conda env create -f conda_env.yml
```

### Patch for Segment Lidar

The Segment Lidar library has performance issue on some methods. To fix this, replace these functions in the `TopView` class in `~/.conda/envs/sam-lidar/lib/python3.9/site-packages/segment_lidar/view.py` file with the following code:

```python
def image_to_cloud(self, points: np.ndarray, image: np.ndarray, resolution: float) -> np.ndarray:
        """
        Converts an image to a point cloud.

        :param points: An array of points in the cloud, where each row represents a point.
                    The array shape can be (N, 3) or (N, 6).
                    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                    If the shape is (N, 6), the last three columns represent the RGB color values for each point.
        :type points: ndarray
        :param image: An image array representing the point cloud, where each pixel contains the RGB color values of the corresponding point in the cloud.
        :type image: ndarray
        :param resolution: The resolution of the image in units per pixel.
        :type resolution: float
        :return: An array of segments' IDs in the cloud, where each row represents the segment's ID of a point.
        :rtype: ndarray
        :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
        """
        segment_ids = []
        unique_values = {}
        minx, maxy = np.min(points[:, 0]), np.max(points[:, 1])

        # Calculate pixel coordinates for all points
        x = (points[:, 0] - minx) / resolution
        y = (maxy - points[:, 1]) / resolution
        pixel_x = np.floor(x).astype(int)
        pixel_y = np.floor(y).astype(int)

        # Mask points outside image bounds
        out_of_bounds = (pixel_x < 0) | (pixel_x >= image.shape[1]) | (pixel_y < 0) | (pixel_y >= image.shape[0])
        segment_ids.extend([-1] * np.sum(out_of_bounds))

        valid_points = ~out_of_bounds
        rgb = image[pixel_y[valid_points], pixel_x[valid_points]]

        # Map RGB values to unique segment IDs
        for rgb_val in rgb:
            if rgb_val not in unique_values:
                unique_values[rgb_val] = len(unique_values)

        segment_ids.append(unique_values[rgb_val])

        return segment_ids
```

```python
def cloud_to_image(self, points: np.ndarray, resolution: float) -> np.ndarray:
        """
        Converts a point cloud to a planar image.

        :param points: An array of points in the cloud, where each row represents a point.
                    The array shape can be (N, 3) or (N, 6).
                    If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                    If the shape is (N, 6), the last three columns represent the RGB color values for each point.
        :type points: ndarray
        :param minx: The minimum x-coordinate value of the cloud bounding box.
        :type minx: float
        :param maxx: The maximum x-coordinate value of the cloud bounding box.
        :type maxx: float
        :param miny: The minimum y-coordinate value of the cloud bounding box.
        :type miny: float
        :param maxy: The maximum y-coordinate value of the cloud bounding box.
        :type maxy: float
        :param resolution: The resolution of the image in units per pixel.
        :type resolution: float
        :return: An image array representing the point cloud, where each pixel contains the RGB color values
                of the corresponding point in the cloud.
        :rtype: ndarray
        :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
        """
        minx, maxx = np.min(points[:, 0]), np.max(points[:, 0])
        miny, maxy = np.min(points[:, 1]), np.max(points[:, 1])

        if points.shape[1] == 3:
            colors = np.array([255, 255, 255])
        else:
            colors = points[:, -3:]

        x = (points[:, 0] - minx) / resolution
        y = (maxy - points[:, 1]) / resolution
        pixel_x = np.floor(x).astype(int)
        pixel_y = np.floor(y).astype(int)

        width = int((maxx - minx) / resolution) + 1
        height = int((maxy - miny) / resolution) + 1

        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[pixel_y, pixel_x] = colors

        return image
```

The Segment Anything library has some default properties to generate the segments. To change these properties, replace the `SamAutomaticMaskGenerator` constructor parameters in `~/.conda/envs/sam-lidar/lib/python3.9/site-packages/segment_anything/automatic_mask_generator.py` file with the following code:

```python
def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 80,
        points_per_batch: int = 8,
        pred_iou_thresh: float = 0.70,
        stability_score_thresh: float = 0.90,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.9,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    )
```

## Usage

| Notebook                 | Description                                                          |
| ------------------------ | -------------------------------------------------------------------- |
| segment.ipynb            | Segment .las file using Segment Lidar library, classify and evaluate |
| classifier.ipynb         | Classify segments between building and vegetation using .skops model |
| cnn_classification.ipynb | Classify segments using a CNN                                        |
