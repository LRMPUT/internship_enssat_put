## Requirement

- Make sure `python 3.10` is intalled on your computer. Create a virtual environnement is necessary.

```
virtualenv --python="/usr/bin/python3.10" "venv/"
```

Before installing package you can install torch with CPU only using :

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

```

- Install the necessary packages

```
pip install -r requirement.txt
```

### Getting started

- Import a `.las` or `.laz` file on this current directory, named: `segmented.las`

### Files explanations

- `fcn_test.ipynb`: A test of the model `fcn_resnet50` used to detect contour in a image.

- `template_lidar_segment.py`: Code to use the [**segement-lidar**](https://github.com/Yarroudh/segment-lidar) github repository.

- `tif_interpret.ipynb`: notebook to understand how .tif files are processed

- `main.ipynb`: Main code for analyse how segment-lidar evolve when points cloud are reducing
