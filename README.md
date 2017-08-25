# Cavelab
AI &amp; Computer Vision laboratory for rapid prototyping and scale deployment. Currently it will specialize in EM image alignment problem but will have generalizable features.

## Setup
```
./bin/install
```

## Usage
```
import cavelab
```
Should work without GPU tensorflow dependancies


## Docker
Assume you have nvidia-docker
```
./bin/build
./bin/run
```
Will add without nvidia-docker support


## Plans

Todo

```
V Import all h5 data processings into cavelab
V Import data reading from tfrecords
V Global Sess
V Seperate image processing modules (numpy and tensorflow)
- Import data collection module
- Import visualization utils
- Import Tensorflow base layers
- Import Graph development
- Import Inference techniques
- Import Data collection techniques
- Install OpenCV
```
