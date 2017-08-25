# Cavelab
Deep Computer vision toolkit for rapid prototyping and scale deployment using Tensorflow. Specifically developed for Peta-scale alignment, but highly generalizable.

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
V Import visualization utils
V Import Tensorflow helper layers
- Import models (Graph Development)
- Import hparams
- Import blind model loading
- Import Generation of Training data
- Import Inference techniques
- Install OpenCV
```

Multi-server
```
- Notion of tasks
- Experiment tasks
- Inference tasks
- Task Queue
- Running N instances
```
