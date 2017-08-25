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

Minimal Todo

```
V Import all h5 data processings into cavelab
V Import data reading from tfrecords
V Global Sess
V Seperate image processing modules (numpy and tensorflow)
V Import visualization utils
V Import Tensorflow helper layers
V Install OpenCV
- Import models (Graph Development)
- Saving/Loading model
- Hyperparameters
- Import Generation of Training data
- Import Inference techniques

```

Scaling Todo
```
- Hyperparameters
- Notion of tasks and sub-tasks
- Experiment tasks
- Async sub-tasking
- Inference tasks
- Task Queue
- Running N instances
- Experiment logging and hypothesis testing
```

Nice to have
```
- Web-Interface
- Documentation
- Usage in Notebooks
```
