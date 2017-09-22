# Cavelab
Deep-vision environment for rapid prototyping and scale deployment. Includes libraries such as Tensorflow and OpenCV. Specifically developed for Peta-scale alignment on EM images, but highly generalizable.

## Usage
```
from cavelab import image_processing, visual

blend_map = image_processing.get_blend_map(256, 512)
visual.save(blend_map, 'blend.jpg')
```

## Build
Build dockerfile and then run the environment
```
./bin/build
./bin/run
```
You would require to have docker. For GPU version Nvidia-Docker (Linux only).

## Manual Setup
Run install (Linux only)
```
./bin/install
```
and import
```
import cavelab
```

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
V Saving/Loading model
V Import models (Graph Development)
V Rename dual models
V Goal train simple model
V Pair of data samples
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

Generalizable
```
- Generalizable Data reading
```

Nice to have
```
- Web-Interface
- Documentation
- Usage in Notebooks
```
