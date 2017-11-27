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
To do
```
- 3D TFdata full implementation including data augmentation
- Deprecation Warning: /root/.cloudvolume/secrets/google-secret.json is now preferred to /root/.neuroglancer/secrets/google-secret.json.
- hparams save bug is not working
```

Still to abstract
```
- Infer from features infer.py
- Training
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
