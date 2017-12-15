# Cavelab
Deep-vision environment for rapid prototyping and scale deployment. Includes libraries such as Tensorflow and OpenCV. Specifically developed for Peta-scale processing on EM images, but highly generalizable.

## Usage

Start docker
```
nvidia-docker run -it --net=host \
      -v examples:/project \
      davidbuniat/cavelab:latest-gpu bash

cd /project
```
The following simple use case will run a Unet given parameters defined in hparams.json look for examples folder for more use cases
```
import cavelab as cl
hparams = cl.hparams(name="default") # Define hparam.json on main directory and provide parameters

model = cl.Graph()
image = tf.placeholder(tf.float32, shape=hparams.shape, name='image')
label = tf.placeholder(tf.float32, shape=hparams.shape, name='label')

pred = cl.models.FusionNet(image, hparams.kernels_shape)

loss = tf.reduce_mean(tf.square(pred-label))
model.train_step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(g.loss)

cl.tf.train(model,    name=hparams.name,
                      features=hparams.features,
                      batch_size=hparams.batch_size,
                      train_file=hparams.train_file, test_file=hparams.test_file)

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
- model rewrite is not working
```

Still to abstract
```
- Infer from features infer.py
```

Scaling Todo
```
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
