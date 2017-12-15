import tensorflow as tf
import cavelab as cl


def crop(pred, images):
    p_shape = pred.get_shape().as_list()
    i_shape = images.get_shape().as_list()
    width = p_shape[2]
    width_full = i_shape[2]
    crop_begin = (width_full-width)/2
    crop_end= crop_begin+width
    n = p_shape[3]
    n_full = i_shape[3]

    images = images[:,crop_begin:crop_end,crop_begin:crop_end,:n]
    return pred, images

"""
    Implement Negative Log-likelyhood
"""
def loss(pred, images):

    loss = tf.reduce_mean(tf.square(pred-images))
    return loss

def build(hparams):
    #Init Graph and input to the model
    g = cl.Graph()
    shape = [hparams.batch_size, hparams.width, hparams.width, hparams.n_sequence]
    g.image = tf.placeholder(tf.float32, shape=shape, name='image')
    g.ground_truth = tf.expand_dims(g.image, axis=[-1])

    #Model
    g.pred = cl.models.FusionNet2_5D(g.ground_truth[:,:,:,:-1,:], hparams.kernels_shape, activation=tf.nn.relu)
    shape = hparams.kernels_shape[0]
    shape = [3,3,2, shape[4],shape[3]]
    g.pred = cl.tf.layers.conv3d(g.pred, shape, activation=tf.nn.relu)

    #Loss
    g.pred, g.gt = crop(g.pred[:,:,:,-hparams.pred_depth:,:], g.ground_truth)

    #Store
    cl.tf.metrics.image_summary(tf.squeeze(g.pred[:,:,:,-1]), 'last/pred')
    cl.tf.metrics.image_summary(tf.squeeze(g.gt[:,:,:,-1]), 'last/image')

    cl.tf.metrics.image_summary(tf.transpose(tf.squeeze(g.pred[0,:,:,:]), perm=[2,0,1]), 'sequence/prediction')
    cl.tf.metrics.image_summary(tf.transpose(tf.squeeze(g.gt[0,:,:,:]), perm=[2,0,1]), 'sequence/image')

    g.loss = loss(g.pred, g.gt)
    cl.tf.metrics.scalar(g.loss, 'loss')


    #TrainOps
    g.train_step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(g.loss)
    return g


if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    model = build(hparams)

    cl.tf.train(model,  name=hparams.name,
                        features=hparams.features,
                        logging=True,
                        batch_size=hparams.batch_size,
                        train_file=hparams.train_file, test_file=hparams.test_file,
                        training_steps=hparams.training_steps, testing_steps=hparams.testing_steps,
                        augmentation=hparams.augmentation)
