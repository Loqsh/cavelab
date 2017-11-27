import tensorflow as tf
import numpy as np
'''
    NCCNet transforms input images and outputs Normalized cross-correlation of transformed images
    - Downloads the model from specified URL (in progress)
    - Loads the model into tensorflow
    - Provides API for transforming images and running normalized cross-correlation

    Images should be in (8, 512, 512) and templates in (8,160,160) unless models do not change the specification

    Example`
        ncc_net = NCCNet()
        images, templates = ncc_net.transform_full(s, t)
'''

class NCCNet():
    def __init__(self, downlaod=False, url='', directory='/FilterFinder/model/', name='model_nccnetv1'):

        if downlaod:
            directory, name = self._download_model(url)

        self.sess = self._load_model(directory, name)
        self._load_tensors()
        self.empty_template = np.zeros((8, 160, 160))
        self.empty_similarity = np.ones((8))

        self.resize_stack = {}

    # Should donwload from Google storagevideo on demand templates
    def _load_model(self, directory, name):
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        new_saver = tf.train.import_meta_graph(directory+name+'.ckpt.meta', clear_devices=True)
        new_saver.restore(sess, directory+name+'.ckpt')
        return sess

    # Defines inputs and outputs of computional graph
    def _load_tensors(self):
        graph = tf.get_default_graph()
        self.image = graph.get_tensor_by_name("input/image:0")
        self.template = graph.get_tensor_by_name("input/template:0")
        self.similarity = graph.get_tensor_by_name("input/similarity:0")

        self.image_transform = graph.get_tensor_by_name("Passes/image_transformed:0")
        self.template_transform = graph.get_tensor_by_name("Passes/template_transformed:0")

        self.normxcorr_layer = graph.get_tensor_by_name("normxcorr/normxcorr:0")
        self.full_loss = graph.get_tensor_by_name("full_loss:0")

    # Given image transforms through siamese network and returns
    def transform(self, images):
        feed_dict ={self.image: images, self.template: self.empty_template, self.similarity: self.empty_similarity}
        model_run = [self.image_transform]
        args = self.sess.run(model_run,feed_dict=feed_dict)
        return args[0]

    # Given template and image, transforms through the network and returns
    def transform_full(self, images, templates):
        feed_dict ={self.image: images, self.template: templates, self.similarity: self.empty_similarity}
        model_run = [self.image_transform, self.template_transform]

        args = self.sess.run(model_run,feed_dict=feed_dict)
        return args[0], args[1]

    # Given image and template, transforms through the networks, computes NCC and output
    def normxcorr(self, images, templates):
        feed_dict ={self.image: images, self.template: templates, self.similarity: self.empty_similarity}
        model_run = [self.image_transform, self.template_transform, self.normxcorr_layer, self.full_loss]

        args = self.sess.run(model_run,feed_dict=feed_dict)
        return args[0], args[1], args[2], args[3]

    # Building resize Graph
    def _build_resize(self, batch_size, source_width, template_width, scale = 3):
        print('Building resize graph...')
        in_image = tf.placeholder(tf.float32, shape=[batch_size, source_width, source_width])
        in_template = tf.placeholder(tf.float32, shape=[batch_size, template_width, template_width])

        in_image_re = tf.expand_dims(in_image, dim=3)
        in_template_re = tf.expand_dims(in_template, dim=3)

        re_image = tf.squeeze(tf.image.resize_nearest_neighbor(in_image_re, size=[int(scale*source_width), int(scale*source_width)]))
        re_template = tf.squeeze(tf.image.resize_nearest_neighbor(in_template_re, size=[int(scale*template_width), int(scale*template_width)]))

        self.resize_stack[(source_width, template_width, scale)] = [in_image, in_template, re_image, re_template]

    # input batches of pair of images
    def resize(self, image, template, scale):
        if not (image.shape[1], template.shape[1], scale) in self.resize_stack.keys():
            self._build_resize(image.shape[0], image.shape[1], template.shape[1], scale)

        in_image, in_template,re_image, re_template = self.resize_stack[(image.shape[1], template.shape[1], scale)]
        args = self.sess.run([re_image, re_template], feed_dict={in_image: image, in_template: template})
        return args[0], args[1]

    # Normalize and multiply by blend_map
