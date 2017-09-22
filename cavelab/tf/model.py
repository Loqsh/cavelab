import tensorflow as tf

class Graph(object):
    def __init__(self, directory="", name=""):
        self.kernels = []
        self.bias = []

        if directory !="":
            if name == "":
                name = "model"
            self.sess = self._load_model(directory, name)
            self.graph = tf.get_default_graph()

    def _load_model(self, directory, name):
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        new_saver = tf.train.import_meta_graph(directory+name+'.ckpt.meta', clear_devices=True)
        new_saver.restore(sess, directory+name+'.ckpt')
        return sess

    # Given input, transforms through the network and returns
    def process(self, inputs, outputs):
        feed_dict = {self.graph.get_tensor_by_name(name):inputs[name]  for name in inputs.keys()}
        model_run = [self.graph.get_tensor_by_name(name) for name in outputs]

        args = self.sess.run(model_run,feed_dict=feed_dict)
        return args
