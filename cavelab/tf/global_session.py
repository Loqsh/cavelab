import tensorflow as tf

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass

class global_session(Singleton):
    def __init__(self, interactive=False):
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        if interactive:
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.Session(config=config)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def init():
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def add_log_writers(self, log_dir):
        assert log_dir[-1] == '/'
        self.merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + 'train', self.sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + 'test')
        return train_writer, test_writer, self.merged

    def create_saver():
        self.saver = tf.train.Saver()
        return self.saver

    def restor_weights(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(g.sess, ckpt.model_checkpoint_path)

    def get_sess(self):
        return self.sess

    def close_sess(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
