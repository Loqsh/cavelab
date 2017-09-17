import tensorflow as tf
import shutil
import os

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass

class global_session(Singleton):
    def __init__(self, interactive=False, log_dir=""):
        config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        if interactive:
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.Session(config=config)

        if log_dir!="":
            self.add_log_writers(log_dir)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.run_metadata = tf.RunMetadata()

    def init():
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def add_log_writers(self, log_dir, clean_first=False):
        assert log_dir[-1] == '/'
        if clean_first and os.path.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_dir + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(log_dir + 'test')

    def log_save(self, writer, merge, step):
        if writer == self.train_writer:
            writer.add_run_metadata(self.run_metadata, 'step%03d' % step)
        writer.add_summary(merge, step)

    def create_saver(self):
        self.saver = tf.train.Saver()
        return self.saver

    def restore_weights(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(g.sess, ckpt.model_checkpoint_path)

    def get_sess(self):
        return self.sess

    def close_sess(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
