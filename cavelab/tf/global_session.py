import tensorflow as tf
import shutil
import os
import json

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
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.allow_growth = True

        if interactive:
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.Session(config=config)

        self.log_dir = log_dir
        if self.log_dir!="":
            self.add_log_writers(self.log_dir)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.run_metadata = tf.RunMetadata()
        self.saver = tf.train.Saver()


    def init():
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

    def add_log_writers(self, log_dir, hparams={}, clean_first=False):
        assert log_dir[-1] == '/'
        self.log_dir = log_dir
        if clean_first and os.path.exists(log_dir):
            print("Directory is not empty", log_dir)
            name = raw_input("Do you want to rewrite? (Y) ")
            if str(name)=="Y":
                shutil.rmtree(log_dir)
            else:
                raise Exception("Directory was not empty, exiting!", log_dir)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_dir + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(log_dir + 'test')

        # Save meta information
        with open(log_dir+'hparams.json', 'w') as outfile:
            json.dump(hparams, outfile)

    def log_save(self, writer, merge, step):
        if writer == self.train_writer:
            writer.add_run_metadata(self.run_metadata, 'step%03d' % step)
        writer.add_summary(merge, step)

    def model_save(self):
        self.saver.save(self.sess, self.log_dir+"model.ckpt")

    def restore_weights(self):
        model_dir =self.log_dir
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_sess(self):
        return self.sess

    def close_sess(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
