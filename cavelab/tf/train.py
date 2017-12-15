import cavelab as cl

'''
    Takes input model and its params trains the model and saves
    *model must containt model.loss tensor and model.$(Features.keys())
'''

def train(model, name="", features={}, logging=False, batch_size=8,
                train_file="", test_file="",training_steps=0, testing_steps=0,
                augmentation={
                    "flipping": False,
                    "random_brightness": False,
                    "random_elastic_transform": False
                }):

    data = cl.tfdata(train_file,
                    batch_size = batch_size,
                    features = features,
                    flipping = augmentation["flipping"],
                    random_brightness = augmentation["random_brightness"],
                    random_elastic_transform = augmentation["random_elastic_transform"])

    test_data = cl.tfdata(test_file, features=features)
    sess = cl.global_session().get_sess()

    if logging:
        cl.global_session().add_log_writers('logs/'+ name+'/', clean_first=True)
        cl.global_session().restore_weights()

    try:
        for i in range(training_steps):
            outputs = data.get_batch()
            if not isinstance(outputs, basestring):
                outputs = [outputs]

            model_run = [model.train_step, model.loss]
            if logging:
                model_run.append(cl.global_session().merged)

            feed_dict = {k+":0":v for k,v in zip(list(features), outputs)}

            step = sess.run(model_run,
                            feed_dict=feed_dict,
                            run_metadata=cl.global_session().run_metadata)
            print(i, step[1])
            if i%20==0 and logging:
                cl.global_session().log_save(cl.global_session().train_writer, step[-1], i)

            if i%1000 == 0 and logging:
                evaluate(model, features, test_data, testing_steps, i)
                cl.global_session().model_save()
    finally:
        cl.global_session().close_sess()

def evaluate(model, features, data, steps, index):
    sess = cl.global_session().get_sess()

    for i in range(steps):
        outputs = data.get_batch()
        if not isinstance(outputs, basestring):
            outputs = [outputs]

        model_run = [model.loss,
                    cl.global_session().merged]
        feed_dict = {k+":0":v for k,v in zip(list(features), outputs)}

        step = sess.run(model_run, feed_dict=feed_dict)

        if i==0:
            print('evaluate', step[0])
            cl.global_session().log_save(cl.global_session().test_writer, step[-1], index)
