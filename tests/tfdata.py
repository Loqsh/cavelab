from cavelab import tfdata
import os.path
from cavelab.tf import global_session

train_file = '/FilterFinder/data/prepared/bad_trainset_24000_612_324.tfrecords'

data = tfdata(train_file)

for i in range(10):
    print(i)
    data.get_batch()

global_session().close_sess()
