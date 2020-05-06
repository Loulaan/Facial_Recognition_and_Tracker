from arcface.config import get_config
from arcface.Learner import face_learner



net_mode = 'ir_se'
net_depth = 50
lr = 1e-3
batch_size = 96
num_workers = 3
epochs = 20

conf = get_config()

if net_mode == 'mobilefacenet':
    conf.use_mobilfacenet = True
else:
    conf.net_mode = net_mode
    conf.net_depth = net_depth

conf.lr = lr
conf.batch_size = batch_size
conf.num_workers = num_workers
conf.data_mode = 'emore'
learner = face_learner(conf)
learner.model.load_state_dict('arcface/model/model_ir_se50.pth')

learner.train(conf, epochs)