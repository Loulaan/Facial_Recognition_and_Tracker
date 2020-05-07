import os
from arcface.Learner import face_learner


net_mode = 'ir_se'
epochs = 20

learner = face_learner(True, net_mode)
learner.model.load_state_dict(f'{os.getcwd()}/weights/arcface/model_ir_se50.pth')
learner.train(epochs)
