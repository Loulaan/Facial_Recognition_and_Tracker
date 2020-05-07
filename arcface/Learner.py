import os
import math

import numpy as np
import bcolz
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

from .model import Backbone, Arcface, MobileFaceNet, l2_norm
from .utils import get_time, hflip_batch, separate_bn_paras
from .verifacation import evaluate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = Path(os.getcwd())
work_path = root_path/'arcface'
data_path = root_path/'data'
log_path = work_path/'logs'
threshold = 1.5

# for training on emore dataset
emore_folder = data_path/'faces_emore'
model_path = work_path/'models'


def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


def get_train_loader():
    ds, class_num = get_train_dataset(emore_folder/'imgs')
    loader = DataLoader(ds, batch_size=100, shuffle=True, pin_memory=True, num_workers=3)
    return loader, class_num


class face_learner(object):

    def __init__(self, inference=False, backbone='mobilefacenet'):
        if backbone == 'mobilefacenet':
            self.model = MobileFaceNet().to(device)
        if backbone == 'ir_50':
            self.model = Backbone(50, 0.6, 'ir_se').to(device)

        if not inference:
            self.batch_size = 100
            self.lr = 1e-3
            self.momentum = 0.9
            self.milestones = [12, 15, 18]
            self.loader, self.class_num = get_train_loader()

            self.writer = SummaryWriter(log_path)
            self.step = 0
            self.head = Arcface(classnum=self.class_num).to(device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if backbone == 'mobilefacenet':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=self.lr, momentum=self.momentum)
            if backbone == 'ir50':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=self.lr, momentum=self.momentum)

            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // 5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                self.loader.dataset.root.parent)
        else:
            self.threshold = threshold

    def save_state(self, accuracy, extra=None, model_only=False):
        save_path = model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def evaluate(self, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), 512])
        with torch.no_grad():
            while idx + self.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + self.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(device)) + self.model(fliped.to(device))
                    embeddings[idx:idx + self.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + self.batch_size] = self.model(batch.to(device)).cpu()
                idx += self.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(device)) + self.model(fliped.to(device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        return accuracy.mean(), best_thresholds.mean()
    
    def find_lr(self,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []

        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = CrossEntropyLoss(thetas, labels)
          
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                return log_lrs, losses    

    def train(self, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = CrossEntropyLoss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(accuracy)
                self.step += 1
                
        self.save_state(accuracy, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)