# ====================================================>>
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/9/20                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================>>

from model2_MSDB import Model
import torch
import torch.nn as nn
# import visdom
import losses_in_one
import tqdm
import torch.optim as optim
import utils
import logging
import numpy as np


class Trainer():
    def __init__(self, learning_rate, epochs, save_epoch, save_path, show_epoch, data_set, valid_set, log_name, cuda_num, PFA):
        self.lr = learning_rate
        self.epochs = epochs
        self.show_epoch = show_epoch
        self.save_epoch = save_epoch
        self.data_set = data_set
        self.valid_set = valid_set
        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.loss_func1 = losses_in_one.PerceptualLoss()                                              # T.B.D
        self.loss_func2 = nn.MSELoss()
        self.save_path = save_path
        self.device = 'cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu'
        self.max_psnr = []
        self.all_loss = []
        self.log_name = log_name
        self.PFA = PFA
        #self.visualizer = visdom.Visdom(env='MSDB_PFA')

    def get_log(self, file_name):
        logger = logging.getLogger('train')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(file_name, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def build_model(self):
        self.model = Model(feature_channel=64, PFA=self.PFA)                                                       # require params
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              factor=0.1,
                                                              patience=10,
                                                              verbose=True,
                                                              threshold=0.0001,
                                                              threshold_mode='rel',
                                                              cooldown=20,
                                                              min_lr=0.0000001, eps=1e-08)

        print('the model is created')

    def save_model(self, save_point, psnr, ssim):
        save_name = self.save_path + str(save_point) + 'psnr-' + str(psnr) + 'ssim-' + str(ssim)
        torch.save(self.model, save_name)
        print('the model is saved at ' + str(save_point))

    def train(self):
        self.build_model()
        logger = self.get_log(self.log_name)
        for epoch in range(self.epochs):
            self.model.train()
            tq = tqdm.tqdm(self.data_set, total=len(self.data_set))
            step_loss = []
            print('training, epoch={}'.format(epoch + 1))
            for mix in tq:
                data, target = mix['data'], mix['target']
                data = data.to(self.device)
                target = target.to(self.device)
                sharpen_result = self.model(data)
                # perceptual_loss = self.loss_func1(sharpen_result, target)             T.B.D
                loss = self.loss_func2(sharpen_result, target)
                step_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # self.visualizer.image((utils.InverseTensor(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(sharpen_result)), win='sharpend')  # T.B.D.
            tq.close()
            epoch_loss = sum(step_loss) / len(step_loss)
            self.scheduler.step(epoch_loss)
            print('Epoch loss is {}'.format(epoch_loss))
            logger.info("Epoch [%d/%d], Loss: %.4f" % (epoch + 1, self.epochs, epoch_loss))
            # self.visualizer.line(X=[epoch], Y=[epoch_loss], win='Loss', update='append')      # visdom is unavailable in HPC

            # ----------------------test in one epoch  (psnr / ssim
            self.model.eval()
            with torch.no_grad():
                avg_psnr = []
                avg_ssim = []
                print('testing, epoch={}'.format(epoch + 1))
                for i, mix in enumerate(self.valid_set):
                    data, target = mix['data'], mix['target']
                    data = data.to(self.device)
                    test_result = self.model(data)
                    test_result = utils.InverseTensor(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(test_result)
                    target_img = utils.InverseTensor(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(target)
                    psnr, ssim = utils.calculate_all(target_img, test_result)
                    avg_psnr.append(psnr)
                    avg_ssim.append(ssim)
                    print('psnr-----' + str(psnr))
                    print('ssim-----' + str(ssim))
                a_psnr = sum(avg_psnr) / len(avg_psnr)
                a_ssim = sum(avg_ssim) / len(avg_ssim)
                self.max_psnr.append(a_psnr)
                logger.info(" test in epoch [%d/%d], psnr: %.5f, ssim: %.5f" % (epoch + 1, self.epochs, a_psnr, a_ssim))
                print('psnr is {}, ssim is {}'.format(a_psnr, a_ssim))
            if epoch % self.save_epoch == 0 or (epoch >= 500 and a_psnr > max(self.max_psnr)):
                self.save_model(epoch, psnr=a_psnr, ssim=a_ssim)
        self.save_model('last', psnr=a_psnr, ssim=a_ssim)
        np.save('/zxzeng/project1/save_model/all_loss.npy', np.asarray(self.all_loss))
