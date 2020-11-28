# ====================================================>>
# -*- coding:utf-8 -*-                          
# Author: z                                         
# Project: dual_feature_distillation_deblur
# Date: 2020/9/20                                     
# Description:                                            
#  << National University of Defense Technology >>  
# ====================================================>>


import argparse
from trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import torch
from data_reader import GiveMeData
import random

p = argparse.ArgumentParser()
p.add_argument('--dp', type=str, help='data path / def=d_set/data/', default='./GOPRO/GOPRO_3840FPS_AVG_3-21/train/blur/*/*.png')
p.add_argument('--lp', type=str, help='label path / def=d_set/label/', default='./GOPRO/GOPRO_3840FPS_AVG_3-21/train/sharp/*/*.png')
p.add_argument('--tdp', type=str, help='test data path / def=d_set/data/', default='./GOPRO/GOPRO_3840FPS_AVG_3-21/test/blur/*/*.png')
p.add_argument('--tlp', type=str, help='test label path / def=d_set/label/', default='./GOPRO/GOPRO_3840FPS_AVG_3-21/test/sharp/*/*.png')
p.add_argument('--lr', type=float, help='learning rate / def=0.002', default=0.0002)
p.add_argument('--e', type=int, help='epochs / def=100', default=100)
p.add_argument('--s', type=int, help='save epoch / def=200', default=5)
p.add_argument('--img', type=int, help='image visdom show / def=20', default=20)
p.add_argument('--t', type=str, help='is train / def=train', default='train')
p.add_argument('--b', type=int, help='batch size / def=1', default=1)
p.add_argument('--sh', type=bool, help='shuffle / def=T', default=True)
p.add_argument('--sp', type=str, help='save path / def=./save_model/', default='./save_model/')
p.add_argument('--lm', type=str, help='model loading path / no def', default='./save_model/35')
p.add_argument('--w', type=int, help='num of worker / def=0', default=0)
p.add_argument('--c', type=int, help='crop size / def=256', default=256)
p.add_argument('--log', type=str, help='log name', default='./train_log.txt')
p.add_argument('--gpu', type=int, help='num of GPU', default=0)
p.add_argument('--pfa', type=bool, help='pfa', default=False)
args = p.parse_args()


if __name__ == '__main__':
    dataset = GiveMeData(args.dp, args.lp)
    validset = GiveMeData(args.tdp, args.tlp)
    if args.t == 'train':
        train_set = DataLoader(dataset,
                               batch_size=args.b,
                               shuffle=True,
                               num_workers=args.w)

        num = len(validset)
        index = list(range(num))
        random.shuffle(index)
        valid_sampler = SubsetRandomSampler(index[:100])
        valid_set = DataLoader(validset,
                               batch_size=1,
                               shuffle=False,
                               sampler=valid_sampler)

        t = Trainer(learning_rate=args.lr,
                    epochs=args.e,
                    save_epoch=args.s,
                    save_path=args.sp,
                    show_epoch=args.img,
                    data_set=train_set,
                    valid_set=valid_set,
                    log_name='./train_log_PFA.txt',
                    cuda_num=0,
                    PFA=False)
        t.train()
        print('Done')

        # ------------------------------train------------------------------  TBD #
    elif args.t == 'test':
            pass
        # dataset = GiveMeData(args.tdp, args.tlp)
        # test_set = DataLoader(dataset, batch_size=args.b,
        #                       shuffle=args.sh,
        #                       num_workers=args.w)
        # test_model = torch.load(args.lm, map_location='cpu')
        # psnr, ssim = utils.calculate_all(test_model, test_set, cuda=False)
        # print(psnr)
        # print(ssim)
        # ---------------------------auto  TBD --------------------------------- #
    # elif args.t == 'auto':
    #     import optuna_trainer
    #     import optuna
    #
    #     dataset = GiveMeData(args.dp, args.lp)
    #     train_set = DataLoader(dataset,
    #                            batch_size=args.b,
    #                            shuffle=args.sh,
    #                            num_workers=args.w)
    #
    #     dataset = GiveMeData(args.tdp, args.tlp)
    #     test_set = DataLoader(dataset, batch_size=args.b,
    #                           shuffle=args.sh,
    #                           num_workers=args.w)
    #
    #     trainer = optuna_trainer.AutoTrainer(learning_rate=args.lr,
    #                 epochs=args.e,
    #                 save_epoch=args.s,
    #                 save_path=args.sp,
    #                 data_set=train_set,
    #                 show_epoch=args.img,
    #                 test_set=test_set)
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(trainer.train(), n_trials=100, timeout=600)
    #     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    #     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    #     print("Study statistics: ")
    #     print("  Number of finished trials: ", len(study.trials))
    #     print("  Number of pruned trials: ", len(pruned_trials))
    #     print("  Number of complete trials: ", len(complete_trials))
    #
    #     print("Best trial:")
    #     trial = study.best_trial
    #
    #     print("  Value: ", trial.value)
    #     print("  Params: ")
    #     for key, value in trial.params.items():
    #         print("    {}: {}".format(key, value))