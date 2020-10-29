# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:30:23 2019

@author: sigma
"""

import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import gzip
import pickle

import torch
from mypath import Path
from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.consistent_deeplab import ConsistentDeepLab
import losses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.logger import Logger

# torch.cuda.set_device(1)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        # self.saver = Saver(args)
    # Recoder the running processing
        self.saver = Saver(args)
        sys.stdout = Logger(
            os.path.join(self.saver.experiment_dir, 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))  
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        if args.dataset == 'pairwise_lits':
            proxy_nclasses = self.nclass = 3
        elif args.dataset == 'pairwise_chaos':
            proxy_nclasses = 2*self.nclass
        else:
            raise NotImplementedError

        # Define network
        model = ConsistentDeepLab(in_channels=3,
                        num_classes=proxy_nclasses,
                        pretrained=args.pretrained,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
        #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer =torch.optim.Adam(train_params, weight_decay=args.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            weights = calculate_weigths_labels(args.dataset, self.train_loader, proxy_nclasses)
        else:
            weights = None
        
        # Initializing loss
        print("Initializing loss: {}".format(args.loss_type))
        self.criterion = losses.init_loss(args.loss_type, weights=weights)
        
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (sample1, sample2, proxy_label, sample_indices) in enumerate(tbar):
            image1, target1 = sample1['image'], sample1['label']
            image2, target2 = sample2['image'], sample2['label']
            if self.args.cuda:
                image1, target1 = image1.cuda(), target1.cuda()
                image2, target2 = image2.cuda(), target2.cuda()
                proxy_label = proxy_label.cuda()
            
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image1, image2)
            loss = self.criterion(output, proxy_label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                image = torch.cat((image1, image2), dim=-2)
                if len(proxy_label.shape)>3:
                    output = output[:,0:self.nclass]
                    proxy_label = torch.argmax(proxy_label[:,0:self.nclass], dim=1)
                self.summary.visualize_image(self.writer, self.args.dataset, image, proxy_label, output, global_step)
                

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image1.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        val_time = 0
        for i, (sample1, sample2, proxy_label, sample_indices)  in enumerate(tbar):
            image1, target1 = sample1['image'], sample1['label']
            image2, target2 = sample2['image'], sample2['label']
            if self.args.cuda:
                image1, target1 = image1.cuda(), target1.cuda()
                image2, target2 = image2.cuda(), target2.cuda()
                proxy_label = proxy_label.cuda()
            
            with torch.no_grad():
                start = time.time()
                output = self.model(image1, image2, is_val=True)
                end = time.time()
            val_time += end-start
            loss = self.criterion(output, proxy_label)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            proxy_label = proxy_label.cpu().numpy()
            
            # Add batch sample into evaluator
            if len(proxy_label.shape)>3:
                pred = np.argmax(pred[:,0:self.nclass], axis=1)
                proxy_label = np.argmax(proxy_label[:,0:self.nclass], axis=1)
            else: 
                pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(proxy_label, pred)
            
            if self.args.save_predict:
                self.saver.save_predict_mask(pred, sample_indices, self.val_loader.dataset.data1_files)
                
        print("Val time: {}".format(val_time))
        print("Total paramerters: {}".format(sum(x.numel() for x in self.model.parameters())))    
        if self.args.save_predict:
            namelist = []
            for fname in self.val_loader.dataset.data1_files:
                # namelist.append(fname.split('/')[-1].split('.')[0])
                _, name = os.path.split(fname)
                name = name.split('.')[0]
                namelist.append(name)
            file = gzip.open(os.path.join(self.saver.save_dir, 'namelist.pkl.gz'), 'wb')
            pickle.dump(namelist, file, protocol=-1)
            file.close()

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image1.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        
        dice = self.evaluator.Dice()
        # self.writer.add_scalar('val/Dice_1', dice[1], epoch)
        self.writer.add_scalar('val/Dice_2', dice[2], epoch)
        print("Dice:{}".format(dice))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34','resnet50', 'resnet101', 
                                 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pairwise_lits',
                        choices=['pairwise_lits', 'pairwise_chaos'],
                        help='dataset name (default: pairwise_lits)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default=None, 
                        choices=['ce', 'dice', 'multi_class_dice', 'multi_label_dice', 
                                 'multi_label_ce'],
                        help='loss func type (default: auto)')    
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                    help='whether to use pretrained backbone (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--only-val', action='store_true', default=False,
                        help='just evaluation but not training')
    parser.add_argument('--save-predict', action='store_true', default=False,
                        help='save predicted mask in disk')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'pairwise_lits': 40,
            'pairwise_chaos': 200,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    
    if args.loss_type is None:
        loss_type = {
            'pairwise_lits': 'multi_class_dice',
            'pairwise_chaos': 'multi_label_dice',
                }
        args.loss_type=loss_type[args.dataset.lower()]

    if args.lr is None:
        lrs = {
            'pairwise_lits': 0.0002,
            'pairwise_chaos': 0.0002,
        }
        # args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
        args.lr = lrs[args.dataset.lower()]

    if args.checkname is None:
        args.checkname = 'consistent_deeplab-'+str(args.backbone)
    
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    if args.only_val:
        if args.resume is not None:
            trainer.validation(trainer.args.start_epoch)
            return
        else:
            raise NotImplementedError
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    
    start = time.time()
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    end = time.time()
    print('Training time is: ', end-start)
    trainer.writer.close()

if __name__ == "__main__":
   main()
