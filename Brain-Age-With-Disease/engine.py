import math
import sys
from typing import Iterable

import torch
from utils.Tools import *


def train_one_epoch(model, data_loader, optimizer, device, epoch, criterion_1, criterion_2, args):
    
    model.train(True)
    num_steps = len(data_loader)

    losses = AverageMeter()
    MAE = AverageMeter()
    MAE1 = AverageMeter()
    MAE2 = AverageMeter()
    LOSS1 = AverageMeter()
    LOSS2 = AverageMeter()

    for idx, (img, img_mask, sid, target, male) in enumerate(data_loader):

        # if(idx==10):
        #     break
        # =========== convert male lable to one hot type =========== #
        img = img.to(device)
        img_mask = img_mask.to(device)

        # 使用性别标签，处理性别
        male = torch.unsqueeze(male, 1)
        male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
        male = male.to(device).type(torch.FloatTensor)

        target = torch.cat([target,target],dim = 0)
        target = target.type(torch.FloatTensor).to(device)

        # =========== compute output and loss =========== #
        model.train()
        model.zero_grad()
        # print(img.shape)

        #现在增设性别标签
        raw_img_out = model(img,male,is_train = True)

        mask_img_out = model(img_mask,male,is_train = True)

        alpha = 2
        
        if args.model == 'glt':
            Loss1_list, Loss2_list = [], []
            for y_pred in out:
                sub_loss1 = criterion_1(y_pred, target)
                #这里计算的应该是总体上的平均值
                Loss1_list.append(sub_loss1)
                
                if args.lambd > 0:
                    sub_loss2 = criterion_2(y_pred, target)
                else:
                    sub_loss2 = 0
                Loss2_list.append(sub_loss2)
            loss1 = sum(Loss1_list)
            loss2 = sum(Loss2_list)
            out = sum(out) / len(out)
            
        else:
            # =========== compute loss =========== #
            # 混合在一起用MAE,因为需要masked_img，所以loss1_1赋予一个较大的权重
            loss1_0 = criterion_1(raw_img_out, target)
            loss1_1 = criterion_1(mask_img_out, target)
            loss1_2 = criterion_1(mask_img_out, raw_img_out)
            loss1 = (loss1_0 + alpha*loss1_1) + args.gamma * loss1_2

            if args.lambd > 0:

                loss2_0 = criterion_2(raw_img_out, target)
                loss2_1 = criterion_2(mask_img_out, target)
                loss2_2 = criterion_2(mask_img_out, raw_img_out)
                loss2 = (loss2_0 + alpha*loss2_1) + args.gamma * loss2_2
            else:
                loss2 = 0
        loss_1 = loss1 + args.lambd * loss2

        #后半部分
        half_size = target.size(0)//2

        loss1_0_2 = criterion_1(raw_img_out[:half_size], target[:half_size])
        loss1_1_2 = criterion_1(raw_img_out[:half_size], target[:half_size])
        loss1_2_2 = criterion_1(raw_img_out[:half_size], target[:half_size])
        loss1_2 = (loss1_0_2 + alpha*loss1_1_2) + args.gamma * loss1_2_2

        loss2_0_2 = criterion_2(raw_img_out[:half_size], target[:half_size])
        loss2_1_2 = criterion_2(raw_img_out[:half_size], target[:half_size])
        loss2_2_2 = criterion_2(raw_img_out[:half_size], target[:half_size])
        loss2_2 = (loss2_0_2 + alpha*loss2_1_2) + args.gamma * loss2_2_2
        
        loss_2 = loss1_2 + args.lambd * loss2_2


        #前半部分
        loss1_0_3 = criterion_1(raw_img_out[half_size:], target[half_size:])
        loss1_1_3 = criterion_1(raw_img_out[half_size:], target[half_size:])
        loss1_2_3 = criterion_1(raw_img_out[half_size:], target[half_size:])
        loss1_3 = (loss1_0_3 + alpha*loss1_1_3) + args.gamma * loss1_2_3

        loss2_0_3 = criterion_2(raw_img_out[half_size:], target[half_size:])
        loss2_1_3 = criterion_2(raw_img_out[half_size:], target[half_size:])
        loss2_2_3 = criterion_2(raw_img_out[half_size:], target[half_size:])
        loss2_3 = (loss2_0_3 + alpha*loss2_1_3) + args.gamma * loss2_2_3
        
        loss_3 = loss1_3 + args.lambd * loss2_3
        
        loss = loss_1 + loss_2 +loss_3        

        mae = metric(mask_img_out.detach(), target.detach().cpu())
        half_size = target.size(0)//2
        mae1 = metric(mask_img_out[:half_size].detach(), target[:half_size].detach().cpu())
        mae2 = metric(mask_img_out[half_size:].detach(), target[half_size:].detach().cpu())
        losses.update(loss, img.size(0))
        LOSS1.update(loss1,img.size(0))
        LOSS2.update(loss2,img.size(0))
        MAE.update(mae, img.size(0))
        MAE1.update(mae1, img.size(0))
        MAE2.update(mae2, img.size(0))
        if idx % args.print_freq == 0:
            print(
                  'Epoch: [{0} / {1}]   [step {2}/{3}]\t'
                  'Loss1 {LOSS1.val:.3f} ({LOSS1.avg:.3f})\t'
                  'Loss2 {LOSS2.val:.3f} ({LOSS2.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE1 {MAE.val:.3f} ({MAE1.avg:.3f})\t'
                  'MAE2 {MAE.val:.3f} ({MAE2.avg:.3f})\t'
                  'MAE {MAE.val:.3f} ({MAE.avg:.3f})\t'.format
                  ( epoch, args.epochs, idx, len(data_loader)
                  , LOSS1=LOSS1
                  , LOSS2=LOSS2
                  , loss=losses
                  , MAE1=MAE1
                  , MAE2=MAE2
                  , MAE=MAE ))

        # =========== loss gradient back progation and argsimizer parameter =========== #
        loss.backward()
        
        if args.accumulation_steps > 1:
            if ((idx + 1) % args.accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()

        # break
    return {'loss':losses.avg, 'mae':MAE.avg}

def validate_one_epoch(model, data_loader, criterion_1, criterion_2, device, args):
    '''
    For validation process
    
    Args:
        valid_loader (data loader): validation data loader.
        model (CNN model): convolutional neural network.
        criterion1 (loss fucntion): main loss function.
        criterion2 (loss fucntion): aux loss function.
        device (torch device type): default: GPU
    Returns:
        [float]: training loss average and MAE average
    '''
    losses = AverageMeter()
    MAE = AverageMeter()

    # =========== switch to evaluate mode ===========#
    model.eval()

    with torch.no_grad():
        for _, (img, img_mask, sid, target, male) in enumerate(data_loader):
            img = img.to(device)
            img_mask = img_mask.to(device)
            
            target = target.type(torch.FloatTensor).to(device)

            # 使用性别标签，处理性别
            male = torch.unsqueeze(male, 1)
            male = torch.zeros(male.shape[0], 2).scatter_(1, male, 1)
            male = male.to(device).type(torch.FloatTensor)

            # =========== compute output and loss =========== #
            out = model(img_mask,male,is_train = False)
            
            if args.model == 'glt':
                Loss1_list, Loss2_list = [], []
                for y_pred in out:
                    sub_loss1 = criterion_1(y_pred, target)
                    Loss1_list.append(sub_loss1)
                    
                    if args.lambd > 0:
                        sub_loss2 = criterion_2(y_pred, target)
                    else:
                        sub_loss2 = 0
                    Loss2_list.append(sub_loss2)
                loss1 = sum(Loss1_list)
                loss2 = sum(Loss2_list)
                out = sum(out) / len(out)
            else:
                # =========== compute loss =========== #
                loss1 = criterion_1(out, target)
                if args.lambd > 0:
                    loss2 = criterion_2(out, target)
                else:
                    loss2 = 0
            loss = loss1 + args.lambd * loss2
            mae = metric(out.detach(), target.detach().cpu())

            # =========== measure accuracy and record loss =========== #
            losses.update(loss, img.size(0))
            MAE.update(mae, img.size(0))
        print(
                'Valid: [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f}'.format(
                len(data_loader), loss=losses, MAE=MAE))

        return {'loss':losses.avg, 'mae':MAE.avg}