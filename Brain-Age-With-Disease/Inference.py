import numpy as np
import torch.nn as nn
import os,shutil,torch
import matplotlib.pyplot as plt
from utils.Tools import *
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error
import pandas as pd


def Inference(valid_loader, model, criterion, device,args
            , ues_masked_img = True
            , save_npy=False,npy_name='test_result.npz'
            , figure=False, figure_name='True_age_and_predicted_age.png'
            , excel_name='Test_result.xlsx'):

    '''
    [Do Test process according pretrained model]
    Args:
        valid_loader (torch.dataloader): [test set dataloader defined in 'main']
        model (torch CNN model): [pre-trained CNN model, which is used for brain age estimation]
        criterion (torch loss): [loss function defined in 'main']
        device (torch device): [GPU]
        save_npy (bool, optional): [If choose to save predicted brain age in npy format]. Defaults to False.
        npy_name (str, optional): [If choose to save predicted brain age, what is the npy filename]. Defaults to 'test_result.npz'.
        figure (bool, optional): [If choose to plot and save scatter plot of predicted brain age]. Defaults to False.
        figure_name (str, optional): [If choose to save predicted brain age scatter plot, what is the png filename]. Defaults to 'True_age_and_predicted_age.png'.
    Returns:
        [float]: MAE and pearson correlation coeficent of predicted brain age in teset set.
    '''

    losses = AverageMeter()
    MAE = AverageMeter()

    model.eval() # switch to evaluate mode
    out, targ, ID = [], [], []
    target_numpy, predicted_numpy, ID_numpy = [], [], []
    print('======= start prediction =============')
    # ======= start test programmer ============= #
    with torch.no_grad():
        for _, (img, img_mask, sid, target, male) in enumerate(valid_loader):
            
            img = img.to(device).type(torch.FloatTensor)
            img_mask = img_mask.to(device).type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).to(device)

            # ======= compute output and loss ======= #
            if ues_masked_img:
                output = model(img_mask)
            else:
                output = model(img)
                
            if args.model == 'glt':
                output = sum(output) / len(output)
            else:
                output = output
            out.append(output.cpu().numpy())
            targ.append(target.cpu().numpy())
            ID.append(sid)
            loss = criterion(output, target)
            mae = metric(output.detach(), target.detach().cpu())

            # ======= measure accuracy and record loss ======= #
            losses.update(loss, img.size(0))
            MAE.update(mae, img.size(0))

        targ = np.asarray(targ)
        out = np.asarray(out)
        ID = np.asarray(ID)

        for idx in ID:
            for i in idx:
                ID_numpy.append(i)
        
        for idx in out:
            for i in idx:
                predicted_numpy.append(i)
        
        for idx in targ:
            for i in idx:
                target_numpy.append(i)

        target_numpy = np.asarray(target_numpy)
        predicted_numpy = np.asarray(predicted_numpy)
        ID_numpy = np.expand_dims(np.asarray(ID_numpy), axis=1)
        
        print(target_numpy.shape, predicted_numpy.shape, ID_numpy.shape)
        
        Excel_data = np.concatenate([ID_numpy, target_numpy, predicted_numpy], axis=1)

        errors = predicted_numpy - target_numpy
        errors = np.squeeze(errors,axis=1)
        target_numpy = np.squeeze(target_numpy,axis=1)
        predicted_numpy = np.squeeze(predicted_numpy,axis=1)


        # ======= output several results  ======= #
        print('===============================================================\n')
        print(
            'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f} \n'.format(
            len(valid_loader), loss=losses, MAE=MAE))

        print('STD_err = ', np.std(errors))  
        print(' CC:    ',np.corrcoef(target_numpy,predicted_numpy)[0][1])
        print('PAD spear man cc',spearmanr(errors,target_numpy,axis=1))
        print('spear man cc',spearmanr(predicted_numpy,target_numpy,axis=1))

        print('\n =================================================================')

        if save_npy:
            savepath = os.path.join(args.output_dir,npy_name)
            np.savez(savepath 
                    ,target=target_numpy
                    ,prediction=predicted_numpy
                    )
            
        df = pd.DataFrame(Excel_data)
        df.columns = ["sub_id", "Chronological Age", "Predicted Brain Age"]
        df.to_excel(os.path.join(args.output_dir, excel_name))
        
        
        # ======= Draw scatter plot of predicted age against true age ======= #
        if figure is True:
            plt.switch_backend('agg')
            plt.figure()
            lx = np.arange(np.min(target_numpy),np.max(target_numpy))
            plt.plot(lx,lx,color='red',linestyle='--')
            plt.scatter(target_numpy,predicted_numpy)
            plt.xlabel('Chronological Age')
            plt.ylabel('predicted brain age')
            plt.savefig(args.output_dir + figure_name)
            
        
            
        return {'mae':MAE.avg ,'cc': np.corrcoef(target_numpy, predicted_numpy)[0][1]}
