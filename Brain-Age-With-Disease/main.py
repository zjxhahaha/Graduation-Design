import argparse
import datetime
import json
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import tensorboardX   
from DataSet import IMG_Folder, IMG_Folder_with_mask, Integer_Multiple_Batch_Size
import timm.optim.optim_factory as optim_factory
from network.ScaleDense import ScaleDense
from Inference import Inference
from utils.Tools import *
from utils.lr_scheduler import build_scheduler
from utils.EarlyStopping import EarlyStopping
from engine import train_one_epoch,validate_one_epoch
# from network.Matrix_loss import *
from loss.Ranking_loss import rank_difference_loss
from loss.Matrix_loss import Matrix_distance_L2_loss, Matrix_distance_L3_loss,Matrix_distance_loss
from network.resnet import *
from network.CNN import CNN
from network.Global_Local_Transformer import GlobalLocalBrainAge
import torchio as tio 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# Set Default Parameter
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
cudnn.enabled = True
cudnn.benchmark = True
    
def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DataSet
    parser.add_argument('--train_folder'   ,default="/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/train/"         ,type=str, help="Train set data path ")
    parser.add_argument('--valid_folder'   ,default="/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/val/"         ,type=str, help="Validation set data path ")
    parser.add_argument('--test_folder'    ,default="/opt/zhaojinxin/TSAN/brain_age_estimation_transfer_learning/test/"          ,type=str, help="Test set data path ")
    parser.add_argument('--excel_path'     ,default="/opt/zhaojinxin/TSAN/18_combine.xls",type=str, help="Excel file path ")
    parser.add_argument('--mask_type'      ,default='cube')
    parser.add_argument('--output_dir', type=str, default='./ckpt_scale/', help='root path for storing checkpoints, logs')
    parser.add_argument('--num_workers', type=int, default=8, help='pytorch number of worker')

    # Model
    parser.add_argument('--model', type=str, default='scale', help='model name')
    parser.add_argument('--store_name', type=str, default='', help='experiment store name')
    parser.add_argument('--gpu', type=int, default=None)

    # ScaleDense
    parser.add_argument('--scale_block', type=int, default=5)
    parser.add_argument('--scale_channel', type=int, default=8)

    # Loss function
    parser.add_argument('--loss', type=str, default='l1', help='training loss type')
    parser.add_argument('--aux_loss', type=str, default='mse',  help='Aux training loss type')
    parser.add_argument('--lambd', type=float, default=10.0, help='Loss weighted between main loss and aux loss')
    parser.add_argument('--beta', type=float, default=10.0, help='Loss weighted between ranking loss and age difference loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Loss weighted between groud truth loss and constraint loss')
    parser.add_argument('--sorter', default='./Sodeep_pretrain_weight/best_lstmla_slen_32.pth.tar',type=str, help="When use ranking, the pretrained SoDeep sorter network weight need to be appointed")

    # Optimizer and Learning Rate
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='optimizer type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='optimizer weight decay')
    parser.add_argument('--schedular', type=str, default='step',help='choose the scheduler')
    parser.add_argument('--accumulation_steps', type=int, default=1)
    
    # StepLRScheduler
    parser.add_argument('--lr_decay_epochs', type=int, default=20 )
    parser.add_argument('--lr_decay_rate', type=float, default=0.5 )
    
    # warmup and cosine_lr_scheduler
    parser.add_argument('--warmup_lr_init', type=float, default=5e-7 )
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=5e-6 )

    # Training process
    parser.add_argument('--start_epoch',type=int, default=0,metavar='N', help='start epoch')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--print_freq', type=int, default=20, help='logging frequency')
    parser.add_argument('--img_size', type=int, default=224, help='image size used in training')


    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
    parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
    parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')
    
    return parser.parse_args()

def main(args, results):
    
    best_metric = 1e+6
    saved_metrics, saved_epochs = [], []

    print('===== hyper-parameter ====== ')
    print("=> network     : {}".format(args.model))
    print("=> batch size  : {}".format(args.batch_size))
    print("=> learning rate    : {}".format(args.lr))
    print("=> weight decay     : {}".format(args.weight_decay))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Save hyper-parameter into json file
    json_path = os.path.join(args.output_dir,'hyperparameters.json')
    with open(json_path,'w') as f:
        f.write(json.dumps(vars(args), ensure_ascii=False, indent=4, separators=(',', ':')))
        
   
    #  DataSet  
    print('=====> Preparing data...')
    
    transforms = tio.Compose(
        [
        #  tio.RandomAffine(scales=0.75),
         tio.RandomFlip(flip_probability=0.5, axes=('LR',)),
         tio.RandomBiasField(coefficients=1),
         tio.RandomSpike(),
         tio.RandomNoise(),
         tio.RandomBlur()
        ]
    )
    
    
    train_dataset = IMG_Folder_with_mask(args.excel_path,args.train_folder, transforms=None, mask_type=args.mask_type)
    val_dataset   = Integer_Multiple_Batch_Size(IMG_Folder_with_mask(args.excel_path,args.valid_folder, mask_type=args.mask_type), args.batch_size)
    test_dataset  = Integer_Multiple_Batch_Size(IMG_Folder_with_mask(args.excel_path, args.test_folder, mask_type=args.mask_type), args.batch_size)
   

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    
    os.system('echo "# Training data size:{} #" >> {}'.format(len(train_dataset), res))
    os.system('echo "# Valid data size:{} #" >> {}'.format(len(val_dataset), res))
    os.system('echo "# Evaluation data size:{} #" >> {}'.format(len(test_dataset), res))

    # Define the Model
    Model_List = {
                  'scale':ScaleDense(nb_block=args.scale_block,nb_filter=args.scale_channel),
                  'res18':resnet18(),
                  'res34':resnet34(),
                  'res50':resnet50(),
                  'cnn':CNN(nb_filter=8, nb_block=5),
                  'glt':GlobalLocalBrainAge(inplace=1,
                                            patch_size=32,
                                            step=32,
                                            nblock=8,
                                            backbone='vgg8')
    }
    
    model = Model_List[args.model]
    model = nn.DataParallel(model).to(device)
    model_test = model
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    # optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.95))

    optimizer = torch.optim.Adam(model.parameters()
                                 , lr=args.lr
                                 , weight_decay=args.weight_decay
                                 , betas=(0.9, 0.95)
                                 , amsgrad=True
                                 )
    # Learning rate Scheduler 
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    # Define Loss function
    loss_func_dict = {'l1': nn.L1Loss().to(device)
                     ,'mse': nn.MSELoss().to(device)
                    #  ,'ranking':rank_difference_loss(sorter_checkpoint_path=args.sorter,beta=args.beta).to(device)
                     ,'matrix_l1':Matrix_distance_loss(p=2)
                     ,'matrix_l2':Matrix_distance_L2_loss(p=2)
                     ,'matrix_l3':Matrix_distance_L3_loss(p=2)
                     }
        
    criterion1 = loss_func_dict[args.loss]#l1？
    criterion2 = loss_func_dict[args.aux_loss]#mse
    
    sum_writer = tensorboardX.SummaryWriter(args.output_dir)

    # Begin to Train the model
    #训练步数有规定？
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
                      model=model, data_loader=train_loader, optimizer=optimizer,epoch=epoch,
                      criterion_1=criterion1, criterion_2=criterion2, args=args,device=device
                      )
        
        valid_stats = validate_one_epoch(
                      model=model, data_loader=val_loader,
                      criterion_1=criterion1, criterion_2=criterion2, args=args,device=device
                      )
        
        # ===========  learning rate decay =========== #  
        lr_scheduler.step_update(epoch)

        for param_group in optimizer.param_groups:
            print("\n*learning rate {:.2e}*\n" .format(param_group['lr']))

        # ===========  write in tensorboard scaler =========== #
        sum_writer.add_scalar('train/loss', train_stats['loss'], epoch)
        sum_writer.add_scalar('train/mae', train_stats['mae'], epoch)
        sum_writer.add_scalar('valid/loss', valid_stats['loss'], epoch)
        sum_writer.add_scalar('valid/mae', valid_stats['mae'], epoch)
        sum_writer.add_scalar('HyperPara/lrt', param_group['lr'], epoch)
        
        valid_metric = valid_stats['mae']
        is_best = valid_metric < best_metric#判断是否是最好的
        #本次测试是最好的
        if is_best:
            best_metric = valid_metric
            
            # if epoch > int(args.epochs) // 10:
            saved_metrics.append(valid_metric)
            saved_epochs.append(epoch)
            save_checkpoint({
                 'epoch': epoch
                ,'arch': args.model
                ,'state_dict': model.state_dict()
                ,'optimizer': optimizer.state_dict()}, is_best, args.output_dir,epoch)
        # ===========  early_stopping needs the validation loss or MAE to check if it has decresed 
        early_stopping(valid_stats['mae'])
        if early_stopping.early_stop:
            print("======= Early stopping =======")
            break
    # cleanup
    torch.cuda.empty_cache()
    
    print('Epo - Mtc')
    mtc_epo = dict(zip(saved_metrics, saved_epochs))
    rank_mtc = sorted(mtc_epo.keys(), reverse=False)
    #还帮你输出一下最佳的 信息
    try:
        for i in range(len(rank_mtc)):
            if i < 5:
                print('{:03} {:.3f}'.format(mtc_epo[rank_mtc[i]]
                                        ,rank_mtc[i]))
                os.system('echo "epo:{:03} mtc:{:.3f}" >> {}'.format(mtc_epo[rank_mtc[i]],rank_mtc[i],results))
            else:
                clean_ckpt = os.path.join(args.output_dir,'ckpt-%s.pth.tar'% mtc_epo[rank_mtc[i]])
                os.system('rm {}'.format(clean_ckpt))
    except:
        pass
    
    if len(rank_mtc) > 0:
        for i in range(5):
            model = model_test
            keymtc = rank_mtc[i]
            ckpt_step_e = 'ckpt-%s.pth.tar' % mtc_epo[keymtc]
            ckpt_file_e = os.path.join(args.output_dir, ckpt_step_e)
            
            if os.path.isfile(ckpt_file_e):
                print("=> loading %s for evaluate" % ckpt_step_e)
                checkpoint_e = torch.load(ckpt_file_e)
                model.load_state_dict(checkpoint_e['state_dict'])
                
                Test_state = Inference(test_loader
                                      ,model
                                      ,criterion1
                                      ,device
                                      ,ues_masked_img= True
                                      ,args=args
                                      ,save_npy=False
                                      ,figure=True
                                      ,figure_name='ckpt-masked_img-'+ str(mtc_epo[keymtc])+'.png'
                                      ,excel_name='ckpt-masked_img-'+ str(mtc_epo[keymtc])+'.xlsx')
                os.system('echo " ================================== "')
                os.system('echo "# evaluate epo:{} #" >> {}'.format(mtc_epo[keymtc], res))
                os.system('echo "Valid model with masked image TEST MAE mtc:{:.5f}" >> {}'.format(Test_state['mae'], results))
                os.system('echo "Valid model with masked image TEST rr mtc:{:.5f}" >> {}'.format(Test_state['cc'], results))
                
                Test_state = Inference(test_loader
                                      ,model
                                      ,criterion1
                                      ,device
                                      ,ues_masked_img= False
                                      ,args=args
                                      ,save_npy=False
                                      ,figure=True
                                      ,figure_name='ckpt-raw_img-'+ str(mtc_epo[keymtc])+'.png'
                                      ,excel_name='ckpt-raw_img-'+ str(mtc_epo[keymtc])+'.xlsx')
                os.system('echo "Valid model with Raw image TEST MAE mtc:{:.5f}" >> {}'.format(Test_state['mae'], results))
                os.system('echo "Valid model with Raw image TEST rr mtc:{:.5f}" >> {}'.format(Test_state['cc'], results))
                
    return

if __name__ == "__main__":
    
    args = get_args_parser()
    #相关属性配置
    res = os.path.join(args.output_dir, 'result.txt')
    if os.path.isdir(args.output_dir): 
        if input("### output_dir exists, rm? ###") == 'y':
            os.system('rm -rf {}'.format(args.output_dir))

    # =========== set train folder =========== #
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print('=> training from scratch.\n')
    os.system('echo "train {}" >> {}'.format(datetime.datetime.now(), res))

    main(args, res)