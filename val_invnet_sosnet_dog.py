import os, argparse, torch, logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from models import invNet
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel
from utils import PerceptualLoss, TqdmLoggingHandler, AverageMeter
import skimage
import math
# Extract all features, and then scale and crop

def call_args():
    parser = argparse.ArgumentParser(description='Inversion model for feature map obtained with DoG keypoint detector')

    parser.add_argument('--descriptor',
                        type=str, required=True, choices=['korniasift', 'hardnet', 'sosnet'],
                        help='Descriptor to inverse')
    
    parser.add_argument('--imglist_path',
                        type=str, default='/workspace/mnt/MegaDepth/pre-processed',
                        help='')
    
    parser.add_argument('--descr_path',
                        type=str, 
                        default='/workspace/local/MegaDepth_DoG/dense/512_sosnet',
                        # default='/workspace/local/MegaDepth_DoG/dense/512_sift',
                        # default='/workspace/local/MegaDepth_DoG/dense/512_hardnet',
                        help='')
    
    parser.add_argument('--img_path',
                        type=str, default='/workspace/local/MegaDepth_DoG', #required=True,
                        help='')
    
    parser.add_argument('--batchsize',
                        type=int, default=128)
    
    # parser.add_argument('--server',
    #                     type=str, required=True, choices=['y', 'n'])
    
    # parser.add_argument('--use_ddp',
    #                     type=str, required=True, choices=['y', 'n'])
    
    parser.add_argument('--save_path',
                        type=str, default='/workspace/mnt/Project_DDE/workspace/PRE/weights/dense2')
    
    parser.add_argument('--inv_opt',
                        type=str, required=True, choices=['rgb', 'gray'])
    
    parser.add_argument('--img_size',
                        type=int, required=True, choices=[256, 512])
    
    parser.add_argument('--epoch_max',
                        type=int, default=200)
    
    parser.add_argument('--lr',
                        type=float, default=1e-4)
    parser.add_argument('--mode',
                        type=str, default='val',
                        choices=['train', 'val', 'test'])

    args = parser.parse_args()

    if args.descr_path is None:
        args.descr_path = os.path.join(args.img_path, 'dense', str(args.img_size))
    if args.imglist_path is None:
        args.imglist_path = args.img_path
    args.img_path = os.path.join(args.imglist_path, str(args.img_size))
    # args.imglist_path = os.path.join(args.imglist_path, 'dense')
    
    args.save_path = os.path.join(args.save_path, f'{args.descriptor}_{args.img_size}_{args.inv_opt}')
    os.makedirs(args.save_path, exist_ok=True)

    if args.inv_opt == 'rgb':
        args.inv_opt = 'RGB'
        args.out_channel = 3
    elif args.inv_opt == 'gray':
        args.inv_opt = 'L'
        args.out_channel = 1
    if args.mode in ['val', 'test']:
        args.batchsize = 1
    return args

class MegaDepthset():
    def __init__(self, mode, args):
        self.mode = mode
        imglist_path = Path(args.imglist_path) / f'{mode}_list_rewrite.txt'
        with open(imglist_path, 'r') as f:
             imglist = f.readlines()
        f.close()
        self.imglist = imglist
        del imglist
        self.descriptor = args.descriptor
        self.img_path = Path(args.img_path)
        self.desc_path = Path(args.descr_path)
        self.inv_opt = args.inv_opt
        self.img_size = args.img_size
        self.len = len(self.imglist)
        print(self.len)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_name = self.imglist[idx].rstrip('\n')
        img = Image.open(self.img_path / self.mode/ img_name).convert(self.inv_opt)
        transform = ToTensor()
        img = transform(img)

        # Bring keypoints and descriptors
        # Since DoG keypoints are not integer, round them to be used as indices
        keypoints_x = torch.from_numpy(np.load(self.desc_path / self.mode /f'{img_name}.{self.descriptor}')['keypoints_x'])
        keypoints_y = torch.from_numpy(np.load(self.desc_path / self.mode /f'{img_name}.{self.descriptor}')['keypoints_y'])
        descriptors = torch.from_numpy(np.load(self.desc_path / self.mode /f'{img_name}.{self.descriptor}')['descriptors'])

        # Form feature map
        fmap = torch.zeros((128, self.img_size, self.img_size))
        # for idx in range(len(descriptors)):
        #     fmap[:, int(keypoints_y[idx]), int(keypoints_x[idx])] = descriptors[idx,:]
        fmap[:, keypoints_y, keypoints_x] = descriptors.T

        return fmap, img
    
def cal_ssim(img1, img2):
    '''
    Input tensor must be CxHxW L2 normalized images
    '''
    img1 = img1.cpu().numpy()#.astype(np.float64)
    img2 = img2.cpu().numpy()#.astype(np.float64)
    ssim = skimage.metrics.structural_similarity(img1, img2, channel_axis=0, data_range=1)
    return ssim

def cal_mae(img1, img2):
    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)
    mae = np.mean(np.abs(img1-img2))
    return mae

def cal_psnr(img1, img2):
    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255. / math.sqrt(mse))

def run_epoch(gpu, model, optimizer, criterion, train_loader, mode, args):
    loss_sum = 0
    loader = train_loader
    len_loader = len(loader)
    if len_loader == 0:
        print('Empty dataloader occurerd')
        return

    ssim = []
    mae = []
    psnr = []
    model = model.eval()
    for idx, data in tqdm(enumerate(loader), total=len(loader), position=1):
        fmaps, imgs = data
        optimizer.zero_grad()

        if mode == 'train':
            recon_imgs = model(fmaps.to(gpu))
        #     loss = criterion(recon_imgs, imgs.to(gpu))
        #     loss.backward()
        #     optimizer.step()
        #     loss_sum += loss.item()
                
        #     if gpu == 0:
        #         # print(f'Epoch:{epoch}/{args.epoch_max}, Batch:{idx+1}/{len_loader}, loss:{loss.item()}\n')

        else:
            with torch.no_grad():
                recon_imgs = model(fmaps.to(gpu))
            if mode == 'val':
                ssim.append(cal_ssim(imgs.squeeze(0), recon_imgs.squeeze(0)))
            elif mode == 'test':
                psnr.append(cal_psnr(imgs, recon_imgs))
                mae.append(cal_mae(imgs, recon_imgs))

    if mode == 'train':
        loss_mean = loss_sum / len_loader
        logging.info(f'Loss: {loss_mean}')
        return loss_mean
    elif mode == 'val':
        return ssim
    elif mode == 'test':
        return [mae, psnr, ssim]
    
def save_results(num, results, args):
    if args.mode == 'train':
        return
    elif args.mode == 'val':
        ssim = results
        mean_ssim = np.mean(np.array(ssim))
        with open(f'{args.save_path}/val_ssim_{num}.txt', 'w') as f:
            f.writelines(f'{ssim_val} \n' for ssim_val in ssim)
            f.write(f'\nMean SSIM: {mean_ssim}')
        f.close()
        return mean_ssim
    elif args.mode == 'test':
        mae, psnr, ssim = results
        mean_mae = np.mean(np.array(mae))
        mean_psnr = np.mean(np.array(psnr))
        mean_ssim = np.mean(np.array(ssim))
        with open(f'{args.save_path}/test_mae_{num}.txt', 'w') as f:
            f.writelines(f'{mae_val} \n' for mae_val in mae)
            f.write(f'\nMean MAE: {mean_mae}')
        f.close()
        with open(f'{args.save_path}/test_psnr_{num}.txt', 'w') as f:
            f.writelines(f'{psnr_val} \n' for psnr_val in psnr)
            f.write(f'\nMean PSNR: {mean_psnr}')
        f.close()
        with open(f'{args.save_path}/test_ssim_{num}.txt', 'w') as f:
            f.writelines(f'{ssim_val} \n' for ssim_val in ssim)
            f.write(f'\nMean SSIM: {mean_ssim}')
        f.close()
        return

def main_worker(args):
    batchsize = args.batchsize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    invnet = invNet(in_channel=128, out_channel=args.out_channel).to(device)
    invnet.eval()

    trainset = MegaDepthset('val', args)
    train_loader = DataLoader(trainset, batchsize, shuffle=False, drop_last=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, invnet.parameters()), betas=(0.9, 0.999), weight_decay=0., lr=args.lr)
    criterion = PerceptualLoss().to(device)

    
    if args.mode == 'val':
        best_ssim = 0
        best_wts_file = 'temp'
        wts_list = os.listdir(args.save_path)
        wts_list = [wts_file for wts_file in wts_list if wts_file.split('.')[-1] == 'pth']

        for i, wts_file in tqdm(enumerate(wts_list), total=len(wts_list), position=0):
            if i > 1:
                invnet.load_state_dict(torch.load(os.path.join(args.save_path,wts_file)))
                results = run_epoch(device, invnet, optimizer, criterion, train_loader, mode='val', args=args)
                mean_ssim = save_results(i+1, results, args)
                if mean_ssim > best_ssim:
                    best_ssim = mean_ssim
                    best_wts_file = wts_file
            print(f'Best SSIM: {best_ssim}, file: {best_wts_file}')
    else:
        results = run_epoch(device, invnet, optimizer, criterion, train_loader, mode='test', args=args)
        save_results(results)

if __name__ == '__main__':
    args = call_args()
    main_worker(args)
