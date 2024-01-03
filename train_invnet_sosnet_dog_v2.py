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
# Extract all features, and then scale and crop

def call_args():
    parser = argparse.ArgumentParser(description='Inversion model for feature map obtained with DoG keypoint detector')

    parser.add_argument('--descriptor',
                        type=str, required=True, choices=['sift', 'hardnet', 'sosnet'],
                        help='Descriptor to inverse')
    
    parser.add_argument('--imglist_path',
                        type=str, default='/workspace/mnt/MegaDepth/pre-processed',
                        help='')
    
    parser.add_argument('--descr_path',
                        type=str, default='/workspace/local/MegaDepth_DoG/dense/256',
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
    
def run_epoch(gpu, model, optimizer, criterion, train_loader, epoch, mode, args):
    loss_sum = 0
    loader = train_loader
    len_loader = len(loader)
    if len_loader == 0:
        print('Empty dataloader occurerd')
        return

    for idx, data in tqdm(enumerate(loader), position=1):
        fmaps, imgs = data
        optimizer.zero_grad()
        recon_imgs = model(fmaps.to(gpu))
        loss = criterion(recon_imgs, imgs.to(gpu))
        if mode == 'train':
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
                
            if gpu == 0:
                print(f'Epoch:{epoch}/{args.epoch_max}, Batch:{idx+1}/{len_loader}, loss:{loss.item()}\n')

    loss_mean = loss_sum / len_loader
    logging.info(f'Loss: {loss_mean}')
    return loss_mean

def main_worker(gpu, ngpus, args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if gpu == 0 else logging.WARNING,
        handlers=[TqdmLoggingHandler()])

    batchsize = int(args.batchsize / ngpus)

    # Initializing the process group
    logging.info('Initializing process group')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', world_size=ngpus, rank=gpu)
    logging.info('Done!')

    logging.info('Make inversion model... at')
    invnet = invNet(in_channel=128, out_channel=args.out_channel).cuda(gpu)
    invnet.cuda(gpu)
    invnet = DistributedDataParallel(invnet, device_ids=[gpu])
    invnet.train()
    logging.info('Done!')
    barrier()

    logging.info('Load data...')
    trainset = MegaDepthset('train', args)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    train_loader = DataLoader(trainset, batchsize, shuffle=False, drop_last=True, pin_memory=True, sampler=train_sampler)
    logging.info('Done!')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, invnet.parameters()), betas=(0.9, 0.999), weight_decay=0., lr=args.lr)
    criterion = PerceptualLoss().cuda(gpu)
    barrier()

    for epoch in tqdm(range(args.epoch_max), position=0):
        train_sampler.set_epoch(epoch)
        run_epoch(gpu, invnet, optimizer, criterion, train_loader, epoch, mode='train', args=args)
        # scheduler.step()
        if gpu == 0:
            torch.save(invnet.module.state_dict(), args.save_path + '/net-epoch-{}.pth'.format(epoch+1))

if __name__ == '__main__':
    args = call_args()
    ngpus = torch.cuda.device_count()
    world_size = ngpus
    print(f'{ngpus} GPU available!')
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
