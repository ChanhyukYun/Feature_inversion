import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import logging

dist_th = 8e-3# threshold from HardNet, negative descriptor pairs with the distances lower than this threshold are treated as false negatives
eps_l2_norm = 1e-10
eps_sqrt = 1e-6

def cal_l2_distance_matrix(x, y, flag_sqrt=True):
    ''''distance matrix of x with respect to y, d_ij is the distance between x_i and y_j'''
    D = torch.abs(2 * (1 - torch.mm(x, y.t())))
    if flag_sqrt:
        D = torch.sqrt(D + eps_sqrt)
    return D

def read_UBC_patch_opencv(train_root, sz_patch):
    patch = []
    file = sorted(os.listdir(train_root))
    nb_file = len(file)
    sz_patch_raw = 64
    flag_resize = False
    if sz_patch_raw != sz_patch:
        flag_resize = True
    for i, img_file in enumerate(file):
        if img_file.find('bmp') > -1:
            img = cv2.imread(os.path.join(train_root, img_file), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = img.shape#height is row, width is column
            print('reading:{} of {}'.format(i, nb_file), end='\r')
            for v in range(0, img_height, sz_patch_raw):  # Vertival
                for h in range(0, img_width, sz_patch_raw):  # Horizontal
                    patch_temp = img[v:v+sz_patch_raw, h:h+sz_patch_raw]#64x64
                    if flag_resize:
                        patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                    patch.append(patch_temp)
    print('reading patch:{} of {}'.format(i, nb_file))


    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def read_UBC_pointID(train_root):
    pointID = []
    with open(os.path.join(train_root, 'info.txt')) as f:
        for line in f:
            id = int(line.split(' ')[0])
            pointID.append(id)
            print('reading pointID:id{}'.format(id), end='\r')
    print('max ID:{}'.format(id))
    return np.array(pointID)

def read_hpatches_patch_opencv(train_root, sz_patch):
    sz_patch_raw = 65
    if sz_patch_raw != sz_patch:
        flag_resize = True
    patch = []
    scene_file = sorted(os.listdir(train_root))
    num_scene = len(scene_file)
    sets = ['e', 'h', 't']
    for i, scene in enumerate(scene_file):#
        print('reading:{} of {} scene'.format(i, num_scene), end='\r')
        img_all = []
        img_all.append(cv2.imread(os.path.join(train_root, scene, 'ref.png'), cv2.IMREAD_GRAYSCALE))
        for j, set in enumerate(sets):
            for k in range(1, 6):
                img_all.append(cv2.imread(os.path.join(train_root, scene, set + str(k)+'.png'), cv2.IMREAD_GRAYSCALE))

        img_height, _ = img_all[0].shape  # height is row, width is column

        for v in range(0, img_height, sz_patch_raw): # only Vertival
            for i, img in enumerate(img_all):
                patch_temp = img[v:v+sz_patch_raw]
                if flag_resize:
                    patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                patch.append(patch_temp)

    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def cal_index_train_all(index_unique_label, inb_label_each_batch, epoch_max):
    index_train = []
    num_label = len(index_unique_label)
    num_patch = 0
    for i in range(num_label):
        num_patch += index_unique_label[i].size

    nb_batch_each_epoch = int(np.ceil(num_label/inb_label_each_batch))

    for e_loop in range(epoch_max):
        #loop over each epoch
        each_epoch_index = []
        print('calculating train index:epoch {} of {}'.format(e_loop+1, epoch_max), end='\r')
        for b_loop in range(nb_batch_each_epoch):
            each_batch_index = []
            #loop over each batch in each epoch
            for i in range(inb_label_each_batch):
                j_max = len(index_unique_label[i])
                for j in range(j_max):
                    each_batch_index.append(index_unique_label[i][j])
            each_epoch_index.append(each_batch_index)
            index_unique_label = np.roll(index_unique_label, -inb_label_each_batch)
        index_train.append(each_epoch_index)
        np.random.shuffle(index_unique_label)
    return np.array(index_train)

def cal_index_train(index_unique_label, num_label_each_batch, num_img_each_label, epoch_max):
    print('calculating index_train...')
    #ensure input is numpy array
    index_train = []

    num_label = len(index_unique_label)
    num_patch = 0
    for i in range(num_label):
        num_patch += index_unique_label[i].size

    index_index = [i for i in range(num_label)]#for random shuffule

    index_unique_label0 = index_unique_label.copy()

    sz_batch = num_img_each_label*num_label_each_batch
    num_batch_each_epoch = int(num_patch/sz_batch)
    for e_loop in range(epoch_max):
        #loop over each epoch
        each_epoch_index = []
        print('calculating train index:epoch {} of {}'.format(e_loop,epoch_max))
        for b_loop in tqdm(range(num_batch_each_epoch)):#num_batch_each_epoch
            #loop over each batch in each epoch
            each_batch_index = []
            for i in range(num_label_each_batch):
                #loop over each label in each batch
                if len(index_unique_label[i]) < num_img_each_label:
                    np.random.shuffle(index_unique_label0[i])
                    index_unique_label[i] = index_unique_label0[i]
                    #refill the variable if less than num_img_each_label
                for j in range(num_img_each_label):
                    each_batch_index.append(index_unique_label[i][0])
                    if b_loop + i + j == 0:
                        unique_label_temp = np.delete(index_unique_label[i], [0])
                        index_unique_label = list(index_unique_label)
                        index_unique_label[i] = unique_label_temp
                        index_unique_label = np.array(index_unique_label, dtype=object)
                    else:
                        index_unique_label[i] = np.delete(index_unique_label[i], [0])

            each_epoch_index.append(each_batch_index)
            index_unique_label = np.roll(index_unique_label, -num_label_each_batch)
            index_unique_label0 = np.roll(index_unique_label0, -num_label_each_batch)

            if (b_loop+1) % int(np.ceil(num_label/num_label_each_batch)) == 0:
                random.shuffle(index_index)
                index_unique_label = index_unique_label[index_index]
                index_unique_label0 = index_unique_label0[index_index]

        index_train.append(each_epoch_index)

    return np.array(index_train)

def load_UBC_for_train(data_root, train_set, sz_patch=32, nb_pt_each_batch=512, nb_pat_per_pt=2, epoch_max=200, flag_load_index=True): # all outputs are numpy arrays
    train_root = os.path.join(data_root, train_set)
    file_data_train = os.path.join(train_root, train_set + '_sz' + str(sz_patch) + '.npz')
    file_index_train = os.path.join(train_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_per_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        patch = read_UBC_patch_opencv(train_root, sz_patch)
        pointID = read_UBC_pointID(train_root)
        index_unique_ID = []  # it is a list
        pointID_unique = np.unique(pointID)
        for id in pointID_unique:
            index_unique_ID.append(np.argwhere(pointID == id).squeeze())
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=np.array(index_unique_ID, dtype=object))
    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            if nb_pat_per_pt == -1:
                index_train = cal_index_train_all(index_unique_ID, nb_pt_each_batch, epoch_max)
            else:
                index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_per_pt, epoch_max)
            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train

def extract_100K_test(patch_train,pointID_train,test_root):
    patch_loc = []
    index_test = []
    with open(os.path.join(test_root, 'm50_100000_100000_0.txt')) as f:
        for line in f:
            id = line.split(' ')
            patch_loc.append(int(id[0]))
            patch_loc.append(int(id[3]))
            index_test.append([int(id[0]), int(id[3])])

    patch_loc = np.array(patch_loc)
    patch_loc = np.unique(patch_loc)
    pointID_test = pointID_train[patch_loc]
    patch_test = patch_train[patch_loc]
    for i in range(len(index_test)):
        index_test[i][0] = np.argwhere(patch_loc == index_test[i][0]).squeeze()
        index_test[i][1] = np.argwhere(patch_loc == index_test[i][1]).squeeze()

    return patch_test, pointID_test, np.array(index_test)

def load_UBC_for_test(data_root, test_set, sz_patch=32): # all outputs are numpy arrays
    test_root = os.path.join(data_root, test_set)
    file_data_test = os.path.join(test_root, test_set + '_sz' + str(sz_patch) + '_100k_test.npz')

    if os.path.exists(file_data_test):
        print('Test data of {} already exists!'.format(test_set))
        data = np.load(file_data_test, allow_pickle=True)
        patch_test = data['patch']
        pointID_test = data['pointID']
        index_test = data['index']#Only tesy data have attribuate 'index'
    else:
        file_data_train = os.path.join(test_root, test_set + '_sz' + str(sz_patch) + '.npz')
        if os.path.exists(file_data_train):
            # If there is train data
            data_train = np.load(file_data_train, allow_pickle=True)
            patch_train = data_train['patch']
            pointID_train = data_train['pointID']
            del data_train
        else:
            # First generate the train data
            print(test_set)
            patch_train = read_UBC_patch_opencv(test_root, sz_patch)
            pointID_train = read_UBC_pointID(test_root)
            np.savez(file_data_train, patch=patch_train, pointID=pointID_train)

        patch_test, pointID_test, index_test = extract_100K_test(patch_train, pointID_train, test_root)
        np.savez(file_data_test, patch=patch_test, pointID=pointID_test, index=index_test)

    return patch_test, pointID_test, index_test

def load_hpatches_for_train(data_root, sz_patch, nb_pt_each_batch, nb_pat_each_pt, epoch_max, flag_load_index=True):
    train_set = 'hpatches'
    train_root = os.path.join(data_root, 'hpatches-benchmark-master/data/hpatches-release')
    save_root = os.path.join(data_root, 'hpatches-benchmark-master/data/')

    file_data_train = os.path.join(save_root, train_set + '_sz' + str(sz_patch) + '.npz')
    file_index_train = os.path.join(save_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_each_pt) + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = data['patch']
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        patch = read_hpatches_patch_opencv(train_root, sz_patch)
        num_uniqueID = int(len(patch)/16)
        pointID_unique = np.array(range(0, num_uniqueID))
        pointID = []
        for i in range(0, num_uniqueID):
            for j in range(0, 16):
                pointID.append(i)

        pointID = np.array(pointID)
        index_unique_ID = []  # it is a list
        for id in pointID_unique:
            index_unique_ID.append(np.argwhere(pointID == id).squeeze())
        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=index_unique_ID)

    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_each_pt, epoch_max)

            np.save(file_index_train, index_train)

    return torch.from_numpy(patch), pointID, index_train

def load_hpatches_split_train(data_root, sz_patch, nb_pt_each_batch=512, nb_pat_each_pt=2, epoch_max=100, split_name='a', flag_load_index=True, flag_std_filter=True):
    train_set = 'hpatches'
    train_root = os.path.join(data_root, 'hpatches-benchmark-master/data/hpatches-release')
    save_root = os.path.join(data_root, 'hpatches-benchmark-master/data/')
    file_data_train = os.path.join(save_root, train_set + '_sz' + str(sz_patch) + '_split_' + split_name + '.npz')
    file_index_train = os.path.join(save_root, train_set + '_index_train_ID' + str(nb_pt_each_batch) + '_pat' + str(nb_pat_each_pt) + '_split_' + split_name + '.npy')
    if os.path.exists(file_data_train):
        print('train data of {} already exists!'.format(train_set))
        data = np.load(file_data_train, allow_pickle=True)
        patch = torch.from_numpy(data['patch'])
        pointID = data['pointID']
        index_unique_ID = data['index_unique_ID']
        del data
    else:
        print(train_set)
        split_file = os.path.join(data_root, 'hpatches-benchmark-master/tasks/splits/splits.json')
        with open(split_file) as f:
            split = json.load(f)
        train_split = split[split_name]['train']
        patch = read_hpatches_patch_split_opencv(train_root, train_split, sz_patch)
        num_uniqueID = int(len(patch)/16)
        pointID_unique = np.array(range(0, num_uniqueID))
        pointID = []
        for i in range(0, num_uniqueID):
            for j in range(0, 16):
                pointID.append(i)
        pointID = np.array(pointID)

        if flag_std_filter:
            patch_raw = read_hpatches_patch_split_opencv(train_root, train_split, sz_patch=65)
            patch_std = compute_patch_contrast(patch_raw)  # return numpy array
            indice_high_std = np.argwhere(patch_std > 0).squeeze()
            patch = patch[indice_high_std]
            pointID = pointID[indice_high_std]
            pointID_unique = np.unique(pointID)


        index_unique_ID = []  # it is a list
        for id in pointID_unique:
            indice_ID = np.argwhere(pointID == id).squeeze()
            if len(indice_ID) >= 2:
                index_unique_ID.append(indice_ID)

        np.savez(file_data_train, patch=patch, pointID=pointID, index_unique_ID=index_unique_ID)

    index_train = []
    if flag_load_index:
        if os.path.exists(file_index_train):
            print('index_train of {} already exists!'.format(train_set))
            index_train = np.load(file_index_train, allow_pickle=True)
        else:
            index_train = cal_index_train(index_unique_ID, nb_pt_each_batch, nb_pat_each_pt, epoch_max)
            np.save(file_index_train, index_train)

    return patch, pointID, index_train

def read_hpatches_patch_split_opencv(train_root, scene_train, sz_patch):
    sz_patch_raw = 65
    flag_resize = False
    if sz_patch_raw != sz_patch:
        flag_resize = True
    patch = []
    num_scene = len(scene_train)

    sets = ['e', 'h', 't']
    for i, scene in enumerate(scene_train):
        print('reading:{} of {} scene'.format(i, num_scene), end='\r')
        img_all = []
        img_all.append(cv2.imread(os.path.join(train_root, scene, 'ref.png'), cv2.IMREAD_GRAYSCALE))
        for j, set in enumerate(sets):
            for k in range(1, 6):
                img_all.append(cv2.imread(os.path.join(train_root, scene, set + str(k)+'.png'), cv2.IMREAD_GRAYSCALE))

        img_height, _ = img_all[0].shape  # height is row, width is column

        for v in range(0, img_height, sz_patch_raw): # only Vertival
            for i, img in enumerate(img_all):
                patch_temp = img[v:v+sz_patch_raw]
                if flag_resize:
                    patch_temp = cv2.resize(patch_temp, (sz_patch, sz_patch))
                patch.append(patch_temp)

    return np.expand_dims(np.array(patch), 1)# N*1*sz_patch*sz_patch

def data_aug(patch, num_ID_per_batch):
    # sz = patch.size()
    patch.squeeze_()
    patch = patch.numpy()
    for i in range(0, num_ID_per_batch):
        if random.random() > 0.5:
            nb_rot = np.random.randint(1, 4)
            patch[2*i] = np.rot90(patch[2*i], nb_rot)
            patch[2*i+1] = np.rot90(patch[2*i + 1], nb_rot)


        if random.random() > 0.5:
            patch[2 * i] = np.flipud(patch[2 * i])
            patch[2 * i + 1] = np.flipud(patch[2 * i + 1])

        # if random.random() > 0.5:
        #     patch[2 * i] = np.fliplr(patch[2*i])
        #     patch[2 * i + 1] = np.fliplr(patch[2*i + 1])



    patch = torch.from_numpy(patch)
    patch.unsqueeze_(1)
    return patch

def cal_fpr95(desc,pointID,pair_index):
    dist = desc[pair_index[:, 0],:] - desc[pair_index[:, 1],:]
    dist.pow_(2)
    dist = torch.sqrt(torch.sum(dist,1))
    pairSim = pointID[pair_index[:, 0]] - pointID[pair_index[:, 1]]
    pairSim = torch.Tensor(pairSim)
    dist_pos = dist[pairSim == 0]
    dist_neg = dist[pairSim != 0]
    dist_pos, indice = torch.sort(dist_pos)
    loc_thr = int(np.ceil(dist_pos.numel() * 0.95))
    thr = dist_pos[loc_thr]
    fpr95 = float(dist_neg.le(thr).sum())/dist_neg.numel()
    return fpr95


def mae_loss(img1, img2):
    return F.l1_loss(img1, img2, reduction='sum').cuda()

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406]).cuda()
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225]).cuda()
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        # self.slice4 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            self.slice1.cuda()
        for x in range(2, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
            self.slice2.cuda()
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
            self.slice3.cuda()
        # for x in range(16, 23):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        # model에서 이미 tanh 결과값에 처리 했으니 생략합시다
        # out = x/2 + 0.5
        out = x
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def forward(self, im1_, im2_):
        EPS = 1e-7
        mae_loss = F.l1_loss(im1_, im2_, reduction='mean').cuda()
        im = torch.cat([im1_,im2_], 0)
        im = self.normalize(im)  # normalize input
        im1, im2 = torch.chunk(im, 2, dim=0)

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        # f = self.slice4(f)
        # feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats:  # use relu3_3 features only
            loss = F.mse_loss(f1, f2, reduction='mean').cuda()
            losses += [loss]
        return sum(losses) + mae_loss
    
        # return F.l1_loss(im1, im2, reduction='sum').cuda()

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = torchvision.models.vgg16(pretrained=True).features
        blocks = []
        blocks.append(vgg_features[:4])
        blocks.append(vgg_features[4:9])
        blocks.append(vgg_features[9:16])
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y) + torch.nn.functional.l1_loss(input-target)
        return loss


# 아래 코드는 https://milkclouds.work/pytorch-ddp-baseline/ 참고
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val=0, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get(self):
        self.count += 1
        return self.count
    
class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)