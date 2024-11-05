#!/usr/bin/env python3

from distutils.errors import DistutilsModuleError
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from srm_filter_kernel import all_normalized_hpf_list

from MPNCOV.python import MPNCOV

IMAGE_SIZE = 256
BATCH_SIZE = 16
DIM = 32
LR = 0.01
WEIGHT_DECAY = 5e-4

TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1

OUTPUT_PATH = Path(__file__).stem


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


def build_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list:
        row_1 = int((5 - hpf_item.shape[0]) / 2)
        row_2 = int((5 - hpf_item.shape[0]) - row_1)
        col_1 = int((5 - hpf_item.shape[1]) / 2)
        col_2 = int((5 - hpf_item.shape[1]) - col_1)
        hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
        filters.append(hpf_item)
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5
        for k in range(2):
          for j in range(2):
                kern = cv2.getGaborKernel((ksize[0], ksize[0]), sigma[k], theta, sigma[k] / 0.56, 0.5, phi[j],
                                          ktype=cv2.CV_32F)
                filters.append(kern)
    return filters


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        filt_list = build_filters()

        hpf_weight = nn.Parameter(torch.Tensor(filt_list).view(62, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 62, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        self.tlu = TLU(5.0)

    def forward(self, input):
        output = self.hpf(input)
        output = self.tlu(output)

        return output  



class Conv_PDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.5):

        super(Conv_PDC, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,8), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).cuda()
        conv_weight = torch.cat((self.conv.weight[:,:,:,0], self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], self.conv.weight[:,:,:,5], self.conv.weight[:,:,:,6], self.conv.weight[:,:,:,7]), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)
        
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class pdcblock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock1, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1), 
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out


class pdcblock2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(pdcblock2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(

                Conv_PDC(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                Conv_PDC(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),

                nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
                )

    def forward(self,x):
        out=self.basic(x)
        return out


class DCONV(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DCONV, self).__init__()

        self.depth = Conv_PDC(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1)

    def forward(self, x):
        result = self.depth(x)
        return result
    

class EnhanceMHSA(nn.Module):
    def __init__(self, input_size, channels, d_k, d_v, stride, heads, dropout):
        super(EnhanceMHSA, self).__init__()
        self.dwconv_k = DCONV(channels, channels, stride = stride)
        self.dwconv_v = DCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2
        self.B = nn.Parameter(torch.Tensor(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)


    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        # x_reshape = nn.LayerNorm(c).cuda()(x_reshape)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        result = result + x
        return result


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.relu = nn.ReLU()
    self.group1 = HPF()

    self.group2 = pdcblock1(186,64)
    self.group3 = pdcblock2(64,128)
    self.group4 = pdcblock2(128,256)

    self.entrans = EnhanceMHSA(32, 256, 64, 64, 2, 2, 0.0)

    self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2048)
    self.fc2 = nn.Linear(2048, 2)


  def forward(self, input):
    output = input
    
    output_y = output[:, 0, :, :]
    output_u = output[:, 1, :, :] 
    output_v = output[:, 2, :, :] 
    out_y = output_y.unsqueeze(1)
    out_u = output_u.unsqueeze(1)
    out_v = output_v.unsqueeze(1)
    y = self.group1(out_y)
    u = self.group1(out_u)
    v = self.group1(out_v)
    output = torch.cat([y, u, v], dim=1)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)

    output = self.entrans(output)

    output = MPNCOV.CovpoolLayer(output)
    output = MPNCOV.SqrtmLayer(output, 5)
    output = MPNCOV.TriuvecLayer(output)
    output = self.gap(output)
    output = output.view(output.size(0), -1)

    output = self.fc1(output)
    output = self.fc2(output)

    return output


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
    batch_time = AverageMeter()  # ONE EPOCH TRAIN TIME
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, sample in enumerate(train_loader):

        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']

        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()

        end = time.time()

        output = model(data)  # FP

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)

        losses.update(loss.item(), data.size(0))

        loss.backward()  # BP
        optimizer.step()

        batch_time.update(time.time() - end)  # BATCH TIME = BATCH BP+FP
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)
            # data, label = data.cuda(), label.cuda()

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
      }
    torch.save(all_state, PARAMS_PATH1)

    if accuracy > best_acc and epoch > TMP:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[2, 3]).copy()  

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = data.astype(np.float32)
        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform

        self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.ppm'
        self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.ppm'

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        cover_path = self.bossbase_cover_path.format(file_index)
        stego_path = self.bossbase_stego_path.format(file_index)

        cover_data = cv2.imread(cover_path, -1)
        cover_data = cv2.cvtColor(cover_data, cv2.COLOR_BGR2RGB)
        cover_data = np.transpose(cover_data, (2, 0, 1))
        stego_data = cv2.imread(stego_path, -1)
        stego_data = cv2.cvtColor(stego_data, cv2.COLOR_BGR2RGB)
        stego_data = np.transpose(stego_data, (2, 0, 1))

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):

    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 2, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE
    TIMES = args.times


    COVER_DIR = '/data/ALASKA_v2_TIFF_256_COLOR_40000_cover'
    STEGO_DIR = '/data/ALASKA_v2_TIFF_256_COLOR_40000_stego'

    TRAIN_INDEX_PATH = '/index_list/train_index_24000.npy'
    VALID_INDEX_PATH = '/index_list/valid_index_4000.npy'
    TEST_INDEX_PATH = '/index_list/test_index_12000.npy'

    LOAD_RATE = float(EMBEDDING_RATE) + 0.1
    LOAD_RATE = round(LOAD_RATE, 1)

    PARAMS_NAME = '{}-{}-{}-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    PARAMS_NAME1 = '{}-{}-{}-process-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES,LR)
    LOG_NAME = '{}-{}-{}-model_log-{}-lr={}.log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, LR)
    #statePath='./log/'+PARAMS_NAME1

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    setLogger(LOG_PATH, mode='w')

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    train_dataset = MyDataset(TRAIN_INDEX_PATH, COVER_DIR, STEGO_DIR, train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, COVER_DIR, STEGO_DIR, eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, COVER_DIR, STEGO_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    model = Net().to(device)
    #model = torch.nn.DataParallel(model)
    #model = model.cuda()
    model.apply(initWeights)

    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

    EPOCHS = 200
    DECAY_EPOCH = [100, 150]
    TMP = 150
    

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    best_acc = 0.0
    for epoch in range(startEpoch, EPOCHS + 1):
        scheduler.step()

        train(model, device, train_loader, optimizer, epoch)

        if epoch % EVAL_PRINT_FREQUENCY == 0:
            adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)

    logging.info('\nTest set accuracy: \n')

    # load best parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    adjust_bn_stats(model, device, train_loader)
    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1,TMP)


def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--DATASET_INDEX',
        help='Path for loading dataset',
        type=str,
        default='1'
    )

    parser.add_argument(
        '-alg',
        '--STEGANOGRAPHY',
        help='embedding_algorithm',
        type=str,
        choices=['HILL-CMDC', 'SUNIWARD-CMDC', 'HILL-GINA', 'SUNIWARD-GINA', 'HILL-ACMP', 'SUNIWARD-ACMP'],
        required=True
    )

    parser.add_argument(
        '-rate',
        '--EMBEDDING_RATE',
        help='embedding_rate',
        type=str,
        choices=[ '0.2', '0.3', '0.4'],
        required=True
    )

    parser.add_argument(
        '-g',
        '--gpuNum',
        help='Determine which gpu to use',
        type=str,
        choices=['0', '1', '2', '3'],
        required=True
    )

    parser.add_argument(
        '-t',
        '--times',
        help='Determine which gpu to use',
        type=str,
        # required=True
        default=''
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = myParseArgs()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)


