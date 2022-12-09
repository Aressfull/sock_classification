import os
import cv2
import copy
import h5py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import gen_args
from models import CNNet
from videomae_data import ActionDataset
from utils import plot_confusion_matrix, accuracy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score
#from sklearn import cross_validation

from transformers import VideoMAEForVideoClassification, VideoMAEForPreTraining, VideoMAEConfig
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor

# os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,6,7"
args = gen_args("VideoMae4layerPretrained_frame_sensor")

print(args)

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

if torch.cuda.device_count() > 1:
    multiple_gpu = True;

config = VideoMAEConfig()
config.image_size = 32
config.patch_size = 4
config.num_channels = 2
config.num_frames = args.input_window_size
config.tubelet_size = args.tubelet_size
config.num_labels = args.n_obj
config.num_hidden_layers = args.num_hidden_layers
config.hidden_size = args.hidden_size
config.num_attention_heads = args.encoder_att_heads

pretrain_path = ('./saved_models/%ddepth-%slr-%sL2-%ddims-%dheads-%dbatch-%depoch-%s-%dtube-%dpatch-%smask-pretrained-videomae-seed10-v2' % 
                         (config.num_hidden_layers, str(args.lr).split('.')[1], str(args.weight_decay).split('.')[1],
                              args.hidden_size, args.encoder_att_heads,
                              args.batch_size, args.n_pretrain, args.mask_type,args.tubelet_size, args.patch_size,
                              str(f'{args.mask_ratio:.2f}').split('.')[1]))
if args.pretrain:
    finetune_path = ('./saved_models/%ddepth-%slr-%sL2-%ddims-%dheads-%dbatch-%depoch-%dpretrain-%s-%dtube-%dpatch-%smask-finetuned-videomae-seed10-v2' % 
                         (config.num_hidden_layers, str(args.lr).split('.')[1], str(args.weight_decay).split('.')[1],
                              args.hidden_size, args.encoder_att_heads,
                              args.batch_size, args.n_epoch, args.n_pretrain,args.mask_type,args.tubelet_size, args.patch_size,
                              str(f'{args.mask_ratio:.2f}').split('.')[1]))
else:
    finetune_path = ('./saved_models/%ddepth-%slr-%sL2-%ddims-%dheads-%dbatch-%depoch-%dpretrain-%s-%dtube-%dpatch-%smask-finetuned-nopretrain-videomae-seed10-v2' % 
                     (config.num_hidden_layers, str(args.lr).split('.')[1], str(args.weight_decay).split('.')[1],
                          args.hidden_size, args.encoder_att_heads,
                          args.batch_size, args.n_epoch, args.n_pretrain,args.mask_type,args.tubelet_size, args.patch_size,
                          str(f'{args.mask_ratio:.2f}').split('.')[1]))
# pretrain_path = './6layer90mask005lr200epochtubepretrained-001L2-30temporal-emb-videomae-seed10'
# finetune_path = './6layer90mask005lr200epochtubefinetuned-001L2-30temporal-emb-videomae-seed10'
print(pretrain_path)
print(finetune_path)
    
'''
dataset
'''
datasets = {}
dataloaders = {}
for phase in ['train', 'valid', 'test']:
    datasets[phase] = ActionDataset(args, phase=phase)
    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)
    
model = VideoMAEForPreTraining(config)

if multiple_gpu:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])

model = model.to(device)
    
pretraining = args.pretrain

if pretraining:
    pretrain_opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_itr = args.n_pretrain
    mask_ratio = args.mask_ratio
    for itr in range(0,n_itr):
        phases = ['train']
        for phase in phases:
            running_loss = 0.0
            count = 0
            batch_num = 0
            for data in tqdm(dataloaders[phase]):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.to(device), labels.to(device)

                num_patches_per_frame = (model.module.config.image_size // model.module.config.patch_size) ** 2
                num_cubes = (model.module.config.num_frames // model.module.config.tubelet_size)
                
                ## OLD MASKING
#                 seq_length = num_cubes * num_patches_per_frame
#                 bool_masked_pos = np.ones(seq_length)
#                 mask_num = math.ceil(seq_length * mask_ratio)
#                 mask = np.random.choice(seq_length, mask_num, replace=False)
#                 bool_masked_pos[mask] = 0

                ## TUBE MASKING (videomae论文 example d)
                if args.mask_type == 'tube':
                    frame_masked_pos = np.zeros(num_patches_per_frame)
                    mask_num = math.ceil(num_patches_per_frame*mask_ratio)
                    mask = np.random.choice(num_patches_per_frame,mask_num,replace=False)
                    frame_masked_pos[mask] = 1
                    bool_masked_pos = np.tile(frame_masked_pos,num_cubes)

                ## RANDOM MASKING
                if args.mask_type == 'random':
                    bool_masked_pos = []
                    for i in range(num_cubes):
                        frame_masked_pos = np.zeros(num_patches_per_frame)
                        mask_num = math.ceil(num_patches_per_frame*mask_ratio)
                        mask = np.random.choice(num_patches_per_frame,mask_num,replace=False)
                        frame_masked_pos[mask] = 1
                        bool_masked_pos.append(frame_masked_pos)
                    bool_masked_pos = np.array(bool_masked_pos).flatten()

                # Torch and bool cast, extra dimension added for concatenation
                bool_masked_pos = torch.as_tensor(bool_masked_pos).bool().unsqueeze(0)
                bool_masked_pos = torch.cat([bool_masked_pos for _ in range(inputs.shape[0])])
                bool_masked_pos = bool_masked_pos.to(device)

                outputs = model(inputs, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss
                pretrain_opt.zero_grad()
                loss.mean().backward()
                pretrain_opt.step()
                if count > 0 and count % args.log_per_iter == 0:    # print every 3000 inputs 
                    print('[%d/%d] %s loss: %.4f (%.4f)' % (itr+1, n_itr, phase,
                        loss.mean().item(), running_loss/batch_num))
                running_loss += loss.mean().item()
                count += inputs.shape[0]
                batch_num += 1

            running_loss = running_loss / len(dataloaders[phase])

            print('[%d/%d] %s loss: %.4f (%.4f)' % (itr+1, n_itr, phase,
                        loss.mean().item(), running_loss))

#     torch.save(model.state_dict(), './pretrained-videomae.pt')
    model.module.save_pretrained(pretrain_path)

if pretraining:
    model = VideoMAEForVideoClassification.from_pretrained(pretrain_path,config=config)
else:
    model = VideoMAEForVideoClassification(config=config)

# model = VideoMAEForVideoClassification.from_pretrained('./finetuned20-videomae',config=config)
# model.cuda()
if multiple_gpu:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
model = model.to(device)

# if use_gpu:
#     model = model.to(device)

criterion = nn.CrossEntropyLoss()
# criterion.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

best_top1 = {}
best_top3 = {}
for phase in ['valid', 'test']:
    best_top1[phase] = 0.
    best_top3[phase] = 0.
    
for epoch in range(args.n_epoch):  # loop over the dataset multiple times
    phases = ['train', 'valid', 'test']
    for phase in phases:
        count = 0
        batch_num = 0
        model.train(phase == 'train')

        running_loss = 0.0
        running_top1, running_top3 = 0, 0
        pred_rec, true_rec = [], []

#         bar = ProgressBar(maxval=len(dataloaders[phase]))

        for data in tqdm(dataloaders[phase]):
            inputs, labels = data
            #inputs:tensor:32*(2*window)*32*32(B,T,H,W) 在data.py里将左右脚的张量放到一个维度下
            #labels:45
            #print(inputs.size())
            if use_gpu:
                inputs, labels = inputs.to(device), labels.to(device) #inputs.cuda(), labels.cuda() 

            if phase == 'test':
                torch.cuda.empty_cache()
                with torch.no_grad():
                    outputs = model_best(inputs).logits
                    loss = criterion(outputs, labels)
                    loss = loss.mean()
            elif phase == 'valid':
                torch.cuda.empty_cache()
                with torch.no_grad():
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                    loss = loss.mean()
            else:
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                loss = loss.mean()

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#             print(loss.item())
            # print statistics
            running_loss += loss.item()

            # record the prediction
            _, predicted = torch.max(outputs.data, 1)
            pred_rec.append(predicted.data.cpu().numpy().astype(np.int64))
            true_rec.append(labels.data.cpu().numpy().astype(np.int64))

            # record the topk accuracy
            top1, top3 = accuracy(outputs, labels, topk=(1, 3))
            running_top1 += top1
            running_top3 += top3
            count += inputs.shape[0]
            batch_num += 1
            if count > 0 and count % args.log_per_iter == 0:    # print every 3000 inputs 
                print('[%d/%d] loss: %.4f (%.4f), acc: top1 %.4f (%.4f) top3 %.4f (%.4f)' % (
                    epoch+1, args.n_epoch,
                    loss.item(), running_loss /batch_num,
                    top1, running_top1 /batch_num,
                    top3, running_top3 /batch_num))

        running_loss = running_loss / len(dataloaders[phase])
        top1_cur = running_top1 / len(dataloaders[phase])
        top3_cur = running_top3 / len(dataloaders[phase])


        if phase in ['valid', 'test']:
            # scheduler.step(running_loss)

            if top1_cur >= best_top1[phase]:
                best_top1[phase] = top1_cur
                best_top3[phase] = top3_cur

                if phase == 'valid':
                    model_best = copy.deepcopy(model)

            print('[%d, %s] loss: %.4f, acc: top1 %.4f top3 %.4f, best_acc: top1 %.4f top3 %.4f' % (
                epoch, phase, running_loss, top1_cur, top3_cur, best_top1[phase], best_top3[phase]))

            pred_rec = np.concatenate(pred_rec)
            true_rec = np.concatenate(true_rec)
            
            f1 = f1_score(true_rec, pred_rec, average='macro')
            print(f1)
            #print(pred_rec)
            #print(true_rec)
            y_test_all = label_binarize(true_rec, classes=[0,1,2,3,4,5,6,7,8])
            y_score_all=label_binarize(pred_rec, classes=[0,1,2,3,4,5,6,7,8])
            
            fpr,tpr,threshold = roc_curve(y_test_all[:,0], y_score_all[:,0])
            roc_auc = auc(fpr,tpr)
#             plt.figure()
#             lw = 2
#             plt.figure(figsize=(10,10))
#             plt.plot(fpr, tpr, color='darkorange',
#                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
#             plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver operating characteristic example')
#             plt.legend(loc="lower right")
#             plt.show()
#             plt.close()
            
#             plot_confusion_matrix(true_rec, pred_rec, args.object_list)

#             plt.savefig(os.path.join(args.rec_path, '%s_%d.pdf' % (phase, epoch)))
#             plt.close()

        else:
            print('[%d, %s] loss: %.4f, acc: top1 %.4f top3 %.4f' % (
                epoch, phase, running_loss, top1_cur, top3_cur))

    print()

plot_confusion_matrix(true_rec, pred_rec, args.object_list)
plt.savefig(finetune_path + '.pdf')
plt.close()
model_best.module.save_pretrained(finetune_path)

running_loss = 0.0
running_top1, running_top3 = 0, 0
pred_rec, true_rec = [], []
output_rec, label_rec = [], []

for data in tqdm(dataloaders['test']):
    inputs, labels = data
    if use_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()
    model_best.eval()
    with torch.no_grad():
        outputs = model_best(inputs).logits
        loss = criterion(outputs, labels)
        loss = loss.mean()

    _, predicted = torch.max(outputs.data, 1)

    pred_rec.append(predicted.data.cpu().numpy().astype(np.int64))
    true_rec.append(labels.data.cpu().numpy().astype(np.int64))

    output_rec.append(outputs.data.cpu().numpy())
    label_rec.append(labels.data.cpu().numpy())

    loss = criterion(outputs, labels)
    running_loss += loss.item()
    top1, top3 = accuracy(outputs, labels, topk=(1, 3))
    running_top1 += top1
    running_top3 += top3

running_loss = running_loss / len(dataloaders['test'])
top1_cur = running_top1 / len(dataloaders['test'])
top3_cur = running_top3 / len(dataloaders['test'])

pred_rec = np.concatenate(pred_rec)
true_rec = np.concatenate(true_rec)
f1 = f1_score(true_rec, pred_rec, average='macro')

y_test_all = label_binarize(true_rec, classes=[0,1,2,3,4,5,6,7,8])
y_score_all=label_binarize(pred_rec, classes=[0,1,2,3,4,5,6,7,8])


print('[Test] loss: %.4f | top1 acc: %.4f | top3 acc: %.4f | f1 score %.4f ' % (
            running_loss, top1_cur, top3_cur, f1))
