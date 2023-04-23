import torch
import numpy as np
from tqdm import tqdm
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import logging
import os
import os.path as osp
import math
import gc
import time, fnmatch, shutil
import datetime

import sys
file_path = './attacks/torchattacks'
sys.path.append(os.path.dirname(file_path))
import torchattacks

# ceiling, floor, wall, beam, column, window and door
g_class2color = {
                    '0': [1,  0,  0], # red ceiling
                    '1': [1,  1,  0], # yellow floor
                    '2': [0, 1, 0], # green wall
                    '3': [0, 1, 1], # cyan beam
                    '4': [0, 0, 1], # blue column
                    '5': [1, 0, 1], # pink window
                    '6': [1, 0.4, 0.4], # coral door

                    # table, chair, sofa, bookcase and board
                    '7': [0.5,  0.5,  0],
                    '8': [0.5, 1, 0.5],
                    '9': [0.5, 1, 1],
                    '10': [0.5, 0.5, 1],
                    '11': [0.4, 1, 0.4],
                    '12': [0.5, 0.5, 0.5],# black cluster

                    '13': [0, 0, 0] #
}

def NU_attack_exp(model, test_loader, opt):
    Is = np.empty((len(test_loader), opt.n_classes))
    Us = np.empty((len(test_loader), opt.n_classes))
    adv_Is = np.empty((len(test_loader), opt.n_classes))
    adv_Us = np.empty((len(test_loader), opt.n_classes))
    dis = np.empty(len(test_loader))
    other_acc = np.empty(len(test_loader))
    acc = np.empty(len(test_loader))

    mious = np.empty(len(test_loader))
    adv_mious = np.empty(len(test_loader))
    num = 548
    save_path = opt.res_dir + '/NU_attack_exp/'
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'w') as f:
        f.write("index\tL2_dis\tother_acc\tacc\tadv_miou\tmiou\n")

    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        batch_size = inputs.shape[0]

        gt = data.y
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096

        # attack
        attack = torchattacks.NU_attack(model, c=1e-1, kappa=0, steps=1000, lr=0.1)
        adv_images = attack(inputs, gt)
        adv_out = model(adv_images)
        adv_pred = adv_out.max(dim=1)[1]
        other_acc[i] = adv_pred.eq(gt.view_as(adv_pred)).sum().item() / 4096
        dis[i] = torch.dist(inputs, adv_images, p=2) / batch_size

        pred_np = pred.cpu().squeeze(0).numpy()
        adv_pred_np = adv_pred.cpu().squeeze(0).numpy()
        target_np = gt.cpu().squeeze(0).numpy()

        for cl in range(opt.n_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            Is[i, cl] = I
            Us[i, cl] = U

            adv_cur_pred_mask = (adv_pred_np == cl)
            adv_I = np.sum(np.logical_and(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_U = np.sum(np.logical_or(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_Is[i, cl] = adv_I
            adv_Us[i, cl] = adv_U

        mious[i] = np.divide(np.sum(Is[i], 0), np.sum(Us[i], 0))
        adv_mious[i] = np.divide(np.sum(adv_Is[i], 0), np.sum(adv_Us[i], 0))
        with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
            f.write("%d\t%.3f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (i, dis[i], other_acc[i], acc[i], np.mean(adv_mious[i]), np.mean(mious[i])))
    return


def NB_attack_exp(model, test_loader, opt):
    Is = np.empty((len(test_loader), opt.n_classes))
    Us = np.empty((len(test_loader), opt.n_classes))
    adv_Is = np.empty((len(test_loader), opt.n_classes))
    adv_Us = np.empty((len(test_loader), opt.n_classes))
    dis = np.empty(len(test_loader))
    other_acc = np.empty(len(test_loader))
    acc = np.empty(len(test_loader))
    mious = np.empty(len(test_loader))
    adv_mious = np.empty(len(test_loader))
    save_path = opt.res_dir + '/NB_attack_exp/'
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'w') as f:
        f.write("index\tL2_dis\tother_acc\tacc\tadv_miou\tmiou\n")
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        batch_size = inputs.shape[0]
        gt = data.y
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096
        attack = torchattacks.NB_attack(model, eps=0.3, alpha=2/255, iters=50)
        adv_images = attack(inputs, gt)
        adv_out = model(adv_images)
        adv_pred = adv_out.max(dim=1)[1]
        other_acc[i] = adv_pred.eq(gt.view_as(adv_pred)).sum().item() / 4096

        dis[i] = torch.dist(inputs, adv_images, p=2) / batch_size
        pred_np = pred.cpu().squeeze(0).numpy()
        adv_pred_np = adv_pred.cpu().squeeze(0).numpy()
        target_np = gt.cpu().squeeze(0).numpy()

        for cl in range(opt.n_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            Is[i, cl] = I
            Us[i, cl] = U

            adv_cur_pred_mask = (adv_pred_np == cl)
            adv_I = np.sum(np.logical_and(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_U = np.sum(np.logical_or(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_Is[i, cl] = adv_I
            adv_Us[i, cl] = adv_U

        mious[i] = np.divide(np.sum(Is[i], 0), np.sum(Us[i], 0))
        adv_mious[i] = np.divide(np.sum(adv_Is[i], 0), np.sum(adv_Us[i], 0))
        with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
            f.write("%d\t%.3f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (i, dis[i], other_acc[i], acc[i], np.mean(adv_mious[i]), np.mean(mious[i])))
    return


def tar_NU_attack_exp(model, test_loader, opt):
    target = opt.target
    ori = opt.origin
    Is = np.empty((len(test_loader), opt.n_classes))
    Us = np.empty((len(test_loader), opt.n_classes))
    adv_Is = np.empty((len(test_loader), opt.n_classes))
    adv_Us = np.empty((len(test_loader), opt.n_classes))
    dis = np.empty(len(test_loader))
    other_acc = np.empty(len(test_loader))
    acc = np.empty(len(test_loader))
    target_acc = np.empty(len(test_loader))
    mious = np.empty(len(test_loader))
    adv_mious = np.empty(len(test_loader))
    save_path = opt.res_dir + '/tar_NU_attack_exp/'+ opt.att_type + str(target) + '_' + str(ori) + '/'
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(osp.join(save_path, 'log_' + timestamp + '.txt'), 'w') as f:
        f.write("left_ratio=" + str(opt.left_ratio) + '\n')
    with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
        f.write("index\tcount\tL2_dis\ttarget_acc\tother_acc\tacc\tadv_miou\tmiou\n")
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        batch_size = inputs.shape[0]
        gt = data.y
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096

        # attack
        mask = (gt == ori)[0]
        count = (mask == True).sum()
        left_num = int(count * opt.left_ratio)

        if (count <= 500):
            continue
        temp_acc = pred[:, mask].eq(gt[:, mask].view_as(pred[:, mask])).sum().item() / mask.sum().item() # check the masked points' accuracy
        if (temp_acc < 0.5):
            continue
        target_labels = torch.full(gt.shape, target).to(opt.device)
        attack = torchattacks.tar_NU_attack(model, c=1e-1, kappa=0, steps=1000, lr=0.1, target=target, mask=mask)
        adv_images = attack(inputs, gt)
        adv_out = model(adv_images)
        adv_pred = adv_out.max(dim=1)[1]
        other_acc[i] = adv_pred[0, ~mask].eq(gt[0, ~mask].view_as(adv_pred[0, ~mask])).sum().item() / 4096
        target_acc[i] = adv_pred[0, mask].eq(target_labels[0, mask].view_as(adv_pred[0, mask])).sum().item() / count
        dis[i] = torch.dist(inputs, adv_images, p=2) / batch_size
        pred_np = pred.cpu().squeeze(0).numpy()
        adv_pred_np = adv_pred.cpu().squeeze(0).numpy()
        target_np = gt.cpu().squeeze(0).numpy()

        for cl in range(opt.n_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            Is[i, cl] = I
            Us[i, cl] = U
            adv_cur_pred_mask = (adv_pred_np == cl)
            adv_I = np.sum(np.logical_and(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_U = np.sum(np.logical_or(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_Is[i, cl] = adv_I
            adv_Us[i, cl] = adv_U

        mious[i] = np.divide(np.sum(Is[i], 0), np.sum(Us[i], 0))
        adv_mious[i] = np.divide(np.sum(adv_Is[i], 0), np.sum(adv_Us[i], 0))
        with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
            f.write("%d\t%d\t%.3f\t%.3f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (i, count,dis[i], target_acc[i], other_acc[i], acc[i], np.mean(adv_mious[i]), np.mean(mious[i])))
    return

def tar_NB_attack_exp(model, test_loader, opt):
    target = opt.target
    ori = opt.origin
    Is = np.empty((len(test_loader), opt.n_classes))
    Us = np.empty((len(test_loader), opt.n_classes))
    adv_Is = np.empty((len(test_loader), opt.n_classes))
    adv_Us = np.empty((len(test_loader), opt.n_classes))
    dis = np.empty(len(test_loader))
    other_acc = np.empty(len(test_loader))
    acc = np.empty(len(test_loader))
    target_acc = np.empty(len(test_loader))
    mious = np.empty(len(test_loader))
    adv_mious = np.empty(len(test_loader))
    other_mious = np.empty(len(test_loader))
    save_path = opt.res_dir + '/tar_NB_attack/'+ opt.att_type + str(target) + '_' + str(ori) + '/'
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(osp.join(save_path, 'log_' + timestamp + '.txt'), 'w') as f:
        f.write("left_ratio=" + str(opt.left_ratio) + '\n')
    with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
        f.write("index\tcount\tL2_dis\ttarget_acc\tother_acc\tacc\tother_miou\tmiou\n")
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        batch_size = inputs.shape[0]

        gt = data.y
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096

        # attack
        mask = (gt == ori)[0]
        count = (mask == True).sum()
        left_num = int(count * opt.left_ratio)

        if (count <= 500):
            continue
        temp_acc = pred[:, mask].eq(gt[:, mask].view_as(pred[:, mask])).sum().item() / mask.sum().item() # check the masked points' accuracy
        print(count.item(), temp_acc)
        if (temp_acc < 0.5):
            continue

        target_labels = torch.full(gt.shape, target).to(opt.device)
        attack = torchattacks.tar_NB_attack(model, eps=0.4, alpha=0.04, iters=50, target=target, mask=mask)
        adv_images = attack(inputs, gt)
        adv_out = model(adv_images)
        adv_pred = adv_out.max(dim=1)[1]
        other_acc[i] = adv_pred[0, ~mask].eq(gt[0, ~mask].view_as(adv_pred[0, ~mask])).sum().item() / 4096
        target_acc[i] = adv_pred[0, mask].eq(target_labels[0, mask].view_as(adv_pred[0, mask])).sum().item() / count
        dis[i] = torch.dist(inputs, adv_images, p=2) / batch_size

        pred_np = pred.cpu().squeeze(0).numpy()
        adv_pred_np = adv_pred.cpu().squeeze(0).numpy()
        target_np = gt.cpu().squeeze(0).numpy()

        for cl in range(opt.n_classes):
            cur_gt_mask = (target_np == cl)
            cur_pred_mask = (pred_np == cl)
            I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            Is[i, cl] = I
            Us[i, cl] = U

            if cl == ori:
                continue
            adv_cur_pred_mask = (adv_pred_np == cl)
            adv_I = np.sum(np.logical_and(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_U = np.sum(np.logical_or(adv_cur_pred_mask, cur_gt_mask), dtype=np.float32)
            adv_Is[i, cl] = adv_I
            adv_Us[i, cl] = adv_U

        mious[i] = np.divide(np.sum(Is[i], 0), np.sum(Us[i], 0))
        other_mious[i] = np.divide(np.sum(adv_Is[i], 0), np.sum(adv_Us[i], 0))
        with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
            f.write("%d\t%d\t%.3f\t%.3f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (i, count,dis[i], target_acc[i], other_acc[i], acc[i], np.mean(other_mious[i]), np.mean(mious[i])))

    return
