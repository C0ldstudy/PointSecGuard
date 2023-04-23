import __init__
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from config import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models
import logging
from attacks import NU_attack_exp, tar_NU_attack_exp, NB_attack_exp, tar_NB_attack_exp

import time
import os
import os.path as osp

def main():
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader...')
    test_dataset = GeoData.S3DIS(opt.data_dir, 5, train=False, pre_transform=T.NormalizeScale())
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = test_loader.dataset.num_classes
    if opt.no_clutter:
        opt.n_classes -= 1
    logging.info('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info('===> Start Evaluation ...')
    attack(model, test_loader, opt)

def attack(model, test_loader, opt):
    adv_ious = []
    avg_l2_dis = []
    if opt.attack == 'random':
        random_noise(model,test_loader, opt)
    elif opt.attack == 'NU_attack':
        NU_attack_exp(model, test_loader, opt)
    elif opt.attack == 'tar_NU_attack':
        tar_NU_attack_exp(model, test_loader, opt)
    elif opt.attack == 'NB_attack':
        NB_attack_exp(model, test_loader, opt)
    elif opt.attack == 'tar_NB_attack':
        tar_NB_attack_exp(model, test_loader, opt)

def random_noise(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))
    adv_Is = np.empty((len(loader), opt.n_classes))
    adv_Us = np.empty((len(loader), opt.n_classes))
    dis = np.empty(len(loader))
    mious = np.empty(len(loader))
    adv_mious = np.empty(len(loader))
    acc = np.empty(len(loader))
    adv_acc = np.empty(len(loader))
    save_path = opt.res_dir + '/random/'
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
        f.write("index\tl2dis\tadv_acc\tacc\tadv_miou\miou\n")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data_result = 1.0 # the range of the random noise
            data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y

            out = model(inputs)
            pred = out.max(dim=1)[1]
            pred_label = pred.detach().cpu().numpy()
            acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096

            noise = torch.empty_like(inputs[:, 3:6]).uniform_(0, 1)
            noise = noise / torch.norm(noise) * float(data_result)
            adv_images = inputs.clone()
            adv_images[:, 3:6] += noise
            adv_out = model(adv_images)
            adv_pred = adv_out.max(dim=1)[1]
            adv_pred_label = adv_pred.detach().cpu().numpy()
            adv_acc[i] = adv_pred.eq(gt.view_as(adv_pred)).sum().item() / 4096
            dis[i] = torch.dist(inputs, adv_images, p=2)

            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()
            adv_pred_np = adv_pred.cpu().squeeze(0).numpy()

            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                Is[i, cl] = I
                Us[i, cl] = U
                adv_cur_pred_mask = (adv_pred_np == cl)
                adv_I = np.sum(np.logical_and(adv_cur_pred_mask, cur_gt_mask), dtype=np.    float32)
                adv_U = np.sum(np.logical_or(adv_cur_pred_mask, cur_gt_mask), dtype=np. float32)
                adv_Is[i, cl] = adv_I
                adv_Us[i, cl] = adv_U

            mious[i] = np.divide(np.sum(Is[i], 0), np.sum(Us[i], 0))
            mious[np.isnan(mious[i])] = 1
            adv_mious[i] = np.divide(np.sum(adv_Is[i], 0), np.sum(adv_Us[i], 0))
            adv_mious[np.isnan(adv_mious[i])] = 1
            with open(osp.join(save_path, 'log_'+timestamp+'.txt'), 'a+') as f:
                f.write("%d\t%.3f\t%.5f\t\t%.5f\t%.5f\t%.5f\n" % (i, dis[i], adv_acc[i], acc[i], np.mean(adv_mious[i]), np.mean(mious[i])))

if __name__ == '__main__':
    main()
