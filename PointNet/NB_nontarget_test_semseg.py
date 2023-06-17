"""
python nontarget_test_semseg.py --log_dir pointnet2_sem_seg --test_area 5
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import time

file_path = './attacks/torchattacks'
sys.path.append(os.path.dirname(file_path))
import torchattacks
import time
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--data_path', type=str, default='datasets/stanford_indoor3d/', help='path to the dataset')
    parser.add_argument('--log_dir', type=str, default='nontar_pointnet2', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    timestr = time.strftime("%Y%m%d_%H%M%S")
    visual_dir = experiment_dir + '/visual_' + timestr + '/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(experiment_dir+'/logs')
    log_dir.mkdir(parents=True, exist_ok=True)


    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    root = args.data_path
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    log_string("The number of test data is: %d" %  len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    print(model_name)
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    with open(os.path.join(experiment_dir, 'log_'+timestamp+'.txt'), 'w') as f:
        f.write("index\tL2_dis\tadv_acc\tacc\tadv_miou\tmiou\n")

    if True:
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        adv_total_correct_class = [0 for _ in range(NUM_CLASSES)]
        adv_total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')
        for batch_idx in range(num_batches):
            print("visualize [%d/%d] %s ..." % (batch_idx+1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            adv_total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            adv_total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.xyzrgb'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.xyzrgb'), 'w')
                fout_raw = open(os.path.join(visual_dir, scene_id[batch_idx] + '_raw.xyzrgb'), 'w')
                fout_adv_raw = open(os.path.join(visual_dir, scene_id[batch_idx] + '_adv_raw.xyzrgb'), 'w')
                fout_adv_pred = open(os.path.join(visual_dir, scene_id[batch_idx] + '_adv_pred.xyzrgb'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            adv_vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            adv_whole_scene = np.zeros(whole_scene_data.shape)
            print(whole_scene_data.shape)

            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0
                    torch_data = torch.Tensor(batch_data[0:real_batch_size])
                    torch_data= torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)

                    seg_pred, _ = classifier(torch_data)
                    # print(batch_idx, sbatch)
                    attack = torchattacks.NB_attack(classifier, eps=0.1, alpha=0.05, iters=10)
                    temp_datga = batch_label[0:real_batch_size, ...]
                    adv_images = attack(torch_data, batch_label[0:real_batch_size, ...])

                    adv_seg_pred, _ = classifier(adv_images)

                    temp_ind = np.reshape(batch_point_index[0:real_batch_size], (-1,)).astype(int)
                    adv_whole_scene[temp_ind] = np.reshape(adv_images.transpose(1, 2)[:,:,:6].detach().cpu().numpy(), (-1, 6))
                    dis = torch.dist(adv_images, torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    adv_batch_pred_label = adv_seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
                    adv_vote_label_pool = add_vote(adv_vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               adv_batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

                    gt = torch.tensor(batch_label[0:real_batch_size, ...], dtype=torch.int).cuda()
                    pred = seg_pred.max(dim=2)[1]
                    acc = pred.eq(gt.view_as(pred)).sum().item() / (4096 * torch_data.shape[0])
                    adv_pred = adv_seg_pred.max(dim=2)[1]
                    adv_acc = adv_pred.eq(gt.view_as(adv_pred)).sum().item() / (4096 * torch_data.shape[0])

                    single_whole_scene_label = np.copy(batch_label[0:real_batch_size])
                    single_total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
                    single_total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
                    single_total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
                    single_adv_total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
                    single_adv_total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
                    for l in range(NUM_CLASSES):
                        single_total_seen_class_tmp[l] += np.sum((single_whole_scene_label == l))
                        single_total_correct_class_tmp[l] += np.sum((pred.detach().cpu().numpy() == l) & (single_whole_scene_label == l))
                        single_total_iou_deno_class_tmp[l] += np.sum(((pred.detach().cpu().numpy() == l) | (single_whole_scene_label == l)))
                        single_adv_total_correct_class_tmp[l] += np.sum((adv_pred.detach().cpu().numpy() == l) & (single_whole_scene_label == l))
                        single_adv_total_iou_deno_class_tmp[l] += np.sum(((adv_pred.detach().cpu().numpy() == l) | (single_whole_scene_label == l)))

                    single_iou_map = np.array(single_total_correct_class_tmp) / (np.array(single_total_iou_deno_class_tmp, dtype=np.      float) + 1e-6)
                    single_adv_iou_map = np.array(single_adv_total_correct_class_tmp) / (np.array(single_adv_total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
                    single_arr = np.array(single_total_seen_class_tmp)

                    single_tmp_iou = np.mean(single_iou_map[single_arr != 0])
                    single_adv_tmp_iou = np.mean(single_adv_iou_map[single_arr != 0])

                    log_string("%d\t%.3f\t%.5f\t%.5f\t%.5f\t\t%.5f\n" % (sbatch, dis, adv_acc, acc, single_adv_tmp_iou, single_tmp_iou))
                    with open(os.path.join(experiment_dir, 'log_'+timestamp+'.txt'), 'a+') as f:
                        f.write("%d\t%.3f\t%.5f\t%.5f\t%.5f\t\t%.5f\n" % (sbatch, dis, adv_acc, acc, single_adv_tmp_iou, single_tmp_iou))
            pred_label = np.argmax(vote_label_pool, 1)
            adv_pred_label = np.argmax(adv_vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                adv_total_correct_class_tmp[l] += np.sum((adv_pred_label == l) & (whole_scene_label == l))
                adv_total_iou_deno_class_tmp[l] += np.sum(((adv_pred_label == l) | (whole_scene_label == l)))

                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]
                adv_total_correct_class[l] += adv_total_correct_class_tmp[l]
                adv_total_iou_deno_class[l] += adv_total_iou_deno_class_tmp[l]


            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            adv_iou_map = np.array(adv_total_correct_class_tmp) / (np.array(adv_total_iou_deno_class_tmp, dtype=np.float) + 1e-6)

            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            adv_tmp_iou = np.mean(adv_iou_map[arr != 0])

            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], adv_tmp_iou))

            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                color_adv = g_label2color[adv_pred_label[i]]
                if args.visual:
                    fout.write('%f %f %f %f %f %f\n' % ( whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1], color[2]))
                    fout_gt.write('%f %f %f %f %f %f\n' % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0], color_gt[1], color_gt[2]))
                    fout_raw.write('%f %f %f %f %f %f\n' % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], whole_scene_data[i, 3] / 255.0, whole_scene_data[i, 4] / 255.0, whole_scene_data[i, 5] / 255.0))
                    fout_adv_raw.write('%f %f %f %f %f %f\n' % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], adv_whole_scene[i, 3], adv_whole_scene[i, 4], adv_whole_scene[i, 5]))
                    fout_adv_pred.write('%f %f %f %f %f %f\n' % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_adv[0], color_adv[1], color_adv[2]))



            if args.visual:
                fout.close()
                fout_gt.close()
                fout_raw.close()
                fout_adv_raw.close()
                fout_adv_pred.close()



        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        adv_IoU = np.array(adv_total_correct_class) / (np.array(adv_total_iou_deno_class, dtype=np.float) + 1e-6)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                    np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
        log_string("-------attack--------")
        log_string('eval point avg class IoU: %f' % np.mean(adv_IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(adv_total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                    np.sum(adv_total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))


        print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
