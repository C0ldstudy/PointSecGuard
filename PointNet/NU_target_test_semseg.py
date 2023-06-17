"""
python target_test_semseg.py --log_dir tar_adv_pointnet2_sem_seg --test_area 5
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
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--data_path', type=str, default='/datasets/stanford_indoor3d/', help='path to the dataset')
    parser.add_argument('--log_dir', type=str, default='tar_adv_pointnet2_sem_seg', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]')

    parser.add_argument('--left_ratio', type=float, default=0.1, help='the left points ratio for the L0 attack')
    parser.add_argument('--origin', default=11, type=int, help='the target points of the original labels')
    parser.add_argument('--target', default=6, type=int, help='the traget label of the target points')
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
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
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


    if True:
        ori = args.origin
        target = args.target
        timestr = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(experiment_dir, 'log_'+timestr+'.txt'), 'w') as f:
            f.write("ori\tindex\tL2_dis\tcount\t target acc\tadv_acc\tacc\tadv_miou\tmiou\n")

        visual_dir = experiment_dir + '/visual_'+ str(ori) + '_' + timestr + '/'
        print(visual_dir)
        visual_dir = Path(visual_dir)
        visual_dir.mkdir(exist_ok=True)

        NUM_CLASSES = 13
        BATCH_SIZE = args.batch_size
        NUM_POINT = args.num_point

        root = args.data_path
        TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test',     test_area=args.test_area, block_points=NUM_POINT)
        log_string("The number of test data is: %d" %  len(TEST_DATASET_WHOLE_SCENE))

        '''MODEL LOADING'''
        model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
        MODEL = importlib.import_module(model_name)
        classifier = MODEL.get_model(NUM_CLASSES).cuda()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()

        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE) # 68
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        adv_total_correct_class = [0 for _ in range(NUM_CLASSES)]
        adv_total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        log_string('---- EVALUATION WHOLE SCENE----')
        # print(scene_id)
        batch_idx = scene_id.index('Area_5_office_33')

        if True:
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

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx] # (852519, 6)
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            adv_vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            adv_whole_scene = np.zeros(whole_scene_data.shape)
            # whole_scene = np.zeros(whole_scene_data.shape)

            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]  # (676, 4096, 9) (676, 4096) (676, 4096) (676, 4096)
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
                    batch_data[:,:, 3:6] /= 1.0
                    torch_data = torch.Tensor(batch_data[0:real_batch_size, ...])
                    torch_data= torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)

                    seg_pred, _ = classifier(torch_data)

                    mask = np.in1d(batch_label[0:real_batch_size, ...], ori).reshape(batch_label[0:real_batch_size, ...].shape)
                    count = (mask == True).sum()

                    if count == 0:
                        adv_images = torch_data.clone()
                    else:
                        attack = torchattacks.tar_NU_attack(classifier, c=1, kappa=0, steps=1000, lr=0.01, target=target, mask=mask[0])
                        adv_images = attack(torch_data, batch_label[0:real_batch_size, ...])  # (batch, 6, 4096)

                    dis = torch.dist(adv_images, torch_data)
                    target_labels = torch.full(batch_label[0:real_batch_size, ...].shape, target).cuda()
                    adv_seg_pred, _ = classifier(adv_images)

                    temp_ind = np.reshape(batch_point_index[0:real_batch_size], (-1,)).astype(int)
                    adv_whole_scene[temp_ind] = np.reshape(adv_images.transpose(1, 2)[:,:,:6].detach().cpu().numpy(), (-1, 6))


                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    adv_batch_pred_label = adv_seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    tmp_adv_batch_pred_label = adv_seg_pred.contiguous().data.max(2)[1]
                    # print(tmp_adv_batch_pred_label, count)
                    if count != 0:
                        target_acc = tmp_adv_batch_pred_label.unsqueeze(0)[:, mask].eq(target_labels.unsqueeze(0)[:, mask].view_as(tmp_adv_batch_pred_label.unsqueeze(0)[:, mask])).sum().item() / count
                        print("target acc: ", target_acc)

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
                    adv_vote_label_pool = add_vote(adv_vote_label_pool, batch_point_index[0:real_batch_size, ...], adv_batch_pred_label[0:real_batch_size, ...], batch_smpw[0:real_batch_size, ...])


                    gt = torch.tensor(batch_label[0:real_batch_size, ...], dtype=torch.int).cuda()

                    # print(gt.shape, seg_pred.shape)
                    pred = seg_pred.max(dim=2)[1]
                    # print(pred.shape, torch_data.shape)
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
                    # print(sbatch, dis, adv_acc, acc, single_adv_iou_map, single_iou_map)

                    single_arr = np.array(single_total_seen_class_tmp)

                    single_tmp_iou = np.mean(single_iou_map[single_arr != 0])
                    single_adv_tmp_iou = np.mean(single_adv_iou_map[single_arr != 0])

                    if count != 0:
                        log_string("%d\t%d\t%.3f\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\n" % (ori,sbatch, dis, count ,target_acc, adv_acc, acc, single_adv_tmp_iou, single_tmp_iou))

                        with open(os.path.join(experiment_dir, 'log_'+timestr+'.txt'), 'a+') as f:
                            f.write("%d\t%d\t%.3f\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t\t%.5f\n" % (ori, sbatch, dis, count ,target_acc,adv_acc, acc, single_adv_tmp_iou, single_tmp_iou))


            print("adv_whole_scene: ", adv_whole_scene.shape)
            # adv_whole_scene = adv_whole_scene.reshape((-1, 6))

            pred_label = np.argmax(vote_label_pool, 1)
            adv_pred_label = np.argmax(adv_vote_label_pool, 1)
            print("=====", pred_label.shape, adv_pred_label.shape)

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


            print("iou_map: ", iou_map, "\tadv_iou_map: ", adv_iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            adv_tmp_iou = np.mean(adv_iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            log_string('Adv Mean IoU of %s: %.4f' % (scene_id[batch_idx], adv_tmp_iou))

            print('----------------------------')
            print("pos distance: ", np.linalg.norm(whole_scene_data[:,:3] - adv_whole_scene[:,:3]))
            print("color distance: ", np.linalg.norm(whole_scene_data[:,3:6]-adv_whole_scene[:,3:6]))

            log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            log_string('eval whole scene point accuracy: %f' % (
                    np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
            log_string('adv eval whole scene point avg class acc: %f' % (
            np.mean(np.array(adv_total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            log_string('adv eval whole scene point accuracy: %f' % (
                    np.sum(adv_total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))


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
            # exit()


        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
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

        print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)
