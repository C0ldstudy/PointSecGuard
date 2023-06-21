from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time
from ares import NUattack, tar_NUattack, CrossEntropyLoss, NBattack, tar_NBattack
import os
import logging
import time, datetime

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

class ModelTester:
    def __init__(self, model, dataset, goal='ut', attack_type='NB', restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open('log_test_' + str(dataset.val_split) + '.txt', 'a')
        self.attack_type = attack_type

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        loss = CrossEntropyLoss(model)
        if goal == 'ut':
            if self.attack_type == 'NB':
                self.nb_ut = NBattack(model=model, batch_size=model.config.val_batch_size, goal='ut', distance_metric='l_2', loss=loss, session=self.sess)
            elif self.attack_type == 'NU':
                self.nu_ut = NUattack(model=model, batch_size=model.config.val_batch_size, goal='ut', distance_metric='l_2')
        elif goal == 't':
            if self.attack_type == 'NB':
                self.nb_t = tar_NBattack(model=model, batch_size=model.config.val_batch_size, goal='t', distance_metric='l_2', loss=loss, session=self.sess)
            elif self.attack_type == 'NU':
                self.nu_t = tar_NUattack(model=model, batch_size=model.config.val_batch_size, goal='t', distance_metric='l_2',)

        self.sess.run(tf.global_variables_initializer())
        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(restore_snap))
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)
        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]



    def NUAttack(self, model, dataset, num_votes=1, attack_bool="True"):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)
        if attack_bool=='True':
            xs_ph = tf.placeholder(tf.float32, shape=(None, model.config.num_points, 3))
            # lgs, lbs = model.logits(xs_ph)
            attack_logger = logging.getLogger('non_target_attack')
            logging.basicConfig(filename='./logs/non_target_NU/non_target_attack_s3dis_'+str(datetime.datetime.now().time())+'.log')
            attack_logger.setLevel(logging.INFO)
            self.nu_ut.config(model=model, cs=0.5, logger=attack_logger)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        if attack_bool=='True':
            file_writer = tf.summary.FileWriter('./logs/', self.sess.graph)
            data_batch = self.sess.run(dataset.flat_inputs)
            num_layers = model.config.num_layers
            xyz = data_batch[:num_layers]
            neigh_idx = data_batch[num_layers: 2 * num_layers]
            sub_idx = data_batch[2 * num_layers:3 * num_layers]
            interp_idx = data_batch[3 * num_layers:4 * num_layers]
            features = data_batch[4 * num_layers]
            labels = data_batch[4 * num_layers + 1]
            input_inds = data_batch[4 * num_layers + 2]
            cloud_inds = data_batch[4 * num_layers + 3]

            ori_logits = self.sess.run(model.logits, feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.nu_ut.features_ph:features,
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

            correct = np.sum(np.argmax(ori_logits, axis=1) == labels)
            prediction = np.argmax(ori_logits, axis=1)
            accs, original_accuracys, new_dists, mIoUs, original_mIoUs, rand_accs, rand_mIoUs = [], [], [], [], [], [], []
            for i in range(1000):
                acc, original_accuracy, new_dist, mIoU, original_mIoU, rand_acc, rand_mIoU = self.nu_ut.batch_attack(model, data_batch, session=self.sess)
                accs.append(acc)
                original_accuracys.append(original_accuracy)
                new_dists.append(new_dist)
                mIoUs.append(mIoU)
                original_mIoUs.append(original_mIoU)
                rand_accs.append(rand_acc)
                rand_mIoUs.append(rand_mIoU)
                try:
                    data_batch = self.sess.run(dataset.flat_inputs)
                    labels = data_batch[4 * num_layers + 1]
                except tf.errors.OutOfRangeError:
                    break
            attack_logger.info('mean_acc={}, mean_original_acc={}, mean_new_dists={}, mean_mIoU={}, mean_original_mIoU={}, mean_rand_acc={}, mean_rand_mIoU={}'.format(sum(accs)/len(accs), sum(original_accuracys)/len(original_accuracys),sum(new_dists)/len(new_dists),sum(mIoUs)/len(mIoUs),sum(original_mIoUs)/len(original_mIoUs), sum(rand_accs)/len(rand_accs), sum(rand_mIoUs)/len(rand_mIoUs)))
        return


    def NBAttack(self, model, dataset, num_votes=1, attack_bool="True"):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)
        if attack_bool=='True':
            xs_ph = tf.placeholder(tf.float32, shape=(None, model.config.num_points, 3))
            attack_logger = logging.getLogger('non_target_attack_NB')
            logging.basicConfig(filename='./logs/non_target_NB/non_target_attack_s3dis_'+str(datetime.datetime.now().time())+'.log')
            attack_logger.setLevel(logging.INFO)
            self.nb_ut.config(model=model, logger=attack_logger, rand_init_magnitude=17.0/5,
            iteration=10,
            magnitude=17.0,
            alpha=17/10)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        if attack_bool=='True':
            file_writer = tf.summary.FileWriter('./logs/', self.sess.graph)
            data_batch = self.sess.run(dataset.flat_inputs)
            num_layers = model.config.num_layers
            xyz = data_batch[:num_layers]
            neigh_idx = data_batch[num_layers: 2 * num_layers]
            sub_idx = data_batch[2 * num_layers:3 * num_layers]
            interp_idx = data_batch[3 * num_layers:4 * num_layers]
            features = data_batch[4 * num_layers]
            labels = data_batch[4 * num_layers + 1]
            input_inds = data_batch[4 * num_layers + 2]
            cloud_inds = data_batch[4 * num_layers + 3]

            ori_logits = self.sess.run(model.logits, feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.nb_ut.features_ph:features,
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

            correct = np.sum(np.argmax(ori_logits, axis=1) == labels)
            original_accuracy = correct / model.config.num_points
            prediction = np.argmax(ori_logits, axis=1)
            accs, original_accuracys, new_dists, mIoUs, original_mIoUs, rand_accs, rand_mIoUs = [], [], [], [], [], [], []
            for i in range(100):
                acc, original_accuracy, new_dist, mIoU, original_mIoU, rand_acc, rand_mIoU = self.nb_ut.batch_attack(model, data_batch, session=self.sess)
                accs.append(acc)
                original_accuracys.append(original_accuracy)
                new_dists.append(new_dist)
                mIoUs.append(mIoU)
                original_mIoUs.append(original_mIoU)
                rand_accs.append(rand_acc)
                rand_mIoUs.append(rand_mIoU)
                try:
                    data_batch = self.sess.run(dataset.flat_inputs)
                    labels = data_batch[4 * num_layers + 1]
                except tf.errors.OutOfRangeError:
                    break
            attack_logger.info('mean_acc={}, mean_original_acc={}, mean_new_dists={}, mean_mIoU={}, mean_original_mIoU={}, mean_rand_acc={}, mean_rand_mIoU={}'.format(sum(accs)/len(accs), sum(original_accuracys)/len(original_accuracys),sum(new_dists)/len(new_dists),sum(mIoUs)/len(mIoUs),sum(original_mIoUs)/len(original_mIoUs), sum(rand_accs)/len(rand_accs), sum(rand_mIoUs)/len(rand_mIoUs)))
            print('mean_acc={}, mean_original_acc={}, mean_new_dists={}, mean_mIoU={}, mean_original_mIoU={}, mean_rand_acc={}, mean_rand_mIoU={}'.format(sum(accs)/len(accs), sum(original_accuracys)/len(original_accuracys),sum(new_dists)/len(new_dists),sum(mIoUs)/len(mIoUs),sum(original_mIoUs)/len(original_mIoUs), sum(rand_accs)/len(rand_accs), sum(rand_mIoUs)/len(rand_mIoUs)))
        return




    def tNUAttack(self, model, dataset, num_votes=1, attack_bool="True", target=2, ori=5):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)
        if attack_bool=='True':
            xs_ph = tf.placeholder(tf.float32, shape=(None, model.config.num_points, 3))
            attack_logger = logging.getLogger('target_attack')

            logging.basicConfig(filename='./logs/target_NU/target_attack_s3dis_'+str(target)+'_'+str(ori)+'_'+str(datetime.datetime.now().time())+'.log')
            attack_logger.setLevel(logging.INFO)
            self.nu_t.config(model=model, cs=1, logger=attack_logger)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        if attack_bool=='True':
            file_writer = tf.summary.FileWriter('./logs/', self.sess.graph)
            accs, original_accuracys, new_dists, mIoUs, original_mIoUs, rand_accs, rand_mIoUs = [], [], [], [], [], [], []
            num_layers = model.config.num_layers

            for i in range(1000):
                try:
                    data_batch = self.sess.run(dataset.flat_inputs)
                    labels = data_batch[4 * num_layers + 1]
                except tf.errors.OutOfRangeError:
                    # exit()
                    break
                mask = np.in1d(labels, ori).reshape(labels.shape)
                count = (mask == True).sum()
                # print(i, count)
                if count < 500:
                    attack_logger.info("No, not enough points: {}, count={}".format(str(i), count))
                    continue

                acc, original_accuracy, new_dist, mIoU, original_mIoU, rand_acc, rand_mIoU = self.nu_t.batch_attack(model, data_batch, session=self.sess, target=target, ori=ori)
            return




    def tNBAttack(self, model, dataset, num_votes=1, attack_bool="True", target=2, ori=5):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)
        if attack_bool=='True':
            xs_ph = tf.placeholder(tf.float32, shape=(None, model.config.num_points, 3))
            attack_logger = logging.getLogger('target_attack')

            logging.basicConfig(filename='./logs/target_NB/target_NB_attack_s3dis_'+str(target)+'_'+str(ori)+'_'+str(datetime.datetime.now().time())+'.log')
            attack_logger.setLevel(logging.INFO)
            self.nb_t.config(model=model, logger=attack_logger, rand_init_magnitude=2,
            iteration=20,
            magnitude=10,
            alpha=1)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        if attack_bool=='True':
            file_writer = tf.summary.FileWriter('./logs/', self.sess.graph)
            accs, original_accuracys, new_dists, mIoUs, original_mIoUs, rand_accs, rand_mIoUs = [], [], [], [], [], [], []
            num_layers = model.config.num_layers

            for i in range(1000):
                try:
                    data_batch = self.sess.run(dataset.flat_inputs)
                    labels = data_batch[4 * num_layers + 1]
                except tf.errors.OutOfRangeError:
                    # exit()
                    break
                mask = np.in1d(labels, ori).reshape(labels.shape)
                count = (mask == True).sum()
                # print(i, count)
                if count < 500:
                    attack_logger.info("No, not enough points: {}, count={}".format(str(i), count))
                    continue
                target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU = self.nb_t.batch_attack(model, data_batch, session=self.sess, target=target, ori=ori)

        return



