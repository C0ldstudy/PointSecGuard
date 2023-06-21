import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import maybe_to_array
import tensorflow.contrib.graph_editor as ge
import logging
from sklearn.metrics import confusion_matrix
import time, datetime
from numpy import linalg as LA

class NUattack(BatchAttack):
    def __init__(self, model, batch_size, goal, distance_metric, cw_loss_c=99999.0, confidence=0.0, learning_rate=0.01):
        self.c = 1.0
        self.goal, self.distance_metric = goal, distance_metric
        self.confidence = confidence
        self.feature_shape = (model.config.val_batch_size, model.config.num_points, 3)
        # flatten shape of xs_ph
        xs_shape_flatten = (model.config.val_batch_size, np.prod((model.config.num_points, 3)))
        num_layers = model.config.num_layers

        self.xs_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.ys_ph = tf.placeholder(tf.int32, (model.config.val_batch_size, model.config.num_points))

        # store adversarial examples and labels in variables to reduce memory copy between tensorflow and python
        xs_var = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=tf.float32))
        ys_var = tf.Variable(tf.zeros(shape=(model.config.val_batch_size, model.config.num_points), dtype=tf.int32))
        self.ys_var = ys_var
        # placeholder for c
        self.cs_ph = tf.placeholder(tf.float32, (model.config.val_batch_size,))
        cs_var = tf.Variable(tf.zeros_like(self.cs_ph))
        d_ws = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=tf.float32))
        ws = tf.atanh(self._scale_to_tanh(xs_var)) + d_ws
        self.xs_adv = self._scale_to_model(tf.tanh(ws))
        self.xs_adv_model = tf.reshape(self.xs_adv, (model.config.val_batch_size, model.config.num_points, 3))
        self.other_features_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.features_ph = tf.concat((self.other_features_ph, self.xs_adv_model), axis=2)
        # the C&W loss term
        self.features_ph = tf.identity( self.features_ph )

        # print(len(self.inputs_features.op), len(model.inputs_features.op))
        ge.connect(ge.sgv(self.features_ph), ge.sgv(model.inputs_features), disconnect_first=True)
        NU_loss = self.NUloss(model, ys_var)
        if self.goal == 't' or self.goal == 'tm':
            self.score = tf.maximum(0.0, NU_loss + confidence)
        elif self.goal == 'ut':
            # self.score = tf.maximum(0.0, tf.negative(NU_loss) + confidence)
            self.score = NU_loss
        else:
            raise NotImplementedError
        # the distance term
        if self.distance_metric == 'l_2':
            # self.dists = tf.reduce_sum(tf.square(self.xs_adv - xs_var), axis=1)#/model.config.num_points
            self.dists = tf.norm((self.xs_adv - xs_var), axis=1)#/model.config.num_points
        else:
            raise NotImplementedError
        loss = self.dists + cs_var * self.score
        # minimize the loss using Adam
        adv_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # adv_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.adv_optimizer_step = adv_optimizer.minimize(loss, var_list=[d_ws])
        self.setup_optimizer = tf.compat.v1.variables_initializer(adv_optimizer.variables())

        self.setup_xs = xs_var.assign(tf.reshape(self.xs_ph, xs_shape_flatten))
        self.setup_ys = ys_var.assign(self.ys_ph)
        self.setup_cs = cs_var.assign(self.cs_ph)
        self.setup_d_ws = d_ws.assign(tf.zeros_like(d_ws))

        # provides default values
        self.iteration = 1000
        self.search_steps = 2
        self.binsearch_steps = 10

        self.details = {}
        self.logger = None

    def config(self, **kwargs):
        if 'cs' in kwargs:
            self.cs = maybe_to_array(kwargs['cs'], target_len=kwargs['model'].config.val_batch_size).astype(np.float32)
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'search_steps' in kwargs:
            self.search_steps = kwargs['search_steps']
        if 'binsearch_steps' in kwargs:
            self.binsearch_steps = kwargs['binsearch_steps']
        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def NUloss(self, model, ys):
        self.adv_logits = model.logits
        logits_mask = tf.one_hot(ys, model.config.num_classes)
        # print(logits_mask.shape, self.adv_logits.shape)
        real_logit = tf.reduce_sum(logits_mask * self.adv_logits, axis=-1) # ys target label goal
        other_logit = tf.reduce_max((1- logits_mask)*self.adv_logits, axis=-1) # adv goal
        loss = tf.maximum(0.0, other_logit - real_logit)
        return tf.reduce_sum(loss, [1])


    def scale(self, x, dst_min, dst_max, src_min, src_max):
        k = (dst_max - dst_min) / (src_max - src_min)
        b = dst_min - k * src_min
        return k * x + b

    def _scale_to_model(self, xs):
        # return scale(xs, model.x_min, model.x_max, -1.0, 1.0)
        return self.scale(xs, 0, 1, -1.0, 1.0)

    def _scale_to_tanh(self, xs):
        # np.arctanh(np.tanh(np.arctanh(1.0 - 1e-6) + 10.0)) == 17.242754385535303
        bound = 1.0 - 1e-6
        # return self.scale(xs, -bound, bound, model.x_min, model.x_max)
        return self.scale(xs, -bound, bound, 0, 1)

    def compute_iou(self, y_pred, y_true):
        # ytrue, ypred is a flatten vector
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        current = confusion_matrix(y_true, y_pred)
        # compute mean iou
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        # print(current ,intersection, union)
        return np.mean(IoU)

    def batch_attack(self, model, data_batch, session=None):
        self._session = session

        num_layers = model.config.num_layers
        xyz = data_batch[:num_layers]
        neigh_idx = data_batch[num_layers: 2 * num_layers]
        sub_idx = data_batch[2 * num_layers:3 * num_layers]
        interp_idx = data_batch[3 * num_layers:4 * num_layers]
        features = data_batch[4 * num_layers]

        labels = data_batch[4 * num_layers + 1]
        input_inds = data_batch[4 * num_layers + 2]
        cloud_inds = data_batch[4 * num_layers + 3]
        ys_input = ys_target if self.goal == 't' or self.goal == 'tm' else labels # (val_batch_size, 40960)

        # create numpy index for fetching the original label's logit value
        ys_flatten = np.zeros_like(labels)
        ys_flatten = ys_flatten.astype(np.int32) + labels
        # store the adversarial examples and its distance to the original examples
        xs = features[:,:,3:].copy()
        xs_adv = np.array(xs).astype(np.float32).copy()
        min_dists = np.repeat(1, model.config.val_batch_size).astype(np.float32)

        self._session.run((self.setup_xs), feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys_input})
        self._session.run(self.setup_d_ws)
        # setup initial cs value
        cs = self.cs.copy()
        self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        ori_score, ori_logits, ori_dists = self._session.run((self.score, model.logits,self.dists), feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            # self.features_ph:features,
            self.other_features_ph:features[:,:,:3],
            self.xs_adv_model: features[:,:,3:],
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        correct = np.sum(np.argmax(ori_logits, axis=1) == labels)
        original_accuracy = correct / model.config.num_points
        prediction = np.argmax(ori_logits, axis=1)
        original_mIoU = self.compute_iou(prediction, labels)
        # print("===original result: score:{}, correct:{}, dists:{}, mIoU:{}".format(ori_score, correct, ori_dists, original_mIoU))
        # find cs to begin with
        found = np.repeat(False, model.config.val_batch_size)
        for search_step in range(self.search_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            for iii in range(self.iteration):
                # print(cloud_inds)
                self._session.run(self.adv_optimizer_step, feed_dict={
                    model.inputs_xyz: xyz,
                    model.is_training: False,
                    model.inputs_neigh_idx: neigh_idx,
                    model.inputs_sub_idx: sub_idx,
                    model.inputs_interp_idx:interp_idx,
                    self.other_features_ph:features[:,:,:3],
                    model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

                new_score, new_logits, new_xs_adv, new_dists, inds, cloud_inds  = self._session.run((self.score, self.adv_logits, self.xs_adv_model, self.dists, model.inputs_input_inds, model.inputs_cloud_inds), feed_dict={
                    model.inputs_xyz: xyz,
                    model.is_training: False,
                    model.inputs_neigh_idx: neigh_idx,
                    model.inputs_sub_idx: sub_idx,
                    model.inputs_interp_idx:interp_idx,
                    self.other_features_ph:features[:,:,:3],
                    model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})
                new_dists = new_dists[0]
                better_dists = new_dists < min_dists
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    new_succ = (new_logits.max(axis=1) - new_logits.take(ys_flatten)) > self.confidence
                else:
                    # for t goal, if the score is smaller than 0, the example is adversarial. The confidence is already
                    # included in the score, no need to add the confidence term here.
                    new_succ = new_score < 1e-12
                correct = np.sum(np.argmax(new_logits, axis=1) == labels)
                acc = correct / model.config.num_points
                mIoU = self.compute_iou(np.argmax(new_logits, axis=1), labels)
                detail = np.mean(new_succ, axis=1)
                new_succ = np.mean(new_succ, axis=1) > 0.8
                if acc < 1/13:
                    noise = np.random.uniform(0,1, size=features[:,:,3:].shape)
                    # print(noise)
                    noise = noise / LA.norm(noise) * float(new_dists)
                    random_features = features[:,:,3:] + noise
                    random_features = np.clip(random_features, 0, 1)
                    rand_score, rand_logits, rand_dists = self._session.run((self.score, model.logits,self.dists),  feed_dict={
                        model.inputs_xyz: xyz,
                        model.is_training: False,
                        model.inputs_neigh_idx: neigh_idx,
                        model.inputs_sub_idx: sub_idx,
                        model.inputs_interp_idx:interp_idx,
                        # self.features_ph:features,
                        self.other_features_ph:features[:,:,:3],
                        self.xs_adv_model: random_features,
                        model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})
                    rand_correct = np.sum(np.argmax(rand_logits, axis=1) == labels)
                    rand_acc = rand_correct / model.config.num_points
                    rand_mIoU = self.compute_iou(np.argmax(rand_logits, axis=1), labels)
                    self.logger.info('cloud={}, acc={}, original_acc={}, new_dists={}, mIoU={}, original_mIoU={}, rand_acc={}, rand_mIoU={}'.format(cloud_inds, acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU))
                    return acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU


            noise = np.random.uniform(0,1, size=features[:,:,3:].shape)
            # print(noise)
            noise = noise / LA.norm(noise) * float(new_dists)
            random_features = features[:,:,3:] + noise
            # print(LA.norm(random_features - features[:,:,3:]))
            random_features = np.clip(random_features, 0, 1)
            rand_score, rand_logits, rand_dists = self._session.run((self.score, model.logits,self.dists),  feed_dict={
                model.inputs_xyz: xyz,
                model.is_training: False,
                model.inputs_neigh_idx: neigh_idx,
                model.inputs_sub_idx: sub_idx,
                model.inputs_interp_idx:interp_idx,
                # self.features_ph:features,
                self.other_features_ph:features[:,:,:3],
                self.xs_adv_model: random_features,
                model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})
            rand_correct = np.sum(np.argmax(rand_logits, axis=1) == labels)
            rand_acc = rand_correct / model.config.num_points
            rand_mIoU = self.compute_iou(np.argmax(rand_logits, axis=1), labels)
            print(rand_score, rand_mIoU, rand_acc, rand_dists)





            self.logger.info('cloud={}, acc={}, original_acc={}, new_dists={}, mIoU={}, original_mIoU={}, rand_acc={}, rand_mIoU={}'.format(cloud_inds, acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU))

            return acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU

            if np.all(found):  # we have found an adversarial example for all inputs
                break
            else:  # update c value for all failed-to-attack inputs
                cs[np.logical_not(found)] *= 10.0
                self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        # prepare cs for binary search, no need to copy cs here
        cs_lo, cs_hi = np.zeros_like(cs), cs
        cs = (cs_hi + cs_lo) / 2
        self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        # binary search on cs
        for binsearch_step in range(self.binsearch_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            succ = np.repeat(False, model.config.val_batch_size)
            for iii in range(self.iteration):
                self._session.run(self.adv_optimizer_step, feed_dict={self.xyz_ph1: xyz[0], self.xyz_ph2: xyz[1], self.xyz_ph3: xyz[2], self.xyz_ph4: xyz[3], self.xyz_ph5: xyz[4], self.neigh_idx1: neigh_idx[0], self.neigh_idx2: neigh_idx[1], self.neigh_idx3: neigh_idx[2], self.neigh_idx4: neigh_idx[3], self.neigh_idx5: neigh_idx[4], self.sub_idx1: sub_idx[0], self.sub_idx2: sub_idx[1], self.sub_idx3: sub_idx[2], self.sub_idx4: sub_idx[3], self.sub_idx5: sub_idx[4], self.interp_idx1: interp_idx[0], self.interp_idx2: interp_idx[1], self.interp_idx3: interp_idx[2], self.interp_idx4: interp_idx[3], self.interp_idx5: interp_idx[4], self.other_features_ph: features[:,:,:3], self.ys_ph: labels, self.input_inds_ph: input_inds, self.cloud_inds_ph: cloud_inds, model.is_training: False})

                new_score, new_logits, new_xs_adv, new_dists = self._session.run((self.score, self.adv_logits, self.xs_adv_model, self.dists), feed_dict={self.xyz_ph1: xyz[0], self.xyz_ph2: xyz[1], self.xyz_ph3: xyz[2], self.xyz_ph4: xyz[3], self.xyz_ph5: xyz[4], self.neigh_idx1: neigh_idx[0], self.neigh_idx2: neigh_idx[1], self.neigh_idx3: neigh_idx[2], self.neigh_idx4: neigh_idx[3], self.neigh_idx5: neigh_idx[4], self.sub_idx1: sub_idx[0], self.sub_idx2: sub_idx[1], self.sub_idx3: sub_idx[2], self.sub_idx4: sub_idx[3], self.sub_idx5: sub_idx[4], self.interp_idx1: interp_idx[0], self.interp_idx2: interp_idx[1], self.interp_idx3: interp_idx[2], self.interp_idx4: interp_idx[3], self.interp_idx5: interp_idx[4], self.other_features_ph: features[:,:,:3], self.ys_ph: labels, self.input_inds_ph: input_inds, self.cloud_inds_ph: cloud_inds, model.is_training: False})

                better_dists = new_dists <= min_dists
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    new_succ = new_logits.max(axis=1) - new_logits.take(ys_flatten) > self.confidence
                else:
                    # for t goal, if the score is smaller than 0, the example is adversarial. The confidence is already
                    # included in the score, no need to add the confidence term here.
                    new_succ = new_score < 1e-12
                # if the example is adversarial and has small distance to the original example, update xs_adv and
                # min_dists

                detail = np.mean(new_succ, axis=1)
                print(iii, detail, new_dists, new_score)
                new_succ = np.mean(new_succ, axis=1) > 0.8

                to_update = np.logical_and(new_succ, better_dists)
                xs_adv[to_update], min_dists[to_update] = new_xs_adv[to_update], new_dists[to_update]
                succ[to_update] = True
                # the initial search for c might fail, while we might succeed finding an adversarial example during
                # binary search
                found[to_update] = True

            # update cs
            not_succ = np.logical_not(succ)
            cs_lo[not_succ], cs_hi[succ] = cs[not_succ], cs[succ]
            cs = (cs_hi + cs_lo) / 2.0
            self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})
            print(('binsearch_step={}, cs_mean={}, success_rate={}'.format(binsearch_step, cs.mean(), new_succ.astype(np.float).mean())))
            if self.logger:
                self.logger.info('binsearch_step={}, cs_mean={}, success_rate={}'.format(
                    binsearch_step, cs.mean(), succ.astype(np.float).mean()))

        self.details['success'] = found
        return xs_adv

