import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import maybe_to_array
import tensorflow.contrib.graph_editor as ge
import logging
from sklearn.metrics import confusion_matrix
import time, datetime
from numpy import linalg as LA

class tar_NUattack(BatchAttack):
    def __init__(self, model, batch_size, goal, distance_metric, confidence=0.0, learning_rate=0.01):
        self.c = 10000.0
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
        # xs_adv = tanh(ws)
        self.mask = tf.Variable(tf.zeros(shape=self.feature_shape, dtype=tf.float32))
        mask_var = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=tf.float32))

        d_ws = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=tf.float32))

        ws = tf.atanh(self._scale_to_tanh(xs_var)) + d_ws
        # self.ws = xs_var
        self.xs_adv = self._scale_to_model(tf.tanh(ws))
        self.xs_adv = tf.math.add(tf.math.multiply(mask_var, self.xs_adv), tf.math.multiply(1-mask_var, xs_var))

        self.xs_adv_model = tf.reshape(self.xs_adv, (model.config.val_batch_size, model.config.num_points, 3))

        self.other_features_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.features_ph = tf.concat((self.other_features_ph, self.xs_adv_model), axis=2)
        # the C&W loss term

        self.features_ph = tf.identity( self.features_ph )
        ge.connect(ge.sgv(self.features_ph), ge.sgv(model.inputs_features), disconnect_first=True)
        mse = tf.keras.losses.MeanSquaredError()
        self.adv_logits = model.logits

        mask_loss = tf.reshape(mask_var, self.feature_shape)
        mask_loss = tf.repeat(mask_loss, repeats=[1, 1, 11], axis=2)
        self.mask_loss = mask_loss

        NU_loss = self.NUloss(model, self.adv_logits, ys_var, mask_loss)

        if self.goal == 't' or self.goal == 'tm':
            # self.score = tf.maximum(0.0, NU_loss + confidence)
            self.score = NU_loss

        elif self.goal == 'ut':
            self.score = tf.maximum(0.0, tf.negative(NU_loss) + confidence)
        else:
            raise NotImplementedError
        # the distance term
        if self.distance_metric == 'l_2':
            self.dists = tf.norm((self.xs_adv - xs_var), axis=1)#/model.config.num_points
        else:
            raise NotImplementedError
        loss = self.dists + cs_var * self.score
        # minimize the loss using Adam
        adv_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.adv_optimizer_step = adv_optimizer.minimize(loss, var_list=[d_ws])
        self.setup_optimizer = tf.compat.v1.variables_initializer(adv_optimizer.variables())

        self.setup_xs = xs_var.assign(tf.reshape(self.xs_ph, xs_shape_flatten))
        self.setup_mask = mask_var.assign(tf.reshape(self.mask, xs_shape_flatten))
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

    def NUloss(self, model, adv_logits, ys, mask_loss):
        logits_mask = tf.one_hot(ys, model.config.num_classes)
        real_logit = tf.reduce_sum(logits_mask * adv_logits, axis=-1) # ys target label goal
        other_logit = tf.reduce_max((1- logits_mask)*adv_logits, axis=-1) # adv goal
        loss = tf.maximum(0.0, other_logit - real_logit) * mask_loss[:,:,0]
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
        return np.mean(IoU)


    def batch_attack(self, model, data_batch, session=None, target=None, ori=2):
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

        ys_target = np.where(labels==ori, target, labels)
        mask_single = np.where(labels==ori, 1.0, 0.0)
        target_points = np.sum(mask_single)
        mask = np.expand_dims(mask_single, axis=2)
        mask = np.concatenate((mask, mask, mask), axis=2)

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
        cs = self.cs.copy()
        self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})
        self._session.run(self.setup_mask, feed_dict={self.mask: mask})

        ori_score, ori_logits, ori_dists = self._session.run((self.score, model.logits,self.dists), feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.other_features_ph:features[:,:,:3],
            self.xs_adv_model: features[:,:,3:],
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        predictions = np.argmax(ori_logits, axis=1)
        mask_single = np.array(mask_single, dtype=bool)[0]
        correct = np.sum(predictions[~mask_single] == labels[0][~mask_single])
        original_other_accuracy = correct / (model.config.num_points - target_points)
        original_other_mIoU = self.compute_iou(predictions[~mask_single], labels[0][~mask_single])


        correct = np.sum(predictions[mask_single] == labels[0][mask_single])
        original_target_accuracy = correct / target_points

        correct = np.sum(predictions[mask_single] == ys_target[0][mask_single])
        sr = correct / target_points

        # find cs to begin with
        found = np.repeat(False, model.config.val_batch_size)
        for search_step in range(self.search_steps):
            self._session.run(self.setup_optimizer)
            for iii in range(self.iteration):
                self._session.run(self.adv_optimizer_step, feed_dict={
                    model.inputs_xyz: xyz,
                    model.is_training: False,
                    model.inputs_neigh_idx: neigh_idx,
                    model.inputs_sub_idx: sub_idx,
                    model.inputs_interp_idx:interp_idx,
                    self.other_features_ph:features[:,:,:3],
                    model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds
                    })

                new_score, new_logits, new_xs_adv, new_dists, inds, cloud_inds  = self._session.run((self.score, self.adv_logits, self.xs_adv_model, self.dists, model.inputs_input_inds, model.inputs_cloud_inds), feed_dict={
                    model.inputs_xyz: xyz,
                    model.is_training: False,
                    model.inputs_neigh_idx: neigh_idx,
                    model.inputs_sub_idx: sub_idx,
                    model.inputs_interp_idx:interp_idx,
                    self.other_features_ph:features[:,:,:3],
                    model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds

                    })
                new_dists = new_dists[0]
                if self.goal == 'ut' or self.goal == 't':
                    new_succ = (new_logits.max(axis=1) - new_logits.take(ys_flatten)) > self.confidence

                predictions = np.argmax(new_logits, axis=1)
                # mask_single = numpy.array(mask_single, dtype=bool)
                attack_success = np.sum(predictions[mask_single] == ys_target[0][mask_single])
                sr = attack_success / target_points
                correct = np.sum(predictions[~mask_single] == ys_target[0][~mask_single])
                other_acc = correct / (model.config.num_points - target_points)
                other_mIoU = self.compute_iou(predictions[~mask_single], labels[0][~mask_single])

                if sr > 0.95:
                    self.logger.info('cloud={}, points={}, sr={},  other_acc={}, original_other_accuracy={}, new_dists={}, other_mIoU={}, original_other_mIoU={}'.format(cloud_inds, target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU))
                    return target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU

            self.logger.info('cloud={}, points={}, sr={},  other_acc={}, original_other_accuracy={}, new_dists={}, other_mIoU={}, original_other_mIoU={}'.format(cloud_inds, target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU))

            return target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU
