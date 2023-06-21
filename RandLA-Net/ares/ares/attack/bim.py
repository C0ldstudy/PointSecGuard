import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit
import tensorflow.contrib.graph_editor as ge


class BIM(BatchAttack):
    ''' Basic Iterative Method (BIM). A white-box iterative constraint-based method. Require a differentiable loss
    function and a ``ares.model.Classifier`` model.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1607.02533.
    '''

    def __init__(self, model, batch_size, loss, goal, distance_metric, session, iteration_callback=None):
        ''' Initialize BIM.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param loss: The loss function to optimize. A ``ares.loss.Loss`` instance.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param distance_metric: Adversarial distance metric. All supported values are ``'l_2'`` and ``'l_inf'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''

        self.model, self.batch_size, self._session = model, batch_size, session
        self.loss, self.goal, self.distance_metric = loss, goal, distance_metric
        self.feature_shape = (model.config.val_batch_size, model.config.num_points, 3)
        self.model.x_dtype = tf.float32
        self.model.x_min = 0.0
        self.model.x_max = 1.0
        self.model.x_shape = self.feature_shape
        # placeholder for batch_attack's input
        self.xs_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.ys_ph = tf.placeholder(tf.int32, (model.config.val_batch_size, model.config.num_points))
        # flatten shape of xs_ph
        xs_flatten_shape = (model.config.val_batch_size, np.prod((model.config.num_points, 3)))
        self.xs_flatten_shape = xs_flatten_shape
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=tf.float32))
        # self.xs_var = xs_var

        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(model.config.val_batch_size, model.config.num_points), dtype=tf.int32))
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        # calculate loss' gradient with relate to the adversarial example


        self.xs_adv_model = tf.reshape(self.xs_adv_var, (model.config.val_batch_size, model.config.num_points, 3))

        self.other_features_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.features_ph = tf.concat((self.other_features_ph, self.xs_adv_model), axis=2)
        # the C&W loss term
        self.features_ph = tf.identity( self.features_ph )
        # print(len(self.inputs_features.op), len(model.inputs_features.op))
        ge.connect(ge.sgv(self.features_ph), ge.sgv(model.inputs_features), disconnect_first=True)
        self.score = self.colperloss(model, self.ys_var)
        self.loss = self.score
        # self.loss = loss(self.xs_adv_model, self.ys_var)
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]
        # print(grad)
        if goal == 't' or goal == 'tm':
            grad = -grad
        elif goal != 'ut':
            raise NotImplementedError
        # update the adversarial example
        if distance_metric == 'l_2':
            grad_unit = get_unit(grad)
            xs_adv_delta = self.xs_adv_var - self.xs_var + alpha * grad_unit
            # clip by max l_2 magnitude of adversarial noise
            self.xs_adv = self.xs_var + tf.clip_by_norm(xs_adv_delta, eps, axes=[1])
        elif distance_metric == 'l_inf':
            xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps
            grad_sign = tf.sign(grad)
            # clip by max l_inf magnitude of adversarial noise
            self.xs_adv = tf.clip_by_value(self.xs_adv_var + alpha * grad_sign, xs_lo, xs_hi)
        else:
            raise NotImplementedError
        # clip by (x_min, x_max)
        self.xs_adv = tf.clip_by_value(self.xs_adv, self.model.x_min, self.model.x_max)
        # self.xs_adv = self.xs_adv_var + alpha * grad_unit
        self.dists = tf.norm((self.xs_adv - self.xs_var), axis=1)
        # self.xs_adv = tf.reshape(self.xs_adv, self.feature_shape)

        # self.update_xs_adv_step = self.xs_adv_var.assign(self.xs_adv)
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)
        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape)), self.xs_adv_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape))]
        # self.setup_adv_xs = self.xs_adv_var.assign(self.xs_adv)
        self.setup_ys = self.ys_var.assign(self.ys_ph)

    def colperloss(self, model, ys):
        self.adv_logits = model.logits
        logits_mask = tf.one_hot(ys, model.config.num_classes)
        real_logit = tf.reduce_sum(logits_mask * self.adv_logits, axis=-1) # ys target label goal
        other_logit = tf.reduce_max((1- logits_mask)*self.adv_logits, axis=-1) # adv goal
        loss = tf.maximum(0.0, other_logit - real_logit)
        return tf.reduce_sum(loss, [1])


    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param magnitude: Max distortion, could be either a float number or a numpy float number array with shape of
            (batch_size,).
        :param alpha: Step size for each iteration, could be either a float number or a numpy float number array with
            shape of (batch_size,).
        :param iteration: Iteration count. An integer.
        '''
        if 'magnitude' in kwargs:
            eps = maybe_to_array(kwargs['magnitude'], self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})
        if 'alpha' in kwargs:
            alpha = maybe_to_array(kwargs['alpha'], self.batch_size)
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: alpha})
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'logger' in kwargs:
            self.logger = kwargs['logger']


    def _batch_attack_generator(self, xs, ys, ys_target=None):
        ''' Attack a batch of examples. It is a generator which yields back ``iteration_callback()``'s return value
        after each iteration if the ``iteration_callback`` is not ``None``, and returns the adversarial examples.
        '''
        labels = ys if self.goal == 'ut' else ys_target
        self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: labels})
        for _ in range(self.iteration):
            self._session.run(self.update_xs_adv_step)
            if self.iteration_callback is not None:
                yield self._session.run(self.iteration_callback)
        return self._session.run(self.xs_adv_model)

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

        # ys_input = labels.clone()
        ys_input = ys_target if self.goal == 't' or self.goal == 'tm' else labels # (val_batch_size, 40960)

        ys_flatten = np.zeros_like(labels)
        ys_flatten = ys_flatten.astype(np.int32) + labels
        # store the adversarial examples and its distance to the original examples
        xs = features[:,:,3:].copy()
        xs_adv = np.array(xs).astype(np.float32).copy()
        min_dists = np.repeat(1, model.config.val_batch_size).astype(np.float32)

        ori_logits = self._session.run(model.logits, feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.features_ph:features,
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        correct = np.sum(np.argmax(ori_logits, axis=1) == labels)
        original_accuracy = correct / model.config.num_points
        prediction = np.argmax(ori_logits, axis=1)
        original_mIoU = self.compute_iou(prediction, labels)
        self._session.run((self.setup_xs), feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys_input})
        # self._session.run(self.setup_adv_xs)

        ori_score, ori_logits, new_xs_adv, ori_dists, eps, alpha = self._session.run((self.score, model.logits, self.xs_adv, self.dists, self.eps_var, self.alpha_var ), feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.other_features_ph:features[:,:,:3],
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        for search_step in range(self.iteration):
            old_new_xs_adv = new_xs_adv.copy()
            new_score, new_logits, new_xs_adv, new_dists, inds, cloud_inds  = self._session.run((self.score, self.adv_logits, self.xs_adv, self.dists, model.inputs_input_inds, model.inputs_cloud_inds), feed_dict={
                model.inputs_xyz: xyz,
                model.is_training: False,
                model.inputs_neigh_idx: neigh_idx,
                model.inputs_sub_idx: sub_idx,
                model.inputs_interp_idx:interp_idx,
                self.other_features_ph:features[:,:,:3],
                self.xs_adv_var: new_xs_adv,
                model.inputs_labels:labels,
                model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})
            from numpy import linalg as LA
            # temp_dist = LA.norm(old_new_xs_adv- xs)
            # print(temp_dist)
            temp_dist = LA.norm(old_new_xs_adv- new_xs_adv)
            new_dists = new_dists[0]
            better_dists = new_dists < min_dists

            correct = np.sum(np.argmax(new_logits, axis=1) == labels)
            acc = correct / model.config.num_points
            mIoU = self.compute_iou(np.argmax(new_logits, axis=1), labels)

        rand_score, rand_logits, rand_dists = self._session.run((self.score, model.logits,self.dists),  feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            # self.features_ph:features,
            self.other_features_ph:features[:,:,:3],
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})
        rand_correct = np.sum(np.argmax(rand_logits, axis=1) == labels)
        rand_acc = rand_correct / model.config.num_points
        rand_mIoU = self.compute_iou(np.argmax(rand_logits, axis=1), labels)

        self.logger.info('cloud={}, acc={}, original_acc={}, new_dists={}, mIoU={}, original_mIoU={}, rand_acc={}, rand_mIoU={}'.format(cloud_inds, acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU))
        return acc, original_accuracy, new_dists, mIoU, original_mIoU, rand_acc, rand_mIoU



    def batch_attack_back(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.

        :return: When the ``iteration_callback`` is ``None``, return the generated adversarial examples. When the
            ``iteration_callback`` is not ``None``, return a generator, which yields back the callback's return value
            after each iteration and returns the generated adversarial examples.
        '''
        g = self._batch_attack_generator(xs, ys, ys_target)
        if self.iteration_callback is None:
            try:
                next(g)
            except StopIteration as exp:
                return exp.value
        else:
            return g





class TBIM(BatchAttack):
    ''' Basic Iterative Method (BIM). A white-box iterative constraint-based method. Require a differentiable loss
    function and a ``ares.model.Classifier`` model.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1607.02533.
    '''

    def __init__(self, model, batch_size, loss, goal, distance_metric, session, iteration_callback=None):
        ''' Initialize BIM.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param loss: The loss function to optimize. A ``ares.loss.Loss`` instance.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param distance_metric: Adversarial distance metric. All supported values are ``'l_2'`` and ``'l_inf'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''

        self.model, self.batch_size, self._session = model, batch_size, session
        self.loss, self.goal, self.distance_metric = loss, goal, distance_metric
        self.feature_shape = (model.config.val_batch_size, model.config.num_points, 3)
        self.model.x_dtype = tf.float32
        self.model.x_min = 0.0
        self.model.x_max = 1.0
        self.model.x_shape = self.feature_shape
        # placeholder for batch_attack's input
        self.xs_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.ys_ph = tf.placeholder(tf.int32, (model.config.val_batch_size, model.config.num_points))
        # flatten shape of xs_ph
        xs_flatten_shape = (model.config.val_batch_size, np.prod((model.config.num_points, 3)))
        self.xs_flatten_shape = xs_flatten_shape
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=tf.float32))
        # self.xs_var = xs_var

        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(model.config.val_batch_size, model.config.num_points), dtype=tf.int32))
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        self.mask = tf.Variable(tf.zeros(shape=self.feature_shape, dtype=tf.float32))

        mask_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=tf.float32))
        self.xs_adv_var = tf.math.add(tf.math.multiply(mask_var, self.xs_adv_var), tf.math.multiply(1-mask_var, self.xs_adv_var))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        # calculate loss' gradient with relate to the adversarial example
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (model.config.val_batch_size, model.config.num_points, 3))

        self.other_features_ph = tf.placeholder(tf.float32, self.feature_shape)
        self.features_ph = tf.concat((self.other_features_ph, self.xs_adv_model), axis=2)
        # the C&W loss term
        self.features_ph = tf.identity( self.features_ph )
        # print(len(self.inputs_features.op), len(model.inputs_features.op))
        ge.connect(ge.sgv(self.features_ph), ge.sgv(model.inputs_features), disconnect_first=True)
        mask_loss = tf.reshape(mask_var, self.feature_shape)
        mask_loss = tf.repeat(mask_loss, repeats=[1, 1, 11], axis=2)
        self.mask_loss = mask_loss
        self.adv_logits = model.logits
        self.score = self.colperloss(model, self.adv_logits, self.ys_var, mask_loss)


        self.loss = self.score
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]
        # print(grad)
        if goal == 't' or goal == 'tm':
            grad = -grad
        elif goal != 'ut':
            raise NotImplementedError
        # update the adversarial example
        if distance_metric == 'l_2':
            grad_unit = get_unit(grad)
            # grad_sign = tf.sign(grad)
            xs_adv_delta = self.xs_adv_var - self.xs_var + alpha * grad_unit
            # clip by max l_2 magnitude of adversarial noise
            self.xs_adv = self.xs_var + tf.clip_by_norm(xs_adv_delta, eps, axes=[1])
        elif distance_metric == 'l_inf':
            xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps
            grad_sign = tf.sign(grad)
            # clip by max l_inf magnitude of adversarial noise
            self.xs_adv = tf.clip_by_value(self.xs_adv_var + alpha * grad_sign, xs_lo, xs_hi)
        else:
            raise NotImplementedError
        # clip by (x_min, x_max)
        self.xs_adv = tf.clip_by_value(self.xs_adv, self.model.x_min, self.model.x_max)
        self.dists = tf.norm((self.xs_adv - self.xs_var))
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)
        self.setup_mask = mask_var.assign(tf.reshape(self.mask, xs_flatten_shape))
        self.setup_xs = self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape))
        self.setup_ys = self.ys_var.assign(self.ys_ph)

    def colperloss(self, model, adv_logits, ys, mask_loss):
        logits_mask = tf.one_hot(ys, model.config.num_classes)
        real_logit = tf.reduce_sum(logits_mask * adv_logits, axis=-1) # ys target label goal
        other_logit = tf.reduce_max((1- logits_mask)*adv_logits, axis=-1) # adv goal
        loss = tf.maximum(0.0, other_logit - real_logit) * mask_loss[:,:,0]
        return tf.reduce_sum(loss, [1])

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param magnitude: Max distortion, could be either a float number or a numpy float number array with shape of
            (batch_size,).
        :param alpha: Step size for each iteration, could be either a float number or a numpy float number array with
            shape of (batch_size,).
        :param iteration: Iteration count. An integer.
        '''
        if 'magnitude' in kwargs:
            eps = maybe_to_array(kwargs['magnitude'], self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})
        if 'alpha' in kwargs:
            alpha = maybe_to_array(kwargs['alpha'], self.batch_size)
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: alpha})
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'logger' in kwargs:
            self.logger = kwargs['logger']


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


        ys_flatten = np.zeros_like(labels)
        ys_flatten = ys_flatten.astype(np.int32) + labels
        # store the adversarial examples and its distance to the original examples
        xs = features[:,:,3:].copy()
        ori_adv = np.array(xs).astype(np.float32).copy().reshape(-1, 122880)

        self._session.run((self.setup_xs), feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys_input})
        self._session.run(self.setup_mask, feed_dict={self.mask: mask})

        ori_logits, ori_dists, new_xs_adv = self._session.run((model.logits, self.dists, self.xs_var), feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            self.features_ph:features,
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        from numpy import linalg as LA
        temp_dist = LA.norm(new_xs_adv- ori_adv)
        predictions = np.argmax(ori_logits, axis=1)
        mask_single = np.array(mask_single, dtype=bool)[0]
        correct = np.sum(predictions[~mask_single] == labels[0][~mask_single])
        original_other_accuracy = correct / (model.config.num_points - target_points)
        original_other_mIoU = self.compute_iou(predictions[~mask_single], labels[0][~mask_single])
        correct = np.sum(predictions[mask_single] == labels[0][mask_single])
        original_target_accuracy = correct / target_points

        correct = np.sum(predictions[mask_single] == ys_target[0][mask_single])
        sr = correct / target_points
        ori_score, ori_logits, new_xs_adv, ori_dists, eps, alpha = self._session.run((self.score, model.logits, self.xs_adv, self.dists, self.eps_var, self.alpha_var), feed_dict={
            model.inputs_xyz: xyz,
            model.is_training: False,
            model.inputs_neigh_idx: neigh_idx,
            model.inputs_sub_idx: sub_idx,
            model.inputs_interp_idx:interp_idx,
            # self.features_ph:features,

            self.other_features_ph:features[:,:,:3],
            self.xs_adv_var: new_xs_adv,
            model.inputs_labels:labels,model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

        for search_step in range(self.iteration):
            # self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys_input})
            old_new_xs_adv = new_xs_adv.copy()
            new_score, new_logits, new_xs_adv, new_dists, inds, cloud_inds  = self._session.run((self.score, self.adv_logits, self.xs_adv, self.dists, model.inputs_input_inds, model.inputs_cloud_inds), feed_dict={
                model.inputs_xyz: xyz,
                model.is_training: False,
                model.inputs_neigh_idx: neigh_idx,
                model.inputs_sub_idx: sub_idx,
                model.inputs_interp_idx:interp_idx,
                self.other_features_ph:features[:,:,:3],
                self.xs_adv_var: new_xs_adv,
                model.inputs_labels:labels,
                model.inputs_input_inds:input_inds,model.inputs_cloud_inds:cloud_inds})

            predictions = np.argmax(new_logits, axis=1)
            # mask_single = numpy.array(mask_single, dtype=bool)
            attack_success = np.sum(predictions[mask_single] == ys_target[0][mask_single])
            sr = attack_success / target_points
            correct = np.sum(predictions[~mask_single] == ys_target[0][~mask_single])
            other_acc = correct / (model.config.num_points - target_points)
            other_mIoU = self.compute_iou(predictions[~mask_single], labels[0][~mask_single])
            if sr > 0.90:
                break
        self.logger.info('cloud={}, points={}, sr={},  other_acc={}, original_other_accuracy={}, new_dists={}, other_mIoU={}, original_other_mIoU={}'.format(cloud_inds, target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU))

        return target_points,  sr, other_acc, original_other_accuracy, new_dists, other_mIoU, original_other_mIoU



    def batch_attack_back(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.

        :return: When the ``iteration_callback`` is ``None``, return the generated adversarial examples. When the
            ``iteration_callback`` is not ``None``, return a generator, which yields back the callback's return value
            after each iteration and returns the generated adversarial examples.
        '''
        g = self._batch_attack_generator(xs, ys, ys_target)
        if self.iteration_callback is None:
            try:
                next(g)
            except StopIteration as exp:
                return exp.value
        else:
            return g
