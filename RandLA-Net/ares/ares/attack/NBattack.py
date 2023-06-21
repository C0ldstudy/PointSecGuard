import tensorflow as tf
import numpy as np

from ares.attack.bim import BIM, TBIM
from ares.attack.utils import maybe_to_array, uniform_l_2_noise, uniform_l_inf_noise


class NBattack(BIM):
    def __init__(self, model, batch_size, loss, goal, distance_metric, session, iteration_callback=None):
        super().__init__(model, batch_size, loss, goal, distance_metric, session, iteration_callback)
        # random init magnitude
        self.rand_init_eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.rand_init_eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # calculate init point within rand_init_eps
        d = np.prod(self.model.x_shape)
        if distance_metric == 'l_inf':
            noise = uniform_l_inf_noise(batch_size, d, self.rand_init_eps_var, self.model.x_dtype)
            noise = tf.reshape(noise, self.feature_shape)

        elif distance_metric == 'l_2':
            noise = uniform_l_2_noise(batch_size, d, self.rand_init_eps_var, self.model.x_dtype)
            noise = tf.reshape(noise, self.feature_shape)
        else:
            raise NotImplementedError
        # clip by (x_min, x_max)
        xs_init = tf.clip_by_value(tf.reshape(self.xs_ph, self.feature_shape) + noise,
                                   self.model.x_min, self.model.x_max)
        self.setup_xs = self.xs_var.assign(tf.reshape(self.xs_ph, self.xs_flatten_shape))

        self.config_rand_init_eps = self.rand_init_eps_var.assign(self.rand_init_eps_ph)



    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param rand_init_magnitude: Random init max distortion, could be either a float number or a numpy float number
            array with shape of ``(self.batch_size,)``.
        :param magnitude: Max distortion, could be either a float number or a numpy float number array with shape of
            ``(self.batch_size,)``.
        :param alpha: Step size for each iteration, could be either a float number or a numpy float number array with
            shape of ``(self.batch_size,)``.
        :param iteration: Iteration count. An integer.
        '''
        super().config(**kwargs)
        if 'rand_init_magnitude' in kwargs:
            rand_init_eps = maybe_to_array(kwargs['rand_init_magnitude'], self.batch_size)
            self._session.run(self.config_rand_init_eps, feed_dict={self.rand_init_eps_ph: rand_init_eps})




class tar_NBattack(TBIM):
    def __init__(self, model, batch_size, loss, goal, distance_metric, session, iteration_callback=None):
        super().__init__(model, batch_size, loss, goal, distance_metric, session, iteration_callback)
        # random init magnitude
        self.rand_init_eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.rand_init_eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        self.config_rand_init_eps = self.rand_init_eps_var.assign(self.rand_init_eps_ph)

    def config(self, **kwargs):
        super().config(**kwargs)
        if 'rand_init_magnitude' in kwargs:
            rand_init_eps = maybe_to_array(kwargs['rand_init_magnitude'], self.batch_size)
            self._session.run(self.config_rand_init_eps, feed_dict={self.rand_init_eps_ph: rand_init_eps})
