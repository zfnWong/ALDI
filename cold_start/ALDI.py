import numpy as np
import tensorflow as tf


def dense_batch_fc_tanh(x, units, is_training, scope, reg, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init,
                               regularizer=tf.contrib.layers.l2_regularizer(reg), )
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(reg), )
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.layers.batch_normalization(h1, training=is_training, name=scope + '_bn')
            return tf.nn.tanh(h2)
        else:
            return tf.nn.tanh(h1)


def dense_fc(x, units, scope, reg):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init,
                               regularizer=tf.contrib.layers.l2_regularizer(reg), )
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(reg), )
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1


class ALDI(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.transformed_layers = [200, 200]
        self.warm_item_ids = None
        self.cold_item_ids = None
        self.lr = args.lr
        self.reg = args.reg
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.freq_coef_a = args.freq_coef_a
        self.freq_coef_M = args.freq_coef_M

        self.item_content = tf.placeholder(tf.float32, [None, content_dim], name='item_content')  # [2 * batch]
        self.true_item_emb = tf.placeholder(tf.float32, [None, emb_dim], name='true_item_emb')  # [2 * batch]
        self.true_user_emb = tf.placeholder(tf.float32, [None, emb_dim], name='true_user_emb')  # [batch]
        self.training_flag = tf.placeholder(tf.bool, name='training_flag')
        self.item_freq = tf.placeholder(tf.float32, [None], name='item_frequency')
        if args.tws:
            self.item_weight = tf.clip_by_value(tf.nn.tanh(self.freq_coef_a * self.item_freq), 0,
                                                np.tanh(self.freq_coef_M))  # 这里用 tanh(M) 来作为封顶值而不是 1 不一定对，但是更符合实际
        else:
            self.item_weight = 1.0
        self.one_labels = tf.constant(np.ones(shape=[args.batch_size], dtype=np.float32))

        # build Teacher - pre-trained embedding
        # true item emb - pos, neg
        pos_true_item_emb = tf.reshape(self.true_item_emb, [2, -1, emb_dim])
        neg_true_item_emb = tf.squeeze(tf.gather(pos_true_item_emb, indices=[1], axis=0), axis=0)
        pos_true_item_emb = tf.squeeze(tf.gather(pos_true_item_emb, indices=[0], axis=0), axis=0)

        # build Student - f_U, f_I
        with tf.variable_scope('f'):
            gen_item_emb = self.item_content
            for ihid, hid in enumerate(self.transformed_layers[:-1]):
                gen_item_emb = dense_batch_fc_tanh(gen_item_emb, hid, self.training_flag, 'item_layer_%d' % ihid,
                                                   self.reg, True)
                user_emb = dense_batch_fc_tanh(self.true_user_emb, hid, self.training_flag, 'user_layer_%d' % ihid,
                                               self.reg, True)
            gen_item_emb = dense_fc(gen_item_emb, self.transformed_layers[-1], 'item_output', self.reg)
            user_emb = dense_fc(user_emb, self.transformed_layers[-1], 'user_output', self.reg)
            self.gen_item_emb = gen_item_emb
            self.map_user_emb = user_emb
        # gen item emb - pos, neg
        pos_gen_item_emb = tf.reshape(self.gen_item_emb, [2, -1, emb_dim])
        neg_gen_item_emb = tf.squeeze(tf.gather(pos_gen_item_emb, indices=[1], axis=0), axis=0)
        pos_gen_item_emb = tf.squeeze(tf.gather(pos_gen_item_emb, indices=[0], axis=0), axis=0)

        """Construct Loss"""
        # supervised loss
        student_pos_logit = tf.reduce_sum(tf.multiply(self.map_user_emb, pos_gen_item_emb), axis=1)
        student_neg_logit = tf.reduce_sum(tf.multiply(self.map_user_emb, neg_gen_item_emb), axis=1)
        student_rank_distance = student_pos_logit - student_neg_logit  # (batch, )
        self.supervised_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=student_rank_distance, labels=self.one_labels)
        )

        # ranking difference
        teacher_pos_logit = tf.reduce_sum(tf.multiply(self.true_user_emb, pos_true_item_emb), axis=1)
        teacher_neg_logit = tf.reduce_sum(tf.multiply(self.true_user_emb, neg_true_item_emb), axis=1)
        teacher_rank_distance = teacher_pos_logit - teacher_neg_logit
        if args.tws:
            separate_item_freq = tf.reshape(self.item_weight, [2, -1])
            pos_item_freq = tf.squeeze(tf.gather(separate_item_freq, indices=[0], axis=0), axis=0)
        else:
            pos_item_freq = 1.0
        self.distill_loss = self.alpha * tf.reduce_mean(
            pos_item_freq * tf.nn.sigmoid_cross_entropy_with_logits(
                # remember to activate teacher output by sigmoid
                logits=student_rank_distance, labels=tf.nn.sigmoid(teacher_rank_distance))
        )

        # identification difference
        student_ii_logit = tf.reduce_sum(tf.multiply(pos_gen_item_emb, pos_gen_item_emb), axis=1)
        student_ij_logit = tf.reduce_mean(tf.matmul(pos_gen_item_emb, neg_gen_item_emb, transpose_b=True), axis=1)
        student_iden_distance = tf.subtract(student_ii_logit, student_ij_logit)

        teacher_ii_logit = tf.reduce_sum(tf.multiply(pos_true_item_emb, pos_true_item_emb), axis=1)
        teacher_ij_logit = tf.reduce_mean(tf.matmul(pos_true_item_emb, neg_true_item_emb, transpose_b=True), axis=1)
        teacher_iden_distance = tf.subtract(teacher_ii_logit, teacher_ij_logit)
        self.distill_loss += self.beta * tf.reduce_mean(
            pos_item_freq * tf.nn.sigmoid_cross_entropy_with_logits(
                # remember to activate teacher output by sigmoid
                logits=student_iden_distance, labels=tf.nn.sigmoid(teacher_iden_distance))
        )

        # rating distribution difference.
        self.distill_loss += self.gamma * tf.reduce_mean(
            tf.abs(teacher_pos_logit - student_pos_logit) + tf.abs(teacher_neg_logit - student_neg_logit))

        # regularization
        self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='f')
        self.total_loss = self.supervised_loss + self.distill_loss + tf.add_n(self.reg_loss)

        self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='f')
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='f')
        with tf.control_dependencies(g_update_ops):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(
                self.total_loss, var_list=self.G_var)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def train(self, pos_item_content, pos_item_emb,
              neg_item_content, neg_item_emb,
              user_emb,
              pos_item_freq, neg_item_freq):
        item_content = np.concatenate([pos_item_content, neg_item_content], axis=0)
        true_item_emb = np.concatenate([pos_item_emb, neg_item_emb], axis=0)
        freq = np.concatenate([pos_item_freq, neg_item_freq], axis=0)
        _, total_loss = self.sess.run(
            [self.train_step, self.total_loss],
            feed_dict={self.item_content: item_content,
                       self.true_item_emb: true_item_emb,
                       self.true_user_emb: user_emb,
                       self.training_flag: True,
                       self.item_freq: freq,
                       })
        return total_loss

    def get_user_rating(self, uids, iids, uemb, iemb):
        out_rating = np.zeros(shape=(len(uids), len(iids)), dtype=np.float32)
        warm_rating = self.sess.run(self.user_rating,
                                    feed_dict={self.uemb: uemb[0][uids],
                                               self.iemb: iemb[self.warm_item_ids], })
        out_rating[:, self.warm_item_ids] = warm_rating
        cold_rating = self.sess.run(self.user_rating,
                                    feed_dict={self.uemb: uemb[1][uids],
                                               self.iemb: iemb[self.cold_item_ids], })
        out_rating[:, self.cold_item_ids] = cold_rating
        return out_rating

    def get_item_emb(self, item_content, item_emb, warm_item_ids, cold_item_ids):
        self.warm_item_ids = warm_item_ids
        self.cold_item_ids = cold_item_ids
        true_item_emb = np.zeros((len(cold_item_ids), self.emb_dim), dtype=np.float32)
        out_emb = np.copy(item_emb)
        out_emb[cold_item_ids] = self.sess.run(self.gen_item_emb,
                                               feed_dict={self.item_content: item_content[cold_item_ids],
                                                          self.true_item_emb: true_item_emb,
                                                          self.training_flag: False})
        return out_emb

    def get_user_emb(self, user_emb):
        trans_user_emb = self.sess.run(self.map_user_emb, feed_dict={self.true_user_emb: user_emb,
                                                                     self.training_flag: False})
        return np.stack([user_emb, trans_user_emb], axis=0)  # (2, n_user, emb)

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_index = self.sess.run([self.top_score, self.top_item_index],
                                                   feed_dict={self.rat: ratings,
                                                              self.k: k})
        return ranked_score, ranked_index
