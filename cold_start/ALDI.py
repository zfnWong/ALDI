import numpy as np
import tensorflow as tf


def dense_batch_fc_tanh(x, units, is_training, scope, do_norm=False):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init,
                               regularizer=tf.contrib.layers.l2_regularizer(1e-3), )
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(1e-3), )
        h1 = tf.matmul(x, h1_w) + h1_b
        if do_norm:
            h2 = tf.layers.batch_normalization(h1, training=is_training, name=scope + '_bn')
            return tf.nn.tanh(h2)
        else:
            return tf.nn.tanh(h1)


def dense_fc(x, units, scope):
    with tf.variable_scope(scope):
        init = tf.truncated_normal_initializer(stddev=0.01)
        h1_w = tf.get_variable(scope + '_w',
                               shape=[x.get_shape().as_list()[1], units],
                               initializer=init,
                               regularizer=tf.contrib.layers.l2_regularizer(1e-3), )
        h1_b = tf.get_variable(scope + '_b',
                               shape=[1, units],
                               initializer=tf.zeros_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(1e-3), )
        h1 = tf.matmul(x, h1_w) + h1_b
        return h1


class ALDI(object):
    def __init__(self, sess, args, emb_dim, content_dim):
        self.sess = sess
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.transformed_layers = [200, 200]
        self.lr = args.lr
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.freq_coef_a = args.freq_coef_a
        self.freq_coef_M = args.freq_coef_M
        self.warm_item = None
        self.cold_item = None

        self.content = tf.placeholder(tf.float32, [None, content_dim], name='condition')
        self.real_emb = tf.placeholder(tf.float32, [None, emb_dim], name='real_emb')
        self.opp_emb = tf.placeholder(tf.float32, [None, emb_dim], name='opposite_emb')
        self.target = tf.placeholder(tf.float32, [None], name='label')
        self.g_training = tf.placeholder(tf.bool, name='generator_training_flag')
        self.diag_mask = tf.sparse_add(
            tf.ones(shape=[2 * args.batch_size, 2 * args.batch_size], dtype=tf.float32),
            tf.SparseTensor(indices=[[i, i] for i in range(2 * args.batch_size)],
                            values=[-1.0 for _ in range(2 * args.batch_size)],
                            dense_shape=[2 * args.batch_size, 2 * args.batch_size])
        )
        self.one_labels = tf.constant(np.ones(shape=[1, args.batch_size], dtype=np.float32))
        self.item_freq = tf.placeholder(tf.float32, [None], name='item_frequency')
        self.item_weight = tf.clip_by_value(tf.nn.tanh(self.freq_coef_a * self.item_freq), 0,
                                            np.tanh(self.freq_coef_M))

        # build student network - cold model
        with tf.variable_scope('G'):
            # Transform embeddings
            mask_emb = self.content
            for ihid, hid in enumerate(self.transformed_layers[:-1]):
                mask_emb = dense_batch_fc_tanh(mask_emb, hid, self.g_training, 'item_layer_%d' % ihid, True)
                user_emb = dense_batch_fc_tanh(self.opp_emb, hid, self.g_training, 'user_layer_%d' % ihid, True)
            mask_emb = dense_fc(mask_emb, self.transformed_layers[-1], 'item_output')
            user_emb = dense_fc(user_emb, self.transformed_layers[-1], 'user_output')
            self.gen_emb = mask_emb
            self.user_emb = user_emb

        """Construct Loss"""
        # supervised loss
        flatten_output_logit = tf.reduce_sum(tf.multiply(tf.tile(self.user_emb, [2, 1]), self.gen_emb), axis=-1)
        output_logit = tf.reshape(flatten_output_logit, [2, -1])
        pos_logit = tf.gather(output_logit, indices=[0], axis=0)
        neg_logit = tf.gather(output_logit, indices=[1], axis=0)
        output_logit = pos_logit - neg_logit  # (batch, )
        self.supervised_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logit, labels=self.one_labels)
        )

        # label distillation
        flatten_teacher_output_logit = tf.reduce_sum(tf.multiply(tf.tile(self.opp_emb, [2, 1]), self.real_emb),
                                                     axis=1)
        teacher_output_logit = tf.reshape(flatten_teacher_output_logit, [2, -1])  # (2, batch)
        teacher_pos_output = tf.gather(teacher_output_logit, indices=[0], axis=0)  # (batch, )
        teacher_neg_output = tf.gather(teacher_output_logit, indices=[1], axis=0)  # (batch, )
        teacher_output_logit = teacher_pos_output - teacher_neg_output

        # choose the positive item frequence
        separate_item_freq = tf.reshape(self.item_weight, [2, -1])
        pool_item_freq = tf.gather(separate_item_freq, indices=[0], axis=0)

        self.distill_loss = self.alpha * tf.reduce_mean(
            tf.multiply(pool_item_freq,
            tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logit,
                                                    labels=tf.nn.sigmoid(teacher_output_logit),
                                                    )
            )
        )

        # contrastive item-item and user-user relation distillation
        student_pos_ii_logit = tf.reduce_sum(tf.multiply(self.gen_emb, self.gen_emb), axis=1)  # (batch, )
        student_neg_ii_logit = tf.reduce_mean(tf.matmul(self.gen_emb, self.gen_emb, transpose_b=True),
                                              axis=1)  # (batch, )
        student_ii_logit = tf.subtract(student_pos_ii_logit, student_neg_ii_logit)

        teacher_pos_ii_logit = tf.reduce_sum(tf.multiply(self.real_emb, self.real_emb), axis=1)
        teacher_neg_ii_logit = tf.reduce_mean(tf.matmul(self.real_emb, self.real_emb, transpose_b=True), axis=1)
        teacher_ii_logit = tf.subtract(teacher_pos_ii_logit, teacher_neg_ii_logit)

        self.distill_loss += self.beta * tf.reduce_mean(
            tf.multiply(self.item_weight,
            tf.nn.sigmoid_cross_entropy_with_logits(logits=student_ii_logit,
                                                    labels=tf.nn.sigmoid(teacher_ii_logit)
                                                    )
            )
        )

        student_pos_uu_logit = tf.reduce_sum(tf.multiply(self.user_emb, self.user_emb), axis=1)
        student_neg_uu_logit = tf.reduce_mean(tf.matmul(self.user_emb, self.user_emb, transpose_b=True), axis=1)
        student_uu_logit = tf.subtract(student_pos_uu_logit, student_neg_uu_logit)

        teacher_pos_uu_logit = tf.reduce_sum(tf.multiply(self.opp_emb, self.opp_emb), axis=1)
        teacher_neg_uu_logit = tf.reduce_mean(tf.matmul(self.opp_emb, self.opp_emb, transpose_b=True), axis=1)
        teacher_uu_logit = tf.subtract(teacher_pos_uu_logit, teacher_neg_uu_logit)

        self.distill_loss += self.beta * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=student_uu_logit,
                                                    labels=tf.nn.sigmoid(teacher_uu_logit))
        )

        # output distribution distillation.
        self.distill_loss += self.gamma * tf.reduce_mean(
            tf.multiply(self.item_weight,
            tf.abs(flatten_teacher_output_logit - flatten_output_logit)
            )
        )

        # regularization
        self.g_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='G')
        self.g_loss = self.supervised_loss + self.distill_loss + tf.add_n(self.g_reg_loss)

        self.G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_train_step = tf.train.AdamOptimizer(self.lr).minimize(
                self.g_loss, var_list=self.G_var)

        # get user rating through dot
        self.uemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='user_embedding')
        self.iemb = tf.placeholder(tf.float32, [None, self.emb_dim], name='item_embedding')
        self.user_rating = tf.matmul(self.uemb, tf.transpose(self.iemb))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        self.sess.run(tf.global_variables_initializer())

    def train(self, pos_content, pos_emb, neg_content, neg_emb, opp_emb, pos_freq, neg_freq):
        content = np.concatenate([pos_content, neg_content], axis=0)
        emb = np.concatenate([pos_emb, neg_emb], axis=0)
        freq = np.concatenate([pos_freq, neg_freq], axis=0)
        _, g_loss, item_weight = self.sess.run(
            [self.g_train_step, self.g_loss, self.item_weight],
            feed_dict={self.content: content,
                       self.real_emb: emb,
                       self.opp_emb: opp_emb,
                       self.g_training: True,
                       self.item_freq: freq,
                       })
        return g_loss

    def get_user_rating(self, uids, iids, uemb, iemb):
        out_rating = np.zeros(shape=(len(uids), len(iids)), dtype=np.float32)
        warm_rating = self.sess.run(self.user_rating,
                                    feed_dict={self.uemb: uemb[0][uids],
                                               self.iemb: iemb[self.warm_item], })
        out_rating[:, self.warm_item] = warm_rating
        cold_rating = self.sess.run(self.user_rating,
                                    feed_dict={self.uemb: uemb[1][uids],
                                               self.iemb: iemb[self.cold_item], })
        out_rating[:, self.cold_item] = cold_rating
        return out_rating

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        self.warm_item = warm_item
        self.cold_item = cold_item
        real_emb = np.zeros((len(cold_item), self.emb_dim), dtype=np.float32)
        out_emb = np.copy(item_emb)
        out_emb[cold_item] = self.sess.run(self.gen_emb, feed_dict={self.content: content[cold_item],
                                                                    self.real_emb: real_emb,
                                                                    self.g_training: False})
        return out_emb

    def get_user_emb(self, user_emb):
        trans_user_emb = self.sess.run(self.user_emb, feed_dict={self.opp_emb: user_emb,
                                                                 self.g_training: False})
        return np.stack([user_emb, trans_user_emb], axis=0)  # (2, n_user, emb)

    def get_ranked_rating(self, ratings, k):
        ranked_score, ranked_rat = self.sess.run([self.top_score, self.top_item_index],
                                                 feed_dict={self.rat: ratings,
                                                            self.k: k})
        return ranked_score, ranked_rat
