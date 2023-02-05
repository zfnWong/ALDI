import pickle
import time
import uuid
import numpy as np
import tensorflow as tf
import sys

sys.path.append("../")
import utils
from metric import ndcg
import argparse
from pprint import pprint
import os
import pandas as pd


class BPRMF:
    def __init__(self, user_num, item_num, namespace):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.u_index = tf.placeholder(tf.int64, shape=[None], name='user_index')
        self.i_index = tf.placeholder(tf.int64, shape=[None], name='item_index')
        self.pos_i_index = tf.placeholder(tf.int64, shape=[None], name='positive_item_index')
        self.neg_i_index = tf.placeholder(tf.int64, shape=[None], name='neg_item_index')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')
        self.ones_label = tf.constant(np.ones(shape=[args.batch_size], dtype=np.float32))
        self.float_batch = tf.constant(args.batch_size, dtype=tf.float32)

        with tf.variable_scope("Embedding"):
            emb_init = tf.truncated_normal_initializer(stddev=0.01)
            self.user_emb_layer = tf.keras.layers.Embedding(input_dim=user_num,
                                                            output_dim=namespace.factor_num,
                                                            embeddings_initializer=emb_init,
                                                            )
            self.item_emb_layer = tf.keras.layers.Embedding(input_dim=item_num,
                                                            output_dim=namespace.factor_num,
                                                            embeddings_initializer=emb_init,
                                                            )

        self.user_emb = self.user_emb_layer(self.u_index)
        self.item_emb = self.item_emb_layer(self.i_index)
        pos_item_emb = self.item_emb_layer(self.pos_i_index)
        neg_item_emb = self.item_emb_layer(self.neg_i_index)

        with tf.variable_scope("loss"):
            pos_pred = tf.reduce_sum(self.user_emb * pos_item_emb, axis=-1)
            neg_pred = tf.reduce_sum(self.user_emb * neg_item_emb, axis=-1)
            self.bpr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos_pred - neg_pred,
                                                                                   labels=self.ones_label))
            self.reg_loss_in_bpr = namespace.reg_rate * 1/2 * \
                                   (tf.norm(self.user_emb) + tf.norm(pos_item_emb) + tf.norm(neg_item_emb)) \
                                   / self.float_batch
            self.bpr_loss += self.reg_loss_in_bpr

        with tf.variable_scope("rating"):
            self.rating = tf.sigmoid(tf.matmul(self.user_emb, tf.transpose(self.item_emb)))

        # rank user rating
        self.rat = tf.placeholder(tf.float32, [None, None], name='user_rat')
        self.k = tf.placeholder(tf.int32, name='atK')
        self.top_item_score, self.top_item_index = tf.nn.top_k(self.rat, k=self.k)

        # 取得 batch_norm 的 moving_mean 和 moving_var 的 update_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.bpr_optimizer = tf.train.AdamOptimizer(namespace.lr).minimize(self.bpr_loss)

    def train_bpr(self, sess, u_index, pos_i_index, neg_i_index):
        _train_dict = {
            self.is_training: True,
            self.u_index: u_index,
            self.pos_i_index: pos_i_index,
            self.neg_i_index: neg_i_index
        }
        _, loss = sess.run([self.bpr_optimizer, self.bpr_loss], feed_dict=_train_dict)
        return loss

    def get_user_rating(self, sess, u_index, i_index):
        _rating_dict = {
            self.is_training: False,
            self.u_index: u_index,
            self.i_index: i_index,
        }
        rating = sess.run(self.rating, feed_dict=_rating_dict)
        return rating

    def get_ranked_rating(self, ratings, k, sess):
        ranked_score, ranked_rating = sess.run([self.top_item_score, self.top_item_index],
                                               feed_dict={self.rat: ratings,
                                                          self.k: k})
        return ranked_score, ranked_rating

    def get_embedding(self, sess):
        return sess.run([self.user_emb_layer.embeddings, self.item_emb_layer.embeddings])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="../data/", help='Director of the dataset.')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--reg_rate", type=float, default=1e-3, help="Model regularization rate")
parser.add_argument("--factor_num", type=int, default=200, help="Embedding dimension")
parser.add_argument('--Ks', nargs='?', default='[20,50,100]', help='Output sizes of every layer.')
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--n_test_user', type=int, default=2000)
parser.add_argument('--val_start', type=int, default=1, help="Output beginning point.")
parser.add_argument("--interval", type=int, default=1, help="Output interval.")
parser.add_argument("--patience", type=int, default=10, help="Patience number")
parser.add_argument('--restore', type=str, default="", help="Name of restoring model")
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--loss', type=str, default='BPR')
args, _ = parser.parse_known_args()
args.Ks = eval(args.Ks)
pprint(vars(args))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# init
timer = utils.Timer("BPRMF")
utils.set_seed_tf(args.seed)
ndcg.init(args)

# load data
data_path = os.path.join(args.datadir, args.dataset)
para_dict = pickle.load(open(os.path.join(data_path, 'convert_dict.pkl'), 'rb'))
timer.logging("Data loaded. user:{} item:{}".format(para_dict['user_num'], para_dict['item_num']))
train_data = pd.read_csv(os.path.join(data_path, 'warm_emb.csv'), dtype=np.int64).values

# Create model
model = BPRMF(para_dict['user_num'], para_dict['item_num'], args)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # model path
    saver = tf.train.Saver()
    save_dir = './model_save'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'BPRMF_{args.dataset}_{str(uuid.uuid4())[:4]}')

    # restore model
    if len(args.restore) > 1:
        restore_model = os.path.join(save_dir, f'BPRMF_{args.dataset}_{args.restore}')
        timer.logging("Restore model from {}".format(restore_model))
        if os.path.exists(restore_model + '.meta'):
            saver.restore(sess, restore_model)
            saver.save(sess, model_path)  # save as a new model
            timer.logging("Restore finished")
        else:
            timer.logging("Can't find a existing model, now start to train.")

    # prepare the pairs that are excluded when validating or testing, since they are known to be positive pairs.
    exclude_val_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                    para_dict['warm_val_user'][:args.n_test_user],
                                                    para_dict['warm_val_user_nb'],
                                                    args.test_batch_us)
    exclude_test_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                     para_dict['warm_test_user'][:args.n_test_user],
                                                     para_dict['warm_test_user_nb'],
                                                     args.test_batch_us)
    timer.logging("Build excluded data.")

    # training setting
    samp_time = 0.0
    train_time = 0.0
    metric_time = 0.0
    best_val_value = 0.0
    patient_count = 0
    get_user_rating_func = lambda u, v: model.get_user_rating(sess, u_index=u, i_index=v)
    get_topk = lambda rat, k: model.get_ranked_rating(rat, k, sess)
    train_batch = [(begin, begin + args.batch_size)
                   for begin in range(0, len(train_data) - args.batch_size, args.batch_size)]
    timer.logging("Begin training...")

    for epoch in range(1, args.max_epoch + 1):
        # sampling
        samp_time_begin = time.time()
        pair_train_data = utils.bpr_neg_samp(
            uni_users=para_dict['warm_user'],
            n_users=len(train_data),
            support_dict=para_dict['emb_user_nb'],
            item_array=para_dict['warm_item'],
        )
        samp_time_end = time.time()
        samp_time += samp_time_end - samp_time_begin

        # optimization
        train_time_begin = time.time()
        for beg, end in train_batch:
            loss = model.train_bpr(sess,
                                   pair_train_data[:, 0][beg:end],
                                   pair_train_data[:, 1][beg:end],
                                   pair_train_data[:, 2][beg:end])
        train_time_end = time.time()
        train_time += train_time_end - train_time_begin

        # validation
        if epoch % args.interval == 0 and epoch >= args.val_start:
            metric_time_begin = time.time()
            val_res, _ = ndcg.test(get_topk, get_user_rating_func,
                                   ts_nei=para_dict['warm_val_user_nb'],
                                   ts_user=para_dict['warm_val_user'][:args.n_test_user],
                                   item_array=para_dict['item_array'],
                                   masked_items=para_dict['cold_item'],
                                   exclude_pair_cnt=exclude_val_warm,
                                   )
            metric_time_end = time.time()
            metric_time += metric_time_end - metric_time_begin

            val_value = val_res['recall'][0]
            if val_value > best_val_value:
                patient_count = 0
                best_val_value = val_value
                saver.save(sess, model_path)

            timer.logging(
                "Epo{} [{}/{}] ".format(epoch, patient_count, args.patience) +
                "samp:{:.0f}s opt:{:.0f}s val:{:.0f}s ".format(samp_time, train_time, metric_time) +
                "loss:{:.4f} metric:{:.4f}".format(loss, val_value))

            if patient_count >= args.patience:
                break
            patient_count += 1

    # Test
    saver.restore(sess, model_path)
    timer.logging('Load best model.')
    ts_res, _ = ndcg.test(get_topk, get_user_rating_func,
                          ts_nei=para_dict['warm_test_user_nb'],
                          ts_user=para_dict['warm_test_user'][:args.n_test_user],
                          item_array=para_dict['item_array'],
                          masked_items=para_dict['cold_item'],
                          exclude_pair_cnt=exclude_test_warm,
                          )
    timer.logging(
        '[Test] Time Pre Rec nDCG: ' +
        '{:.4f} {:.4f} {:.4f}'.format(ts_res['precision'][0], ts_res['recall'][0], ts_res['ndcg'][0]))

    result_file = './result/'
    os.makedirs(result_file, exist_ok=True)
    with open(result_file + 'BPRMF-%s.txt' % args.dataset, 'a') as f:
        f.write(str(vars(args)))
        for i in range(len(args.Ks)):
            f.write('| %.4f %.4f %.4f ' % (ts_res['precision'][i], ts_res['recall'][i], ts_res['ndcg'][i]))
        f.write('\n')

    # store embedding
    embedding = np.concatenate(model.get_embedding(sess), axis=0)
    emb_save_path = os.path.join(args.datadir, args.dataset, 'bprmf-{}.npy'.format(args.loss))
    np.save(emb_save_path, embedding)
    timer.logging(
        'Embeddings {} '.format(embedding.shape) +
        'of bprmf is stored in {}'.format(emb_save_path))
