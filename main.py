import os
from metric import ndcg
import utils
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint
import cold_start

# running setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--n_jobs', type=int, default=4, help='Multiprocessing number.')

# dataset
parser.add_argument('--datadir', type=str, default="./data/", help='Director of the dataset.')
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')

# warm model
parser.add_argument('--embed_meth', type=str, default='bprmf', help='Emebdding method')
parser.add_argument('--embed_loss', type=str, default='BPR', help='Type of aggregation.')

# training
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=1024, help='Normal batch size.')
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--restore', type=str, default="")
parser.add_argument('--patience', type=int, default=10, help='Early stop patience.')

# validation & testing
parser.add_argument('--Ks', type=str, default='[20]', help='Top K recommendation')
parser.add_argument('--val_start', type=int, default=0, help='Validation per training batch.')
parser.add_argument('--val_interval', type=float, default=1)
parser.add_argument('--test_batch_us', type=int, default=200)
parser.add_argument('--n_test_user', type=int, default=2000)

# cold-start method parameter
parser.add_argument('--model', type=str, default='ALDI')
parser.add_argument('--reg', type=float, default=1e-4, )
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--tws', type=int, default=0, choices=[0, 1])
parser.add_argument('--freq_coef_M', type=float, default=4)
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
args.Ks = eval(args.Ks)
utils.set_seed_tf(args.seed)
pprint(vars(args))
timer = utils.Timer(name='main')
ndcg.init(args)

""" Prepare data"""
# read content, preprocess_dict, training set, embeddings
dataset_path = os.path.join(args.datadir, args.dataset)
content_data = np.load(dataset_path + f'/{args.dataset}_item_content.npy')
para_dict = pickle.load(open(dataset_path + '/convert_dict.pkl', 'rb'))
train_data = pd.read_csv(dataset_path + '/warm_emb.csv', dtype=np.int64).values

emb_path = os.path.join(dataset_path, "{}-{}.npy".format(args.embed_meth, args.embed_loss))
USER_NUM = para_dict['user_num']
ITEM_NUM = para_dict['item_num']
emb = np.load(emb_path)
user_emb = emb[:USER_NUM]
item_emb = emb[USER_NUM:]
timer.logging('Load embeddings from {}'.format(emb_path))

# calculate item frequency
item_freq = np.ones(shape=(para_dict['item_num'],), dtype=np.float32)
item_to_user_neighbors = para_dict['emb_item_nb'][para_dict['warm_item']]
for item_index, user_neighbor_list in zip(para_dict['warm_item'], item_to_user_neighbors):
    item_to_item_neighborhoods = para_dict['emb_user_nb'][user_neighbor_list]
    item_freq[item_index] = sum([1.0 / len(neighborhood) for neighborhood in item_to_item_neighborhoods])
x_expect = (len(train_data) / para_dict['item_num']) * (1 / (len(train_data) / para_dict['user_num']))
args.freq_coef_a = args.freq_coef_M / x_expect
timer.logging('Finished computing item frequencies.')

# prepare the pairs that are excluded when validating or testing, since they are known to be positive pairs.
exclude_val_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                para_dict['cold_val_user'][:args.n_test_user],
                                                para_dict['cold_val_user_nb'],
                                                args.test_batch_us)
exclude_val_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                  para_dict['hybrid_val_user'][:args.n_test_user],
                                                  para_dict['hybrid_val_user_nb'],
                                                  args.test_batch_us)
exclude_test_warm = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                 para_dict['warm_test_user'][:args.n_test_user],
                                                 para_dict['warm_test_user_nb'],
                                                 args.test_batch_us)
exclude_test_cold = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                 para_dict['cold_test_user'][:args.n_test_user],
                                                 para_dict['cold_test_user_nb'],
                                                 args.test_batch_us)
exclude_test_hybrid = utils.get_exclude_pair_count(para_dict['pos_user_nb'],
                                                   para_dict['hybrid_test_user'][:args.n_test_user],
                                                   para_dict['hybrid_test_user_nb'],
                                                   args.test_batch_us)
timer.logging("Loaded excluded pairs for validation and test.")

""" Train """
patience_count = 0
va_metric_max = 0
train_time = 0
val_time = 0
stop_flag = 0
batch_count = 0
epoch = 0

# session config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设置tf模式为按需赠长模式
sess = tf.Session(config=config)
model = eval("cold_start.{}".format(args.model))(sess, args, emb.shape[-1], content_data.shape[-1])  # 自适应 GAN 模型

# model path
save_dir = './cold_start/model_save/'
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir + args.dataset + '-' + args.model + '-'
param_file = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
save_file = save_path + param_file
args.param_file = param_file
timer.logging('Model will be stored in ' + save_file)
saver = tf.train.Saver()
if len(args.restore) > 1:
    saver.restore(sess, save_path + args.restore)
    timer.logging("Restored model from " + save_path + args.restore)
saver.save(sess, save_file)  # save as a new model

timer.logging("Training Model...")
for epoch in range(1, args.max_epoch + 1):
    train_input = utils.bpr_neg_samp(para_dict['warm_user'], len(train_data),
                                     para_dict['emb_user_nb'], para_dict['warm_item'])
    n_batch = len(train_input) // args.batch_size
    for beg in range(0, len(train_input) - args.batch_size, args.batch_size):
        end = beg + args.batch_size
        batch_count += 1
        t_train_begin = time.time()
        batch_lbs = train_input[beg: end]
        loss = model.train(content_data[batch_lbs[:, 1]],
                           item_emb[batch_lbs[:, 1]],
                           content_data[batch_lbs[:, 2]],
                           item_emb[batch_lbs[:, 2]],
                           user_emb[batch_lbs[:, 0]],
                           item_freq[batch_lbs[:, 1]],
                           item_freq[batch_lbs[:, 2]]
                           )
        t_train_end = time.time()
        train_time += t_train_end - t_train_begin

        # validation
        if (batch_count % int(n_batch * args.val_interval) == 0) and (epoch >= args.val_start):
            t_val_begin = time.time()
            gen_user_emb = model.get_user_emb(user_emb)
            gen_item_emb = model.get_item_emb(content_data, item_emb,
                                              para_dict['warm_item'], para_dict['cold_item'])
            va_metric, _ = ndcg.test(model.get_ranked_rating,
                                     lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                                     ts_nei=para_dict['cold_val_user_nb'],
                                     ts_user=para_dict['cold_val_user'][:args.n_test_user],
                                     item_array=para_dict['item_array'],
                                     masked_items=para_dict['warm_item'],
                                     exclude_pair_cnt=exclude_val_cold,
                                     )
            va_metric_current = va_metric['ndcg'][0]
            if va_metric_current > va_metric_max:
                va_metric_max = va_metric_current
                saver.save(sess, save_file)
                patience_count = 0
            else:
                patience_count += 1
            if patience_count > args.patience:
                stop_flag = 1
                break

            t_val_end = time.time()
            val_time += t_val_end - t_val_begin
            timer.logging('Epo%d(%d/%d) Loss:%.4f|va_metric:%.4f|Best:%.4f|Time_Tr:%.2fs|Val:%.2fs' %
                          (epoch, patience_count, args.patience, loss,
                           va_metric_current, va_metric_max, train_time, val_time))
    if stop_flag:
        break
timer.logging("Finish training model at epoch {}.".format(epoch))

""" Test """
saver.restore(sess, save_file)

# Generate and user/item embeddings
gen_user_emb = model.get_user_emb(user_emb)
gen_item_emb = model.get_item_emb(content_data, item_emb, para_dict['warm_item'], para_dict['cold_item'])

# cold recommendation performance
cold_res, _ = ndcg.test(model.get_ranked_rating,
                        lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                        ts_nei=para_dict['cold_test_user_nb'],
                        ts_user=para_dict['cold_test_user'][:args.n_test_user],
                        item_array=para_dict['item_array'],
                        masked_items=para_dict['warm_item'],
                        exclude_pair_cnt=exclude_test_cold,
                        )
timer.logging(
    'Cold-start recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}'.format(
        args.Ks[0], cold_res['precision'][0], cold_res['recall'][0], cold_res['ndcg'][0]))

# warm recommendation performance
warm_res, warm_dist = ndcg.test(model.get_ranked_rating,
                                lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                                ts_nei=para_dict['warm_test_user_nb'],
                                ts_user=para_dict['warm_test_user'][:args.n_test_user],
                                item_array=para_dict['item_array'],
                                masked_items=para_dict['cold_item'],
                                exclude_pair_cnt=exclude_test_warm,
                                )
timer.logging("Warm recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], warm_res['precision'][0], warm_res['recall'][0], warm_res['ndcg'][0]))

# hybrid recommendation performance
hybrid_res, _ = ndcg.test(model.get_ranked_rating,
                          lambda u, i: model.get_user_rating(u, i, gen_user_emb, gen_item_emb),
                          ts_nei=para_dict['hybrid_test_user_nb'],
                          ts_user=para_dict['hybrid_test_user'][:args.n_test_user],
                          item_array=para_dict['item_array'],
                          masked_items=None,
                          exclude_pair_cnt=exclude_test_hybrid,
                          )
timer.logging("hybrid recommendation result@{}: PRE, REC, NDCG: {:.4f}, {:.4f}, {:.4f}".format(
    args.Ks[0], hybrid_res['precision'][0], hybrid_res['recall'][0], hybrid_res['ndcg'][0]))

# save results
sess.close()
result_dir = './cold_start/result/'
os.makedirs(result_dir, exist_ok=True)
with open(result_dir + f'{args.model}.txt', 'a') as f:
    f.write(str(vars(args)))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (cold_res['precision'][i], cold_res['recall'][i], cold_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (warm_res['precision'][i], warm_res['recall'][i], warm_res['ndcg'][i]))
    f.write(' | ')
    for i in range(len(args.Ks)):
        f.write('%.4f %.4f %.4f ' % (hybrid_res['precision'][i], hybrid_res['recall'][i], hybrid_res['ndcg'][i]))
    f.write('\n')
