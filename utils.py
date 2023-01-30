import time
import numpy as np
import tensorflow as tf
import random
import os


def set_seed_tf(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def df_get_neighbors(input_df, obj, max_num):
    """
    Get users' neighboring items.
    return:
        nei_array - [max_num, neighbor array], use 0 to pad users which have no neighbors.
    """
    group = tuple(input_df.groupby(obj))
    keys, values = zip(*group)  # key: obj scalar, values: neighbor array

    keys = np.array(keys, dtype=np.int64)
    opp_obj = 'item' if obj == 'user' else 'user'
    values = list(map(lambda x: x[opp_obj].values, values))
    values.append(0)
    values = np.array(values, dtype=object)

    nei_array = np.zeros((max_num,), dtype=object)
    nei_array[keys] = values[:-1]
    return nei_array


class Timer(object):
    def __init__(self, name=''):
        self._name = name
        self.begin_time = time.time()
        self.last_time = time.time()
        self.current_time = time.time()
        self.stage_time = 0.0
        self.run_time = 0.0

    def logging(self, message):
        """
        output the time information, including current datetime, time of duration, message

        :parameter:
            message - operation information
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.update()
        message = '' if message is None else message
        print("{} {} {:.0f}s {:.0f}s | {}".format(current_time,
                                                  self._name,
                                                  self.run_time,
                                                  self.stage_time,
                                                  message))

    def update(self):
        self.current_time = time.time()

        self.stage_time = self.current_time - self.last_time
        self.last_time = self.current_time
        self.run_time = self.current_time - self.begin_time
        return self


def bpr_neg_samp(uni_users, n_users, support_dict, item_array):
    """
    :parameter:
        uni_users - unique users in training data
        dict - {uid: array[items]}
        n_users - sample n users
        neg_num - n of sample pairs for a user.
        item_array - sample item in this array.

    :return:
        ret_array - [uid pos_iid neg_iid] * n_records
    """
    pos_items = []
    users = np.random.choice(uni_users, size=n_users, replace=True)
    for user in users:
        # pos sampling
        pos_candidates = support_dict[user]
        # if not hasattr(pos_candidates, 'shape'):
        #     continue
        pos_item = random.choice(pos_candidates)
        pos_items.append(pos_item)

    pos_items = np.array(pos_items, dtype=np.int32).flatten()
    neg_items = np.random.choice(item_array, len(users), replace=True)
    ret = np.stack([users, pos_items, neg_items], axis=1)
    return ret


def negative_sampling(pos_user_array, pos_item_array, neg, warm_item):
    """
    :parameter:
        pos_user_array: users in train interactions
        pos_item_array: items in train interactions
        neg: num of negative samples
        warm_item: train item set

    :return:
        user: concat pos users and neg ones
        item: concat pos item and neg ones
        target: scores of both pos interactions and neg ones
    """
    user_pos = pos_user_array.reshape((-1))
    if neg >= 1:
        user_neg = np.tile(pos_user_array, int(neg)).reshape((-1))
    else:
        user_neg = np.random.choice(pos_user_array, size=(int(neg * len(user_pos))), replace=True)
    user_array = np.concatenate([user_pos, user_neg], axis=0)
    item_pos = pos_item_array.reshape((-1))
    item_neg = np.random.choice(warm_item, size=user_neg.shape[0], replace=True).reshape((-1))
    item_array = np.concatenate([item_pos, item_neg], axis=0)
    target_pos = np.ones_like(item_pos)
    target_neg = np.zeros_like(item_neg)
    target_array = np.concatenate([target_pos, target_neg], axis=0)
    random_idx = np.random.permutation(user_array.shape[0])  # 生成一个打乱的 range 序列作为下标
    return user_array[random_idx], item_array[random_idx], target_array[random_idx]


def get_exclude_pair(pos_user_nb, u_pair, ts_nei):
    """Find the items in the complete dataset but not in the test set for a user"""
    pos_item = np.array(list(set(pos_user_nb[u_pair[0]]) - set(ts_nei[u_pair[0]])),
                        dtype=np.int64)
    pos_user = np.array([u_pair[1]] * len(pos_item), dtype=np.int64)
    return np.stack([pos_user, pos_item], axis=1)


def get_exclude_pair_count(pos_user_nb, ts_user, ts_nei, batch):
    exclude_pair_list = []
    exclude_count = [0]
    for i, beg in enumerate(range(0, len(ts_user), batch)):
        end = min(beg + batch, len(ts_user))
        batch_user = ts_user[beg:end]
        batch_range = list(range(end - beg))
        batch_u_pair = tuple(zip(batch_user.tolist(), batch_range))  # (org_id, map_id)

        specialize_get_exclude_pair = lambda x: get_exclude_pair(pos_user_nb, x, ts_nei)
        exclude_pair = list(map(specialize_get_exclude_pair, batch_u_pair))
        exclude_pair = np.concatenate(exclude_pair, axis=0)

        exclude_pair_list.append(exclude_pair)
        exclude_count.append(exclude_count[i] + len(exclude_pair))

    exclude_pair_list = np.concatenate(exclude_pair_list, axis=0)
    return [exclude_pair_list, exclude_count]
