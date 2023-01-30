import pickle
import random
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import os

import sys
sys.path.append("..")

from utils import Timer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="./", help='Director of the dataset.')
parser.add_argument('--warm_ratio', type=float, default=0.8, help='Warm ratio of all items.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--warm_split', nargs='?', default='[0.8, 0.1, 0.1]',
                    help="For embedding training, cold-start model training, warm validation, warm testing, respectively.")
parser.add_argument('--cold_split', nargs='?', default='[0.5, 0.5]',
                    help="For cold-start validation and cold-start testing")
parser.add_argument('--cold_object', type=str, default='item', choices=['user', 'item'])
args = parser.parse_args()
args.warm_split = eval(args.warm_split)
args.cold_split = eval(args.cold_split)
pprint(vars(args))

# set seed
random.seed(args.seed)
np.random.seed(args.seed)

# store path
store_path = os.path.join(args.datadir, args.dataset)

"""read the data"""
# drop duplicates
timer = Timer("Split")
df = pd.read_csv(os.path.join(store_path, args.dataset + '.csv'),
                 header=0,
                 usecols=['user', 'item'],
                 index_col=False,
                 dtype={'user': np.int64, 'item': np.int64})
origin_len = df.shape[0]
df = df.drop_duplicates(['user', 'item']).reset_index(drop=True)
new_len = df.shape[0]
print('Duplicated :%d -> %d' % (origin_len, new_len))

# statistics
user_num = max(df['user']) + 1
item_num = max(df['item']) + 1
info_dict = {'user': user_num, 'item': item_num}
info_dict_path = os.path.join(store_path, 'n_user_item.pkl')
pickle.dump(info_dict, open(info_dict_path, 'wb'))
print('User: %d\tItem: %d' % (user_num, item_num))
print(f'Global sparse rate: %.4f' % ((user_num * item_num - new_len) / (user_num * item_num) * 100.0))
timer.logging("Data reading finished.")

"""warm/cold splitting"""
# (object_id, record_ids) for every group(user).  _[1].index is [record_ids]
# a group is a cold user/item
group = df.groupby(by=args.cold_object)
group = [g[1].index for g in group]
random.shuffle(group)
n_warm_group = int(args.warm_ratio * len(group))
n_cold_group = len(group) - n_warm_group
warm_idx = np.concatenate(group[:n_warm_group], axis=0)
cold_idx = np.concatenate(group[n_warm_group:], axis=0)
timer.logging("User/Item grouping finished.")

df_warm = df.loc[warm_idx]
df_cold = df.loc[cold_idx]
print("[Split]\tuser\titem\trecord")
print("warm\t{}\t{}\t{}".format(len(set(df_warm['user'])), len(set(df_warm['item'])), df_warm.shape[0]))
print("cold\t{}\t{}\t{}".format(len(set(df_cold['user'])), len(set(df_cold['item'])), df_cold.shape[0]))

"""warm subset splitting"""
# warm(interaction) -> emb/val/test
n_warm_val = int(args.warm_split[1] * len(warm_idx))
n_warm_test = int(args.warm_split[2] * len(warm_idx))
n_warm_emb = len(warm_idx) - n_warm_val - n_warm_test

np.random.shuffle(warm_idx)
warm_emb_idx = warm_idx[:n_warm_emb]
warm_val_idx = warm_idx[n_warm_emb:n_warm_emb + n_warm_val]
warm_test_idx = warm_idx[-n_warm_test:]
org_warm_emb_len = len(warm_emb_idx)


# Move the val records whose user/item don't emerge in emb set into emb set.
org_len = len(warm_val_idx)
warm_emb_user_set = set(df.loc[warm_emb_idx, 'user'])
df_warm_val = df.loc[warm_val_idx]
idx_to_move = df_warm_val[True ^ df_warm_val['user'].isin(warm_emb_user_set)].index
warm_val_idx = np.array(list(set(warm_val_idx.tolist()) - set(idx_to_move.tolist())), dtype=np.int64)
warm_emb_idx = np.concatenate([warm_emb_idx, idx_to_move], axis=0)

warm_emb_item_set = set(df.loc[warm_emb_idx, 'item'])
df_warm_val = df.loc[warm_val_idx]
idx_to_move = df_warm_val[True ^ df_warm_val['item'].isin(warm_emb_item_set)].index
warm_val_idx = np.array(list(set(warm_val_idx.tolist()) - set(idx_to_move.tolist())), dtype=np.int64)
warm_emb_idx = np.concatenate([warm_emb_idx, idx_to_move], axis=0)
timer.logging("Warm val splitting finished: {} -> {}".format(org_len, len(warm_val_idx)))

# Move the test records whose user/item don't emerge in emb set into emb set.
org_len = len(warm_test_idx)
warm_emb_user_set = set(df.loc[warm_emb_idx, 'user'])
df_warm_test = df.loc[warm_test_idx]
idx_to_move = df_warm_test[True ^ df_warm_test['user'].isin(warm_emb_user_set)].index
warm_test_idx = np.array(list(set(warm_test_idx.tolist()) - set(idx_to_move.tolist())), dtype=np.int64)
warm_emb_idx = np.concatenate([warm_emb_idx, idx_to_move], axis=0)

warm_emb_item_set = set(df.loc[warm_emb_idx, 'item'])
df_warm_test = df.loc[warm_test_idx]
idx_to_move = df_warm_test[True ^ df_warm_test['item'].isin(warm_emb_item_set)].index
warm_test_idx = np.array(list(set(warm_test_idx.tolist()) - set(idx_to_move.tolist())))
warm_emb_idx = np.concatenate([warm_emb_idx, idx_to_move], axis=0)
timer.logging("Warm test splitting finished: {} -> {}".format(org_len, len(warm_test_idx)))
timer.logging("Warm emb splitting finished: {} -> {}".format(org_warm_emb_len, len(warm_emb_idx)))

# store warm df
df_warm_emb = df.loc[warm_emb_idx]
df_warm_val = df.loc[warm_val_idx]
df_warm_test = df.loc[warm_test_idx]

df_warm_emb.to_csv(os.path.join(store_path, 'warm_emb.csv'), index=False)
df_warm_val.to_csv(os.path.join(store_path, 'warm_val.csv'), index=False)
df_warm_test.to_csv(os.path.join(store_path, 'warm_test.csv'), index=False)
timer.logging("Store warm subset finished.")

print('[warm]\tuser\titem\trecord')
print('emb\t%d\t%d\t%d' %
      (len(set(df_warm_emb['user'])), len(set(df_warm_emb['item'])), len(warm_emb_idx)))
print('val\t%d\t%d\t%d' %
      (len(set(df_warm_val['user'])), len(set(df_warm_val['item'])), len(warm_val_idx)))
print('test\t%d\t%d\t%d' %
      (len(set(df_warm_test['user'])), len(set(df_warm_test['item'])), len(warm_test_idx)))

"""cold subset splitting"""
# drop the records whose users don't emerge in the warm_emb set
org_cold_len = len(cold_idx)
warm_object = 'user' if args.cold_object == 'item' else 'item'
warm_object_set = set(df.loc[warm_emb_idx, warm_object])
df_cold = df.loc[cold_idx]
cold_idx = df_cold[df_cold[warm_object].isin(warm_object_set)].index
timer.logging("cold: {} -> {}".format(org_cold_len, len(cold_idx)))

# split cold val/test set by item ids
df_cold = df.loc[cold_idx]
cold_group = df_cold.groupby(by=args.cold_object)
cold_group = np.array([g[1].index for g in cold_group], dtype=object)
np.random.shuffle(cold_group)
n_cold_val_group = int(args.cold_split[0] * len(cold_group))
cold_val_idx = np.concatenate(cold_group[:n_cold_val_group].tolist(), axis=0)
cold_test_idx = np.concatenate(cold_group[n_cold_val_group:].tolist(), axis=0)
timer.logging("Cold subset splitting finished.")

# store cold df
df_cold_val = df.loc[cold_val_idx]
df_cold_test = df.loc[cold_test_idx]

df_cold_val.to_csv(os.path.join(store_path, f'cold_{args.cold_object}_val.csv'), index=False)
df_cold_test.to_csv(os.path.join(store_path, f'cold_{args.cold_object}_test.csv'), index=False)
timer.logging("Store cold subset finished.")

print('[cold]\tuser\titem\trecord')
print('val\t%d\t%d\t%d' %
      (len(set(df_cold_val['user'])), len(set(df_cold_val['item'])), len(cold_val_idx)))
print('test\t%d\t%d\t%d' %
      (len(set(df_cold_test['user'])), len(set(df_cold_test['item'])), len(cold_test_idx)))

df.to_csv(os.path.join(store_path, 'all.csv'), index=False)
timer.logging("Split finished.")
