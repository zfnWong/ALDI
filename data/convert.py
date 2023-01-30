import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
import os
import sys

sys.path.append("..")
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CiteULike", help='Dataset to use.')
parser.add_argument('--datadir', type=str, default="./", help='Director of the dataset.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--cold_object', type=str, default='item')
args = parser.parse_args()
pprint(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)

store_path = os.path.join(args.datadir, f"{args.dataset}/")
procedure_timer = utils.Timer("Convert")


"""Read data from file"""
df_emb = pd.read_csv(os.path.join(store_path, 'warm_emb.csv'))
df_warm_val = pd.read_csv(os.path.join(store_path, 'warm_val.csv'))
df_warm_test = pd.read_csv(os.path.join(store_path, 'warm_test.csv'))
df_cold_val = pd.read_csv(os.path.join(store_path, f'cold_{args.cold_object}_val.csv'))
df_cold_test = pd.read_csv(os.path.join(store_path, f'cold_{args.cold_object}_test.csv'))
df_pos = pd.read_csv(os.path.join(store_path, 'all.csv'))


"""Build overall validation/test set"""
overall_val_user_set = np.array(list(set(df_cold_val['user']) & set(df_warm_val['user'])), dtype=np.int32)
df_overall_val = pd.concat([df_cold_val, df_warm_val])
df_overall_val = df_overall_val[df_overall_val['user'].isin(overall_val_user_set)]

overall_test_user_set = np.array(list(set(df_cold_test['user']) & set(df_warm_test['user'])), dtype=np.int32)
df_overall_test = pd.concat([df_cold_test, df_warm_test])
df_overall_test = df_overall_test[df_overall_test['user'].isin(overall_test_user_set)]

# user_num, item_num
n_user_item = pickle.load(open(os.path.join(store_path, 'n_user_item.pkl'), 'rb'))
user_num = n_user_item['user']
item_num = n_user_item['item']
procedure_timer.logging('Finish loading data.')
print("Global user_num: {}  item_num: {}".format(user_num, item_num))


"""Get testing users"""
emb_user = np.array(list(set(df_emb['user'])), dtype=np.int32)
warm_val_user = np.array(list(set(df_warm_val['user'])), dtype=np.int32)
warm_test_user = np.array(list(set(df_warm_test['user'])), dtype=np.int32)
cold_val_user = np.array(list(set(df_cold_val['user'])), dtype=np.int32)
cold_test_user = np.array(list(set(df_cold_test['user'])), dtype=np.int32)
overall_val_user = np.array(list(overall_val_user_set), dtype=np.int32)
overall_test_user = np.array(list(overall_test_user_set), dtype=np.int32)
procedure_timer.logging('Finish getting testing users.')


"""Get testing items"""
emb_item = np.array(list(set(df_emb['item'])), dtype=np.int32)
warm_val_item = np.array(list(set(df_warm_val['item'])), dtype=np.int32)
warm_test_item = np.array(list(set(df_warm_test['item'])), dtype=np.int32)
cold_val_item = np.array(list(set(df_cold_val['item'])), dtype=np.int32)
cold_test_item = np.array(list(set(df_cold_test['item'])), dtype=np.int32)
overall_val_item = np.array(list(set(df_overall_val['item'])), dtype=np.int32)
overall_test_item = np.array(list(set(df_overall_test['item'])), dtype=np.int32)
procedure_timer.logging('Finish getting testing items.')


"""Generate users' neighboring items."""
emb_user_nb = utils.df_get_neighbors(df_emb, 'user', user_num)  # index [item_array]
warm_val_user_nb = utils.df_get_neighbors(df_warm_val, 'user', user_num)
warm_test_user_nb = utils.df_get_neighbors(df_warm_test, 'user', user_num)
cold_val_user_nb = utils.df_get_neighbors(df_cold_val, 'user', user_num)
cold_test_user_nb = utils.df_get_neighbors(df_cold_test, 'user', user_num)
pos_user_nb = utils.df_get_neighbors(df_pos, 'user', user_num)
overall_val_user_nb = utils.df_get_neighbors(df_overall_val, 'user', user_num)
overall_test_user_nb = utils.df_get_neighbors(df_overall_test, 'user', user_num)
procedure_timer.logging('Finish getting users\' neighbors.')


"""Generate items' neighboring users."""
emb_item_nb = utils.df_get_neighbors(df_emb, 'item', item_num)
procedure_timer.logging('Finish getting items\' neighbors.')


"""Statistics"""
user_array = np.arange(user_num, dtype=np.int32)
item_array = np.arange(item_num, dtype=np.int32)
warm_user = np.array(list(set(df_emb['user'].tolist())), dtype=np.int32)
warm_item = np.array(list(set(df_emb['item'].tolist())), dtype=np.int32)
cold_user = np.array(list(set(user_array.tolist()) - set(warm_user.tolist())), dtype=np.int32)
cold_item = np.array(list(set(item_array.tolist()) - set(warm_item.tolist())), dtype=np.int32)
procedure_timer.logging('Finish generating warm/cold item/user array.')
print("[warm] user: {}  item: {}".format(len(warm_user), len(warm_item)))
print("[cold] user: {}  item: {}".format(len(cold_user), len(cold_item)))


"""Save results"""
para_dict = {}
para_dict['user_num'] = user_num
para_dict['item_num'] = item_num
para_dict['user_array'] = user_array
para_dict['item_array'] = item_array
para_dict['warm_user'] = warm_user
para_dict['warm_item'] = warm_item
para_dict['cold_user'] = cold_user
para_dict['cold_item'] = cold_item

para_dict['emb_user'] = emb_user
para_dict['warm_val_user'] = warm_val_user
para_dict['warm_test_user'] = warm_test_user
para_dict['cold_val_user'] = cold_val_user
para_dict['cold_test_user'] = cold_test_user
para_dict['overall_val_user'] = overall_val_user
para_dict['overall_test_user'] = overall_test_user

para_dict['emb_item'] = emb_item
para_dict['warm_val_item'] = warm_val_item
para_dict['warm_test_item'] = warm_test_item
para_dict['cold_val_item'] = cold_val_item
para_dict['cold_test_item'] = cold_test_item
para_dict['overall_val_item'] = overall_val_item
para_dict['overall_test_item'] = overall_test_item

para_dict['pos_user_nb'] = pos_user_nb
para_dict['emb_user_nb'] = emb_user_nb  # index [item_array]
para_dict['warm_val_user_nb'] = warm_val_user_nb
para_dict['warm_test_user_nb'] = warm_test_user_nb
para_dict['cold_val_user_nb'] = cold_val_user_nb
para_dict['cold_test_user_nb'] = cold_test_user_nb
para_dict['overall_val_user_nb'] = overall_val_user_nb
para_dict['overall_test_user_nb'] = overall_test_user_nb

para_dict['emb_item_nb'] = emb_item_nb  # for item frequency

dict_path = os.path.join(store_path, 'convert_dict.pkl')
pickle.dump(para_dict, open(dict_path, 'wb'), protocol=4)
procedure_timer.logging('Convert {} successfully, store the dict to {}'.format(args.dataset, dict_path))
