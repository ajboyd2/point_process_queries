import argparse
import csv
import random
import numpy as np
import pandas as pd
import json
import datetime
import zipfile
from io import TextIOWrapper

from collections import defaultdict
from tqdm import tqdm

random.seed(0)
np.random.seed(0)

BASE_TIME = datetime.datetime(2017, 11, 25, 0, 0).timestamp()
MAX_TIME = datetime.datetime(2017, 12, 3, 0, 0).timestamp()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat_dir', default='./data/', type=str, help="Directory where 'mooc_actions.tsv' and 'mooc_action_labels.tsv' are located.")
    parser.add_argument('--out_dir', default='./', type=str, help="Directory where results will be saved.")
    parser.add_argument('--min_events', default=5, type=int, help="Minimum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--max_events', default=200, type=int, help="Maximum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--time_norm', default=60*60, type=int, help="Number of seconds per desired unit of time. Default is for 1 unit == 1 hour.")
    parser.add_argument('--valid_pct', default=0.1, type=float, help="Percentage of sequences to set aside for validation split.")
    parser.add_argument('--test_pct', default=0.15, type=float, help="Percentage of sequences to set aside for test split.")
    # 1*10^6 -> 8868; 1.2*10^6 -> 10614; 2*10^6 -> 17777
    parser.add_argument('--max_lines', default=2_000_000, type=int, help="Read only top k lines of the csv file.")
    parser.add_argument('--top_k_marks', default=1000, type=int, help="Consider top k marks that have the most frequency.")
    args = parser.parse_args()
    args.dat_dir = args.dat_dir.rstrip("/")
    args.out_dir = args.out_dir.rstrip("/")
    return args

def get_mark_vals(args):
    zf = zipfile.ZipFile('{}/UserBehavior.csv.zip'.format(args.dat_dir))
    df = pd.read_csv(zf.open('UserBehavior.csv'), nrows=args.max_lines, header=None)
    df.columns = ['user_id', 'item_id', 'mark_id', 'action', 'timestamp']
    df = df[(df['timestamp'] >= BASE_TIME) & (df['timestamp'] <= MAX_TIME) & (df['action'] == 'pv')]
    mark_val, mark_count = np.unique(df['mark_id'].to_numpy(), return_counts=True)
    mark_least_count = sorted(mark_count, reverse=True)[:args.top_k_marks][-1]
    mark_mask = (mark_count >= mark_least_count)
    mark_vals = mark_val[mark_mask]
    orig_mark_to_new_mark = {v:i for i,v in enumerate(mark_vals)}
    return mark_vals, orig_mark_to_new_mark


def collect_sequences(args, mark_vals, orig_mark_to_new_mark):
    mark_vals = set(mark_vals)
    sequences = defaultdict(list)
    count = args.max_lines

    with zipfile.ZipFile('{}/UserBehavior.csv.zip'.format(args.dat_dir)) as zf:
        with zf.open('UserBehavior.csv', 'r') as f:
            reader = csv.reader(TextIOWrapper(f, 'utf-8'))
            for i, line in tqdm(enumerate(reader)):
                if i >= count:
                    break
                else:
                    user_id, _, mark_id, action, timestamp = line
                    if action != 'pv':  # skip other actions, only consider page view
                        continue
                    user_id, mark_id, timestamp = str(user_id), int(mark_id), float(timestamp)

                    if mark_id not in mark_vals:
                        continue
                    if timestamp < BASE_TIME or timestamp > MAX_TIME:
                        continue
                    if sequences[user_id] and timestamp <= sequences[user_id][-1][0]:
                        timestamp = sequences[user_id][-1][0] + 0.5
                    sequences[user_id].append((timestamp, orig_mark_to_new_mark[mark_id]))
    return sequences

def process_sequences(args, sequences):
    print("   Num Sequences prior to Filtering:", len(sequences))
    for u in tqdm(list(sequences.keys())):
        if args.min_events <= len(sequences[u]) <= args.max_events:
            sequences[u] = list(zip(*sorted(sequences[u], key=lambda x: x[0])))  # Sort by timestamps and then split into two lists: one for timestmaps and one for marks
            sequences[u][0] = [(t - BASE_TIME) / args.time_norm for t in sequences[u][0]]
        else:
            del sequences[u]
    print("   Num Sequences after Filtering:", len(sequences))
    return sequences


def format_sequence(args, sequence, user_id):
    return json.dumps({
        "user": user_id,
        "T": (MAX_TIME - BASE_TIME) / args.time_norm,
        "times": sequence[0],
        "marks": sequence[1],
    }) + "\n"



if __name__=="__main__":
    args = get_args()
    print("Get top {} mark values".format(args.top_k_marks))
    mark_vals, orig_mark_to_new_mark = get_mark_vals(args)
    print("Gathering Event Sequences")
    sequences = collect_sequences(args, mark_vals, orig_mark_to_new_mark)
    print("Processing Event Sequences")
    sequences = process_sequences(args, sequences)

    all_strs, train_strs, valid_strs, test_strs = [], [], [], []

    sequences = list(sequences.items())
    random.shuffle(sequences)

    for i, (user_id, sequence) in tqdm(enumerate(sequences)):
        seq_str = format_sequence(args, sequence, user_id)
        all_strs.append(seq_str)
        progress = (i + 1) / len(sequences)
        if progress <= args.test_pct:
            test_strs.append(seq_str)
        elif progress <= args.test_pct + args.valid_pct:
            valid_strs.append(seq_str)
        else:
            train_strs.append(seq_str)

    dir_strs = [
        ("{}/all_sequences.jsonl", all_strs),
        ("{}/train/train_taobao.jsonl", train_strs),
        ("{}/valid/valid_taobao.jsonl", valid_strs),
        ("{}/test/test_taobao.jsonl", test_strs),
    ]

    for dir, strs in dir_strs:
        with open(dir.format(args.out_dir), "w") as f:
            f.writelines(strs)
    
    json.dump({int(k):int(v) for k,v in orig_mark_to_new_mark.items()}, open("{}/orig_mark_mappings.json".format(args.out_dir), "w"))
