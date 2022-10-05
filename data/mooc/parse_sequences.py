import argparse
import random
import numpy as np
import pandas as pd
import json

from collections import defaultdict
from tqdm import tqdm

random.seed(0)
np.random.seed(0)

BASE_TIME = 0.0
MAX_TIME = 2572086.0
DROP_OUT_MARK = 100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat_dir', default='./data/', type=str, help="Directory where 'mooc_actions.tsv' and 'mooc_action_labels.tsv' are located.")
    parser.add_argument('--out_dir', default='./', type=str, help="Directory where results will be saved.")
    parser.add_argument('--min_events', default=5, type=int, help="Minimum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--max_events', default=200, type=int, help="Maximum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--time_norm', default=60*60, type=int, help="Number of seconds per desired unit of time. Default is for 1 unit == 1 hour.")
    parser.add_argument('--valid_pct', default=0.1, type=float, help="Percentage of sequences to set aside for validation split.")
    parser.add_argument('--test_pct', default=0.15, type=float, help="Percentage of sequences to set aside for test split.")
    args = parser.parse_args()
    args.dat_dir = args.dat_dir.rstrip("/")
    args.out_dir = args.out_dir.rstrip("/")
    return args

def collect_sequences(args):
    drop_out_label_df = pd.read_csv("{}/mooc_action_labels.tsv".format(args.dat_dir), sep='\t')
    drop_out_label = drop_out_label_df['LABEL'].to_dict()  # {actionid: 1 or 0 for whether dropout after the event}
    first_line = True
    sequences = defaultdict(list)
    with open("{}/mooc_actions.tsv".format(args.dat_dir)) as f:
        for event in tqdm(f):
            if first_line:
                first_line = False
                continue
            action_id, user_id, mark_id, timestamp = event.strip().split("\t")
            action_id, mark_id, timestamp = int(action_id), int(mark_id), float(timestamp)
            if timestamp < BASE_TIME:
                continue
            # check events happen at the same time and shift accordingly
            if sequences[user_id] and timestamp <= sequences[user_id][-1][0]:
                timestamp = sequences[user_id][-1][0] + 0.5
            sequences[user_id].append((timestamp, mark_id))
            if drop_out_label[int(action_id)]:
                sequences[user_id].append((timestamp, DROP_OUT_MARK))  # add a label to set T as the last event time
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
        "T": (MAX_TIME - BASE_TIME) / args.time_norm if sequence[1][-1] != 100 else sequence[0][-1],
        "times": sequence[0] if sequence[1][-1] != 100 else sequence[0][:-1],
        "marks": sequence[1] if sequence[1][-1] != 100 else sequence[1][:-1],
    }) + "\n"



if __name__=="__main__":
    args = get_args()
    print("Gathering Event Sequences")
    sequences = collect_sequences(args)
    print("Processing Event Sequences")
    sequences = process_sequences(args, sequences)
    # print(sequences)


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
        ("{}/train/train_mooc.jsonl", train_strs),
        ("{}/valid/valid_mooc.jsonl", valid_strs),
        ("{}/test/test_mooc.jsonl", test_strs),
    ]

    for dir, strs in dir_strs:
        with open(dir.format(args.out_dir), "w") as f:
            f.writelines(strs)