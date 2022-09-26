import argparse
import random
from typing import final
import numpy
import datetime
import json

from collections import defaultdict
from tqdm import tqdm

random.seed(0)
numpy.random.seed(0)

CATEGORIES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", 
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", 
    "Thriller", "War", "Western",
]

BASE_TIME = datetime.datetime(2015,1,1,0,0,0).timestamp()
MAX_TIME = datetime.datetime(2019,11,22,0,0,0).timestamp()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_genres', default=2, type=int, help="The number of genres that are taken for a given movie. If a movie has more than that many genres, then a `num_genres` amount will be randomly selected.")
    parser.add_argument('--dat_dir', default='./ml-25m/', type=str, help="Directory where 'movies.dat', 'ratings.dat', and 'users.dat' are located.")
    parser.add_argument('--out_dir', default='./', type=str, help="Directory where results will be saved.")
    parser.add_argument('--min_events', default=5, type=int, help="Minimum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--max_events', default=200, type=int, help="Maximum number of events needed to have a valid sequence for a user.")
    parser.add_argument('--time_norm', default=60*60, type=int, help="Number of seconds per desired unit of time. Default is for 1 unit == 1 hour.")
    parser.add_argument('--valid_pct', default=0.1, type=float, help="Percentage of sequences to set aside for validation split.")
    parser.add_argument('--test_pct', default=0.15, type=float, help="Percentage of sequences to set aside for test split.")
    parser.add_argument('--end_window_is_max_time', action="store_true", help="If enabled, the end of a sequence window is the max global time (November 2019). Otherwise, the last event time is also the end of the observation window.")
    parser.add_argument('--start_window_is_base_time', action="store_true", help="If enabled, the start of a sequence window is the base global time (January 2015). Otherwise, the original first event time is the start of the observation window (but is not included to avoid identifiability issues when modeling as every sequence would start with an event at time=0).")
    args = parser.parse_args()
    args.dat_dir = args.dat_dir.rstrip("/")
    args.out_dir = args.out_dir.rstrip("/")
    if not args.start_window_is_base_time:  # Increment event amount bounds as the first event will be cutoff
        args.min_events += 1
        args.max_events += 1
    return args

def load_movies(args):
    "Format per line ==> `MovieID::Title::Genre1|Genre2|...`"
    category_ids = {}
    movie_genres = {}
    first_line = True
    with open('{}/movies.csv'.format(args.dat_dir), "r", encoding="UTF-8") as f: #encoding='ISO-8859-1') as f:
        for movie in tqdm(f):
            if first_line:
                first_line = False
                continue
            movie_id, _ = movie.strip().split(",", 1)
            _, genres = movie.strip().rsplit(",", 1)  # This avoids issues with movie titles that contain a comma
            genres = genres.split("|")
            random.shuffle(genres)
            final_genre_combo = tuple(sorted(genres[:args.num_genres]))
            if final_genre_combo not in category_ids:
                category_ids[final_genre_combo] = len(category_ids)
            movie_genres[movie_id] = category_ids[final_genre_combo]

    return category_ids, movie_genres


def collect_sequences(args, movie_genres):
    "Parse the ratings.dat file to collect sequences of events for users."
    "Format per line ==> `UserID::MovieID::Rating::Timestamp`"
    sequences = defaultdict(list)  # dictionary of list of tuples, first element is the timestamp of an event, second is the mark id
    first_line = True
    with open('{}/ratings.csv'.format(args.dat_dir), "r", encoding='UTF-8') as f:
        for event in tqdm(f):
            if first_line:
                first_line = False
                continue
            user_id, movie_id, _, timestamp = event.strip().split(",")
            timestamp = float(timestamp)
            if timestamp < BASE_TIME:
                continue
            
            sequences[user_id].append((timestamp, movie_genres[movie_id]))
    
    return sequences

def process_sequences(args, sequences):
    print("   Num Sequences prior to Filtering:", len(sequences))
    for u in tqdm(list(sequences.keys())):
        if args.min_events <= len(sequences[u]) <= args.max_events:
            sequences[u] = tuple(zip(*sorted(sequences[u], key=lambda x: x[0])))  # Sort by timestamps and then split into two lists: one for timestmaps and one for marks
            if args.start_window_is_base_time:
                sequences[u][0] = [(t - BASE_TIME) / args.time_norm for t in sequences[u][0]]
            else:
                sequences[u] = (
                    [(t - sequences[u][0][0]) / args.time_norm for t in sequences[u][0][1:]],
                    sequences[u][1][1:],
                )
        else:
            del sequences[u]
    print("   Num Sequences after Filtering:", len(sequences))
    return sequences
    
def format_sequence(args, sequence, user_id):
    return json.dumps({
        "user": user_id,
        "T": (MAX_TIME - BASE_TIME) / args.time_norm if args.end_window_is_max_time else sequence[0][-1],
        "times": sequence[0],
        "marks": sequence[1],
    }) + "\n"

if __name__=="__main__":
    args = get_args()
    print("Loading Movies")
    category_ids, movie_genres = load_movies(args)
    print("Gathering Event Sequences")
    sequences = collect_sequences(args, movie_genres)
    print("Processing Event Sequences")
    sequences = process_sequences(args, sequences)

    print("Saving Results")
    with open("{}/category_ids.json".format(args.out_dir), 'w') as f:
        category_ids = {"|".join(cats):id for cats,id in category_ids.items()}
        json.dump(category_ids, f)
    
    all_strs, train_strs, valid_strs, test_strs = [], [], [], []
    
    sequences = list(sequences.items())
    random.shuffle(sequences)

    for i, (user_id, sequence) in tqdm(enumerate(sequences)):
        seq_str = format_sequence(args, sequence, user_id) 
        all_strs.append(seq_str)
        progress = (i+1) / len(sequences)
        if progress <= args.test_pct:
            test_strs.append(seq_str)
        elif progress <= args.test_pct + args.valid_pct:
            valid_strs.append(seq_str)
        else:
            train_strs.append(seq_str)

    dir_strs = [
        ("{}/all_sequences.jsonl", all_strs), 
        ("{}/train/train_movie.jsonl", train_strs), 
        ("{}/valid/valid_movie.jsonl", valid_strs), 
        ("{}/test/test_movie.jsonl", test_strs),
    ]

    for dir, strs in dir_strs:
        with open(dir.format(args.out_dir), "w") as f:
            f.writelines(strs)
