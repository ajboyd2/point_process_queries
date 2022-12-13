from itertools import filterfalse
import torch
import torch.nn.functional as F
import math

from tqdm import tqdm
from time import perf_counter

from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, CensoredPP, UnbiasedHittingTimeQuery, MarginalMarkQuery, ABeforeBQuery

set_random_seed(seed=123)

args = get_args()
args.cuda = False #True    
# args.device = torch.device("cuda:0") #cpu")
args.device = torch.device("cpu")
# args.cuda = False    
# args.device = torch.device("cpu")

args.train_data_percentage = 1.0
args.num_channels = 182
args.channel_embedding_size = 32
args.dec_recurrent_hidden_size = 64
args.dec_num_recurrent_layers = 1
args.dec_intensity_hidden_size = 32
args.dec_num_intensity_layers = 2
args.dec_act_func = "gelu"
args.dropout = 0.01
args.checkpoint_path = "./data/movie/nhp_models/"
args.train_epochs = 100
args.num_workers = 2
args.batch_size = 128
args.log_interval = 500
args.save_epochs = 5
args.optimizer = "adam"
args.grad_clip = 10000.0
args.lr = 0.001
args.weight_decay = 0.0
args.warmup_pct = 0.01
args.lr_decay_style = "constant"
args.train_data_path = ["./data/movie/train/"]
args.valid_data_path = ["./data/movie/valid/"]
args.test_data_path = ["./data/movie/test/"]
args.neural_hawkes = True


# Load data first as this determines the number of total marks
args.batch_size = 1
dl, _, _ = get_data(args)

# m, _, _ = setup_model_and_optim(args, 1000)
# load_checkpoint(args, m)

m = get_model(
    channel_embedding_size=10,
    num_channels=4,
    dec_recurrent_hidden_size=10,
    dec_num_recurrent_layers=1,
    dec_intensity_hidden_size=10,
    dec_intensity_factored_heads=False,
    dec_num_intensity_layers=1,
    dec_intensity_use_embeddings=False,
    dec_act_func="gelu",
    dropout=0.0,
    hawkes=True,
    hawkes_bounded=True,
    neural_hawkes=False,
    rmtpp=False,
)

m.eval()

print(m)