import torch

from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, CensoredPP

set_random_seed(seed=123)

m = get_model(
    channel_embedding_size=16,
    num_channels=6,
    dec_recurrent_hidden_size=10,
    dec_num_recurrent_layers=1,
    dec_intensity_hidden_size=10,
    dec_intensity_factored_heads=False,
    dec_num_intensity_layers=1,
    dec_intensity_use_embeddings=False,
    dec_act_func="gelu",
    dropout=0.0,
    hawkes=False,
    hawkes_bounded=False,
    neural_hawkes=True,
    rmtpp=False,
)

print(m)

pmq = PositionalMarkQuery([[0,1,2], [0,1], [0]])
tmq = TemporalMarkQuery([1.0, 2.0, 4.5], [[0], [1], [2]])

print(pmq.naive_estimate(m, 1000))
print(pmq.estimate(m, 1000, 100))
print()
print(tmq.naive_estimate(m, 1000))
print(tmq.estimate(m, 1000, 100))