import torch

from pp_query.modules import PPDecoder, HawkesDecoder, RMTPPDecoder, TemporalEmbedding
from pp_query.models.model import PPModel
from pp_query.models.hawkes import HawkesModel

def get_model(
    channel_embedding_size,
    num_channels,
    dec_recurrent_hidden_size,
    dec_num_recurrent_layers,
    dec_intensity_hidden_size,
    dec_intensity_factored_heads,
    dec_num_intensity_layers,
    dec_intensity_use_embeddings,
    dec_act_func="gelu",
    dropout=0.2,
    hawkes=False,
    hawkes_bounded=True,
    neural_hawkes=False,
    rmtpp=False,
):
        
    if hawkes:
        return HawkesModel(
            num_marks=num_channels,
            bounded=hawkes_bounded,
        )

    channel_embedding = torch.nn.Embedding(
        num_embeddings=num_channels, 
        embedding_dim=channel_embedding_size
    )

    if neural_hawkes:
        decoder = HawkesDecoder(
            channel_embedding=channel_embedding,
            time_embedding=TemporalEmbedding(
                embedding_dim=1,
                use_raw_time=False,
                use_delta_time=True,
                learnable_delta_weights=False,
                max_period=0,
            ),
            recurrent_hidden_size=dec_recurrent_hidden_size,
        )
    elif rmtpp:
        decoder = RMTPPDecoder(
            channel_embedding=channel_embedding,
            time_embedding=TemporalEmbedding(
                embedding_dim=1,
                use_raw_time=False,
                use_delta_time=True,
                learnable_delta_weights=False,
                max_period=0,
            ),
            recurrent_hidden_size=dec_recurrent_hidden_size,
        )
    else:
        decoder = PPDecoder(
            channel_embedding=channel_embedding,
            act_func=dec_act_func,
            num_intensity_layers=dec_num_intensity_layers,
            intensity_hidden_size=dec_intensity_hidden_size,
            num_recurrent_layers=dec_num_recurrent_layers,
            recurrent_hidden_size=dec_recurrent_hidden_size,
            dropout=dropout,
            factored_heads=dec_intensity_factored_heads,
            use_embedding_weights=dec_intensity_use_embeddings,
        )

    return PPModel(
        decoder=decoder,
    )
