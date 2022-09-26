import argparse
import json

from .utils import print_log


def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    #group.add_argument("--json_config_path", default=None, help="Path to json file containing arguments to be parsed.")
    group.add_argument("--seed", type=int, default=1234321, help="Seed for all random processes.")
    group.add_argument("--dont_print_args", action="store_true", help="Specify to disable printing of arguments.")
    group.add_argument("--cuda", action="store_true", help="Convert model and data to GPU.")
    group.add_argument("--device_num", type=int, default=0, help="Should cuda be enabled, this is the GPU id to use.")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    group.add_argument("--channel_embedding_size", type=int, default=16, help="Size of mark embeddings.")
    group.add_argument("--num_channels", type=int, default=3, help="Number of different possible marks.")
    group.add_argument("--dec_recurrent_hidden_size", type=int, default=16, help="Hidden size for decoder GRU.")
    group.add_argument("--dec_num_recurrent_layers", type=int, default=1, help="Number of recurrent layers in decoder.")
    group.add_argument("--dec_intensity_factored_heads", action="store_true", help="If enabled, models the logged total intensity and proporation of marks separately.")
    group.add_argument("--dec_intensity_use_embeddings", action="store_true", help="If enabled, compute mark intensities by comparing to channel embeddings.")
    group.add_argument("--dec_intensity_hidden_size", type=int, default=16, help="Hidden size of intermediate layers in intensity network.")
    group.add_argument("--dec_num_intensity_layers", type=int, default=1, help="Number of layers in intensity network.")
    group.add_argument("--dec_act_func", type=str, default="gelu", help="Activation function to be used in intensity network.")
    group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate to be applied to all supported layers during training.")
    group.add_argument("--use_hawkes", action="store_true", help="Uses a parametric Hawkes process.")
    group.add_argument("--hawkes_bounded", action="store_true", help="Make Hawkes parameters non-negative.")
    group.add_argument("--neural_hawkes", action="store_true", help="Makes decoder a Neural Hawkes Process.")
    group.add_argument("--rmtpp", action="store_true", help="Makes decoder use the RMTPP architecture.")
    group.add_argument("--personalized_head", action="store_true", help="Adds a linear transformation of the user embedding to the intensity logits.")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    group.add_argument("--checkpoint_path", type=str, default="./", help="Path to folder that contains model checkpoints. Will take the most recent one.")
    group.add_argument("--finetune", action="store_true", help="Will load in a model from the checkpoint path to finetune.")
    group.add_argument("--train_data_percentage", type=float, default=1.0, help="Percentage (between 0 and 1) of training data to keep. Used for ablation studies with relation to amount of data.")
    group.add_argument("--train_epochs", type=int, default=40, help="Number of epochs to iterate over for training.")
    group.add_argument("--train_data_path", nargs="+", type=str, default=["./data/1_pp/training.pickle"], help="Path to training data file.")
    group.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for data loaders.")
    group.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
    group.add_argument("--log_interval", type=int, default=100, help="Number of batches to complete before printing intermediate results.")
    group.add_argument("--save_epochs", type=int, default=1, help="Number of training epochs to complete between model checkpoint saves.")
    group.add_argument("--optimizer", type=str, default="adam", help="Type of optimization algorithm to use.")
    group.add_argument("--grad_clip", type=float, default=1.0, help="Threshold for gradient clipping.")
    group.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    group.add_argument("--loss_monotonic", type=float, default=None, help="Loss scaling parameters annealing monotonic schedule.")
    group.add_argument("--loss_cyclical", type=float, default=None, help="Loss scaling parameters annealing monotonic schedule.")
    group.add_argument("--weight_decay", type=float, default=0.01, help="L2 coefficient for weight decay.")
    group.add_argument("--warmup_pct", type=float, default=0.01, help="Percentage of 'train_iters' to be spent ramping learning rate up from 0.")
    group.add_argument("--lr_decay_style", type=str, default="cosine", help="Decay style for the learning rate, after the warmup period.")
    group.add_argument("--dont_shuffle", action="store_true", help="Don't shuffle training and validation dataloaders.")
    group.add_argument("--early_stop", action="store_true", help="Does not predfine the number of epochs to run, but will instead stop when validation performance slows down or regresses.")
    group.add_argument("--normalize_by_window", action="store_true", help="Will inversely scale log likelihood values by window length to give approximately equal weight to every sequence regardless of observation window.")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    group.add_argument("--valid_data_path", nargs="+", type=str, default=["./data/1_pp/validation.pickle"], help="Path to validation data file.")
    group.add_argument("--test_data_path", nargs="+", type=str, default=["./data/1_pp/validation.pickle"], help="Path to testing data file.")
    group.add_argument("--valid_epochs", type=int, default=5, help="Number of epochs to execute validation on.")
    # group.add_argument("--valid_to_test_pct", type=float, default=0.3, help="Percentage of held out data to be used for validation, rest is used for testing at the end.")
    group.add_argument("--visualize", action="store_true", help="In evaluate.py selects the visualization script to run.")
    group.add_argument("--sample_generations", action="store_true", help="In evaluate.py selects the generations script to run.")
    group.add_argument("--next_event_prediction", action="store_true", help="In evaluate.py selects the next event prediction task to run.")
    group.add_argument("--anomaly_detection", action="store_true", help="In evaluate.py selects the anomaly detection task to run.")
    group.add_argument("--num_samples", type=int, default=1024, help="Number of sequences to generate samples from.")
    group.add_argument("--samples_per_sequence", type=int, default=1, help="Number of samples to generate per sequence.")
    group.add_argument("--top_k", type=int, default=0, help="Enables top_k sampling for marks.")
    group.add_argument("--top_p", type=float, default=0.0, help="Enables top_p sampling for marks.")
    group.add_argument("--likelihood_over_time", action="store_true", help="In evaluate.py analyzes likelihood over time for a given model.")
    group.add_argument("--likelihood_resolution", type=float, default=1.0, help="When likelihood_over_time is enabled, this defines the bucket width to bin likelihood differences into.")
    group.add_argument("--pin_test_memory", action="store_true", help="Pin memory for test dataloader.")
    #group.add_argument("--", type=, default=, help="")pin_test_memory

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    #group.add_argument("--", type=, default=, help="")

def print_args(args):
    max_arg_len = max(len(k) for k, v in args.items())
    key_set = sorted([k for k in args.keys()])
    for k in key_set:
        v = args[k]
        print_log("{} {} {}".format(
            k,
            "." * (max_arg_len + 3 - len(k)),
            v,
        ))

def get_args():
    parser = argparse.ArgumentParser()

    general_args(parser)
    model_config_args(parser)
    training_args(parser)
    evaluation_args(parser)
    sampling_args(parser)

    args = parser.parse_args()

    args.do_valid = args.valid_data_path != ""
    args.shuffle = not args.dont_shuffle

    if not args.dont_print_args:
        print_args(vars(args))

    return args
