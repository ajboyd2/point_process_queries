"""
Command-line utility for training a model
"""
import os
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad
from tqdm import tqdm

from pp_query.data import PointPatternDataset, pad_and_combine_instances
from pp_query.models import get_model
from pp_query.optim import get_optimizer, get_lr_scheduler
from pp_query.arguments import get_args
from pp_query.utils import print_log


def forward_pass(args, batch, model, sample_timestamps=None, num_samples=150, get_raw_likelihoods=False):
    if args.cuda:
        batch = {k:v.cuda(torch.cuda.current_device()) for k,v in batch.items()}

    padding_mask = batch["padding_mask"]
    tgt_marks, tgt_timestamps = batch["tgt_marks"], batch["tgt_times"]
    pp_id = batch["pp_id"]

    T = batch["T"]  

    if sample_timestamps is None:
        sample_timestamps = torch.rand(
            tgt_timestamps.shape[0], 
            num_samples, 
            dtype=tgt_timestamps.dtype, 
            device=tgt_timestamps.device
        ).clamp(min=1e-8) * T # ~ U(0, T)

    # Forward Pass
    results = model(
        marks=tgt_marks, 
        timestamps=tgt_timestamps, 
        sample_timestamps=sample_timestamps,
    )

    # Calculate losses
    ll_results = model.log_likelihood(
        return_dict=results, 
        right_window=T, 
        left_window=0.0, 
        mask=padding_mask,
        reduce=not get_raw_likelihoods,
        normalize_by_window=args.normalize_by_window,
    )


    if get_raw_likelihoods:
        return ll_results, sample_timestamps, tgt_timestamps

    log_likelihood, ll_pos_contrib, ll_neg_contrib = \
        ll_results["log_likelihood"], ll_results["positive_contribution"], ll_results["negative_contribution"]

    loss = -1 * log_likelihood  # minimize loss, maximize log likelihood

    return {
        "loss": loss,
        "log_likelihood": log_likelihood,
        "ll_pos": ll_pos_contrib,
        "ll_neg": ll_neg_contrib,
    }, results

def backward_pass(args, loss, model, optimizer):
    
    optimizer.zero_grad()

    if torch.isnan(loss).any().item():
        return False
    else:
        loss.backward()
        
        clip_grad(parameters=model.parameters(), max_norm=args.grad_clip, norm_type=2)
        return True

def train_step(args, model, optimizer, lr_scheduler, batch):

    loss_results, forward_results = forward_pass(args, batch, model)

    if backward_pass(args, loss_results["loss"], model, optimizer):
        optimizer.step()
        lr_scheduler.step()
    else:
        print_log('======= NAN-Loss =======')
        print_log("Loss Results:", {k:(torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k,v in loss_results.items() if isinstance(v, torch.Tensor)})
        print_log("Loss Results:", loss_results)
        print_log("")
        print_log("Batch:", {k:(torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k,v in batch.items() if isinstance(v, torch.Tensor)})
        print_log("Batch:", batch)
        print_log("")
        print_log("Results:", {k:(torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k,v in forward_results["state_dict"].items()})
        print_log("Results:", {k:(torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k,v in forward_results["intensities"].items()})
        print_log("Results:", {k:(torch.isnan(v).any().item(), v.min().item(), v.max().item()) for k,v in forward_results["sample_intensities"].items()})
        print_log("Results:", forward_results)
        print_log("========================")
        input()

    return loss_results

def train_epoch(args, model, optimizer, lr_scheduler, dataloader, epoch_number):
    model.train()

    total_losses = defaultdict(lambda: 0.0)
    data_len = len(dataloader)
    for i, batch in enumerate(dataloader):
        batch_loss = train_step(args, model, optimizer, lr_scheduler, batch)
        for k, v in batch_loss.items():
            total_losses[k] += v.item()
        if ((i+1) % args.log_interval == 0) or ((i+1 <= 5) and (epoch_number<=1)):
            items_to_print = [("LR", lr_scheduler.get_lr())]
            items_to_print.extend([(k,v/args.log_interval) for k,v in total_losses.items()])
            print_results(args, items_to_print, epoch_number, i+1, data_len, True)
            total_losses = defaultdict(lambda: 0.0)

    if (i+1) % args.log_interval != 0:
        items_to_print = [("LR", lr_scheduler.get_lr())]
        items_to_print.extend([(k,v/(i % args.log_interval)) for k,v in total_losses.items()])
        print_results(args, items_to_print, epoch_number, i+1, data_len, True)

    return {k:v/data_len for k,v in total_losses.items()}

def eval_step(args, model, batch, num_samples=150):
    return forward_pass(args, batch, model, num_samples=num_samples)

def eval_epoch(args, model, eval_dataloader, train_dataloader, epoch_number, num_samples=150):
    model.eval()

    with torch.no_grad():
        total_losses = defaultdict(lambda: 0.0)
        data_len = len(eval_dataloader)
        valid_latents, valid_labels = [], []
        for i, batch in enumerate(eval_dataloader):
            batch_loss, results = eval_step(args, model, batch, num_samples)
            
            for k, v in batch_loss.items():
                total_losses[k] += v.item()

    print_results(args, [(k,v/data_len) for k,v in total_losses.items()], epoch_number, i+1, data_len, False)

    return {k:v/data_len for k,v in total_losses.items()}
        
def print_results(args, items, epoch_number, iteration, data_len, training=True):
    msg = "[{}] Epoch {}/{} | Iter {}/{} | ".format("T" if training else "V", epoch_number, args.train_epochs, iteration, data_len)
    msg += "".join("{} {:.4E} | ".format(k, v) for k,v in items)
    print_log(msg)

def set_random_seed(args=None, seed=None):
    """Set random seed for reproducibility."""

    if (seed is None) and (args is not None):
        seed = args.seed

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def setup_model_and_optim(args, epoch_len):
    model = get_model(
        channel_embedding_size=args.channel_embedding_size,
        num_channels=args.num_channels,
        dec_recurrent_hidden_size=args.dec_recurrent_hidden_size,
        dec_num_recurrent_layers=args.dec_num_recurrent_layers,
        dec_intensity_hidden_size=args.dec_intensity_hidden_size,
        dec_intensity_factored_heads=args.dec_intensity_factored_heads,
        dec_num_intensity_layers=args.dec_num_intensity_layers,
        dec_intensity_use_embeddings=args.dec_intensity_use_embeddings,
        dec_act_func=args.dec_act_func,
        dropout=args.dropout,
        hawkes=args.use_hawkes,
        hawkes_bounded=args.hawkes_bounded,
        neural_hawkes=args.neural_hawkes,
        rmtpp=args.rmtpp,
        dyn_dom_buffer=args.dyn_dom_buffer,
    )

    if args.cuda:
        print(f"USING GPU {args.device_num}")
        torch.cuda.set_device(args.device_num)
        model.cuda(torch.cuda.current_device())

    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_len)

    return model, optimizer, lr_scheduler

def get_data(args):
    train_dataset = PointPatternDataset(
        file_path=args.train_data_path, 
        args=args, 
        keep_pct=args.train_data_percentage, 
        set_dominating_rate=args.evaluate,
        is_test=False,
    )
    args.num_channels = train_dataset.vocab_size
    if args.force_num_channels is not None:
        args.num_channels = args.force_num_channels

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=lambda x: pad_and_combine_instances(x, train_dataset.max_period),
        drop_last=True,
    )

    args.max_period = train_dataset.get_max_T() / 2.0

    print_log("Loaded {} / {} training examples / batches from {}".format(len(train_dataset), len(train_dataloader), args.train_data_path))

    if args.do_valid:
        valid_dataset = PointPatternDataset(
            file_path=args.valid_data_path, 
            args=args, 
            keep_pct=1.0, 
            set_dominating_rate=False,
            is_test=False,
        )

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda x: pad_and_combine_instances(x, valid_dataset.max_period),
            drop_last=True,
        )
        print_log("Loaded {} / {} validation examples / batches from {}".format(len(valid_dataset), len(valid_dataloader), args.valid_data_path))

        test_dataset = PointPatternDataset(
            file_path=args.test_data_path, 
            args=args, 
            keep_pct=1.0,  # object accounts for the test set having (1 - valid_to_test_pct) amount
            set_dominating_rate=False,
            is_test=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=max(args.batch_size // 4, 1),
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=lambda x: pad_and_combine_instances(x, test_dataset.max_period),
            drop_last=True,
            pin_memory=args.pin_test_memory,
        )
        print_log("Loaded {} / {} test examples / batches from {}".format(len(test_dataset), len(test_dataloader), args.valid_data_path))
    else:
        valid_dataloader = None
        test_dataloader = None


    return train_dataloader, valid_dataloader, test_dataloader 

def save_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    # Create folder if not already created
    folder_path = args.checkpoint_path
    folders = folder_path.split("/")
    for i in range(len(folders)):
        if folders[i] == "":
            continue
        intermediate_path = "/".join(folders[:i+1])
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)

    final_path = "{}/model_{:03d}.pt".format(folder_path.rstrip("/"), epoch)
    if os.path.exists(final_path):
        os.remove(final_path)
    torch.save(model.state_dict(), final_path)
    print_log("Saved model at {}".format(final_path))

def load_checkpoint(args, model):
    folder_path = args.checkpoint_path
    if not os.path.exists(folder_path):
        print_log(f"Checkpoint path [{folder_path}] does not exist.")
        return 0

    print_log(f"Checkpoint path [{folder_path}] does exist.")
    files = [f for f in os.listdir(folder_path) if ".pt" in f]
    if len(files) == 0:
        print_log("No .pt files found in checkpoint path.")
        return 0

    latest_model = sorted(files)[-1]
    file_path = "{}/{}".format(folder_path.rstrip("/"), latest_model)

    if not os.path.exists(file_path):
        print_log(f"File [{file_path}] not found.")
        return 0

    model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
    if args.cuda:
        model.cuda(torch.cuda.current_device())
    print_log("Loaded model from {}".format(file_path))
    return int(latest_model.replace("model_", "").replace(".pt", "")) + 1

def report_model_stats(model):
    encoder_parameter_count = 0
    aggregator_parameter_count = 0
    decoder_parameter_count = 0
    total = 0 
    for name, param in model.named_parameters():
        if name.startswith("encoder"):
            encoder_parameter_count += param.numel()
        elif name.startswith("aggregator"):
            aggregator_parameter_count += param.numel()
        else:
            decoder_parameter_count += param.numel()
        total += param.numel()

    print_log()
    print_log("<Total Parameter Count>: {}".format(decoder_parameter_count))
    print_log()

def main(args):
    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    train_dataloader, valid_dataloader, test_dataloader = get_data(args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, optimizer, lr_scheduler = setup_model_and_optim(args, len(train_dataloader))

    report_model_stats(model)

    if args.finetune:
        epoch = load_checkpoint(args, model)
    else:
        epoch = 0
    original_epoch = epoch

    print_log("Starting training.")
    results = {"valid": [], "train": [], "test": []}
    last_valid_ll = -float('inf')
    epsilon = 0.03

    while epoch < args.train_epochs or args.early_stop:
        results["train"].append(train_epoch(args, model, optimizer, lr_scheduler, train_dataloader, epoch+1))

        if args.do_valid and ((epoch+1) % args.valid_epochs == 0):
            new_valid = eval_epoch(args, model, valid_dataloader, train_dataloader, epoch+1)
            results["valid"].append(new_valid)
            if args.early_stop:
                if new_valid["log_likelihood"] - last_valid_ll < epsilon:
                    break
            last_valid_ll = new_valid["log_likelihood"]


        if ((epoch+1) % args.save_epochs == 0):
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch)
        
        epoch += 1
        
    if args.save_epochs > 0 and original_epoch != epoch:
        save_checkpoint(args, model, optimizer, lr_scheduler, epoch)

    if args.do_valid:
        overall_test_results = {}
        reps = 5    
        for _ in range(reps):
            test_results = eval_epoch(args, model, test_dataloader, train_dataloader, epoch+1, num_samples=500)
            results["test"].append(test_results)
    
    del model
    del optimizer
    del lr_scheduler
    del train_dataloader
    del valid_dataloader
    del test_dataloader
    torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    print_log("Getting arguments.")
    args = get_args()
    main(args)
