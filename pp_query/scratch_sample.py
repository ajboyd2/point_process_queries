from itertools import filterfalse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from time import perf_counter

from pp_query.models import get_model
from pp_query.train import load_checkpoint, setup_model_and_optim, set_random_seed, get_data
from pp_query.arguments import get_args
from pp_query.query import PositionalMarkQuery, TemporalMarkQuery, CensoredPP, UnbiasedHittingTimeQuery, MarginalMarkQuery

set_random_seed(seed=123)
torch.use_deterministic_algorithms(True)

print("PLEASE WORK")

args = get_args()
# args.cuda = True    # <=== WE GET BAD RESULTS WHEN THIS IS TRUE
# args.device = torch.device("cuda:0")  # TODO: RESTORE PRINT_LOG
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

m, _, _ = setup_model_and_optim(args, 1000)
load_checkpoint(args, m)

# m = get_model(
#     channel_embedding_size=10,
#     num_channels=4,
#     dec_recurrent_hidden_size=10,
#     dec_num_recurrent_layers=1,
#     dec_intensity_hidden_size=10,
#     dec_intensity_factored_heads=False,
#     dec_num_intensity_layers=1,
#     dec_intensity_use_embeddings=False,
#     dec_act_func="gelu",
#     dropout=0.0,
#     hawkes=True,
#     hawkes_bounded=True,
#     neural_hawkes=False,
#     rmtpp=False,
# )

m.eval()

print(m)

n_seqs = 1000
n_pts = 1000

n = 3 
accepted_marks = torch.tensor([0, 3, 7, 8, 9, 13, 15, 24, 34, 39, 62, 104, 167]).to(args.device)  #[ 4,  7,  8, 12, 15, 16, 24, 25, 31, 53, 68, 82]) 
num_channels = 180
times = torch.tensor([[0.0005555556272156537, 0.002500000176951289, 0.0036111113149672747, 0.004444445017725229, 0.006111111491918564]]).to(args.device)
marks = torch.tensor([[31, 9, 16, 9, 29]]).to(args.device)

pmq = MarginalMarkQuery(n=n, marks_of_interest=accepted_marks, total_marks=num_channels, batch_size=128, device=args.device, use_tqdm=True)  # Choose marginal set to be half of the unique marks in the sequence

print("SEED", args.seed)
print("TIMES", times.shape, times.squeeze().cpu().tolist())
print("MARKS", marks.shape, marks.squeeze().cpu().tolist())
print("MARKS OF INTEREST", pmq.marks_of_interest.shape, pmq.marks_of_interest.squeeze().cpu().tolist())
print("MARK RESTRICTIONS", pmq.mark_restrictions)
print("MAX EVENTS", pmq.max_events)
print("RES POSITIONS", pmq.restricted_positions.shape, pmq.restricted_positions.squeeze().cpu().tolist())
print("BATCH SIZE", pmq.batch_size)
print("DEVICE", pmq.device)
print("TQDM", pmq.use_tqdm)

conditional_marks=marks
timestamps=conditional_times=times
model=m

cond_len = 0 if conditional_times is None else conditional_times.numel()
mark_mask = torch.ones((pmq.max_events+1+cond_len, model.num_channels,), dtype=torch.float32).to(pmq.device)  # +1 to have an unrestricted final mask
for i in range(cond_len, pmq.max_events+cond_len):
    mark_mask[i, pmq.mark_restrictions[i-cond_len]] = 0.0
mask_dict = {
    "positional_mark_restrictions": mark_mask,
}
# mask_dict = {}
# mark_mask = 1.0
if isinstance(mark_mask, (int, float)):
    pass
else:
    print("PMR MASK", mark_mask.shape, mark_mask.squeeze().cpu().tolist())

self = m
left_window=0.0 if conditional_times is None else conditional_times.max()
length_limit=pmq.max_events + (0 if conditional_times is None else conditional_times.numel())
timestamps=conditional_times
marks=conditional_marks
mark_mask=1.0
num_samples=n_seqs
mask_dict=mask_dict
adapt_dom_rate=True

dominating_rate=None
T=float('inf')
top_k=0
top_p=0.0
proposal_batch_size=n_pts#1024
dyn_dom_buffer=100
MAX_SAMPLE_BATCH_SIZE = 1024

assert((T < float('inf')) or (length_limit < float('inf')))
if mask_dict is None:
    mask_dict = {}
if dominating_rate is None:
    dominating_rate = self.dominating_rate
if marks is None:
    marks = torch.LongTensor([[]], device=next(self.parameters()).device)
    timestamps = torch.FloatTensor([[]], device=next(self.parameters()).device)
if isinstance(left_window, torch.Tensor):
    left_window = left_window.item()
if isinstance(T, torch.Tensor):
    T = T.item()
if length_limit == float('inf'):
    length_limit = torch.iinfo(torch.int64).max  # Maximum Long value

sample_lens = torch.zeros((num_samples,), dtype=torch.int64).to(next(self.parameters()).device) + timestamps.numel()
marks, timestamps = marks.expand(num_samples, *marks.shape[1:]), timestamps.expand(num_samples, *timestamps.shape[1:])
time_pad, mark_pad = torch.nan_to_num(torch.tensor(float('inf'), dtype=timestamps.dtype)).item(), 0
state = self.forward(marks, timestamps)
state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
batch_idx = torch.arange(num_samples).to(state_values.device)
finer_proposal_batch_size = max(proposal_batch_size // 4, 16)

dist = torch.distributions.Exponential(dominating_rate)
dist.rate = dist.rate.to(state_values.device)*0+1  # We will manually apply the scale to samples
dominating_rate = torch.ones((num_samples,1), dtype=torch.float32).to(state_values.device)*dominating_rate
last_time = torch.ones_like(dominating_rate) * left_window if isinstance(left_window, (int, float)) else left_window
#new_time = last_time + dist.sample(sample_shape=torch.Size((1,1))).to(state_values.device)
#new_times = last_time + dist.sample(sample_shape=torch.Size((num_samples, proposal_batch_size))).cumsum(dim=-1)/dominating_rate
new_times = last_time + torch.ones((num_samples, proposal_batch_size)).to(args.device).cumsum(dim=-1)/dominating_rate

calculate_mark_mask = isinstance(mark_mask, float) and (("temporal_mark_restrictions" in mask_dict) or ("positional_mark_restrictions" in mask_dict))
resulting_times, resulting_marks, resulting_states = [], [], []

if adapt_dom_rate:
    dynamic_dom_rates = torch.ones((num_samples, dyn_dom_buffer,)).to(state_values.device)*dominating_rate
    k = 0
j = -1
while (new_times <= T).any() and (sample_lens < length_limit).any():
    print()
    j += 1
    within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1)
    to_stay = within_range_mask.any(dim=-1)
    to_go = ~to_stay
    if to_go.any():
        leaving_times, leaving_marks, leaving_states = timestamps[to_go, ...], marks[to_go, ...], state_values[to_go, ...]
        resulting_times.append(leaving_times)
        resulting_marks.append(leaving_marks)
        resulting_states.append(leaving_states)
        # for i,l in enumerate(sample_lens[to_go]):
        #     resulting_times.append(leaving_times[i, :l])
        #     resulting_marks.append(leaving_marks[i, :l])
        new_times = new_times[to_stay, ...]
        sample_lens = sample_lens[to_stay]
        timestamps = timestamps[to_stay, ...]
        marks = marks[to_stay, ...]
        state_values = state_values[to_stay, ...]
        state_times = state_times[to_stay, ...]
        batch_idx = batch_idx[:timestamps.shape[0]]
        last_time = last_time[to_stay, ...]
        dominating_rate = dominating_rate[to_stay, ...]
        if adapt_dom_rate:
            dynamic_dom_rates = dynamic_dom_rates[to_stay, ...]

        # print(j, "ADDING TIMES", leaving_times.shape, leaving_times.squeeze().cpu().tolist())
        # print(j, "ADDING MARKS", leaving_marks.shape, leaving_marks.squeeze().cpu().tolist())
        #print(j, "ADDING STATES", leaving_states.shape, leaving_states.squeeze().cpu().tolist())

    if calculate_mark_mask:
        mark_mask = self.determine_mark_mask(new_times, sample_lens, mask_dict)
        # print(j, "NEW TIMES", new_times.shape, new_times.squeeze().cpu().tolist())
        # print(j, "SAMPLE LENS", sample_lens.shape, sample_lens.squeeze().cpu().tolist())
        # print(j, "MARK MASK", mark_mask.shape, mark_mask.squeeze().cpu().tolist())

    within_range_mask = (new_times <= T) & (sample_lens < length_limit).unsqueeze(-1)

    # print(dist.rate, new_times.min(), sample_lens.min(), new_times.shape[0])

    # print(j, "INPUT TIMES", state_times.shape, state_times.squeeze().cpu().tolist())
    # print(j, "INPUT TIMESv2", timestamps.shape, timestamps.squeeze().cpu().tolist())
    # print(j, "INPUT MARKS", marks.shape, marks.squeeze().cpu().tolist())
    # print(j, "INPUT STATES", state_values.shape, state_values.squeeze().cpu().tolist())
    sample_intensities = self.get_intensity(
        state_values=None, #state_values,
        state_marks=marks,
        state_times=state_times,
        timestamps=new_times,
        marks=None,
        mark_mask=mark_mask,
    )
    # print(j, "TOTAL INTENSITY", sample_intensities["total_intensity"].shape, sample_intensities["total_intensity"].squeeze().cpu().tolist())
    redo_sample = False
    if adapt_dom_rate:  # Need to check and make sure that we don't break the sampling assumption
        redo_sample = (sample_intensities["total_intensity"] > dominating_rate).any()
        if not redo_sample:
            #finer_new_times = last_time + dist.sample(sample_shape=torch.Size((last_time.shape[0], finer_proposal_batch_size))).cumsum(dim=-1)/(dominating_rate*proposal_batch_size/4)
            finer_new_times = last_time + torch.ones((last_time.shape[0], finer_proposal_batch_size)).to(args.device).cumsum(dim=-1)/(dominating_rate*proposal_batch_size/4)
            finer_mark_mask = self.determine_mark_mask(finer_new_times, sample_lens, mask_dict) if calculate_mark_mask else 1.0
            finer_sample_intensities = self.get_intensity(  # Finer resolution check just after `last_time`
                state_values=state_values,
                state_times=state_times,
                timestamps=finer_new_times,
                marks=None,
                mark_mask=finer_mark_mask,
            )
            redo_sample = (finer_sample_intensities["total_intensity"] > dominating_rate).any()

    if not redo_sample:
        # print(j, new_times.min(), T, sample_lens.min(), length_limit, sample_lens.shape, dist.rate, sample_intensities["total_intensity"].mean())
        #acceptances = torch.rand_like(new_times) <= (sample_intensities["total_intensity"] / dominating_rate)  #dominating_rate)
        prob = torch.rand(new_times.shape).to(args.device)
        thresh = (sample_intensities["total_intensity"] / dominating_rate)
        acceptances = prob <= thresh  #dominating_rate)
        # print(j, "PROB", prob.shape, prob.squeeze().cpu().tolist())
        # print(j, "THRESH", thresh.shape, thresh.squeeze().cpu().tolist())
        # print(j, "REJ ACCEPT", acceptances.shape, acceptances.squeeze().cpu().tolist())
        acceptances = acceptances & within_range_mask  # Don't accept any sampled events outside the window
        samples_w_new_events = acceptances.any(dim=-1)
        if samples_w_new_events.any():
            event_idx = acceptances.int().argmax(dim=-1)
            new_time = new_times[batch_idx, event_idx].unsqueeze(-1)

            logits = sample_intensities["all_log_mark_intensities"][batch_idx, event_idx, :].unsqueeze(-2)
            mark_probs = F.softmax(logits, -1)
            mark_dist = torch.distributions.Categorical(mark_probs.cpu())
            new_mark = mark_dist.sample().to(args.device)

            # print(j, "POS W NEW EVENTS", samples_w_new_events.squeeze().shape, samples_w_new_events.squeeze().cpu().tolist())
            # print(j, "NEW EVENTS", new_time.squeeze(-1)[samples_w_new_events].shape, new_time.squeeze(-1)[samples_w_new_events].cpu().tolist())

            # Need to store sampled events into timestamps and marks
            # Some need to be appended, some need to overwrite previously written padded values
            to_append = (samples_w_new_events & (sample_lens == timestamps.shape[-1])).unsqueeze(-1)
            to_pad = ~to_append
            if to_append.any():
                timestamps = torch.cat((timestamps, torch.where(to_append, new_time, time_pad)), -1)  #new_time*to_append + time_pad*to_pad), -1)
                marks = torch.cat((marks, torch.where(to_append, new_mark, mark_pad)), -1)  #new_mark*to_append + mark_pad*to_pad), -1)
                # print(j, "APPENDED", new_time.squeeze(-1)[to_append.squeeze(-1)].shape, new_time.squeeze(-1)[to_append.squeeze(-1)].cpu().tolist())

            to_overwrite = samples_w_new_events & (sample_lens < timestamps.shape[-1])
            if to_overwrite.any():
                timestamps[to_overwrite, sample_lens[to_overwrite]] = new_time.squeeze(-1)[to_overwrite]
                marks[to_overwrite, sample_lens[to_overwrite]] = new_mark.squeeze(-1)[to_overwrite]
                # print(j, "OVER_WRITTEN", new_time.squeeze(-1)[to_overwrite].shape, new_time.squeeze(-1)[to_overwrite].cpu().tolist())

    
            sample_lens[samples_w_new_events] += 1  # Guaranteed at least one event was either appended or overwritten
            state = self.forward(marks, timestamps)
            state_values, state_times = state["state_dict"]["state_values"], state["state_dict"]["state_times"]
            last_time = torch.where(samples_w_new_events.unsqueeze(-1), new_time, torch.max(new_times, dim=-1, keepdim=True).values)  #new_time*samples_w_new_events.unsqueeze(-1) + (torch.max(new_times, dim=-1).values*(~samples_w_new_events)).unsqueeze(-1)
        else:
            last_time = torch.max(new_times, dim=-1, keepdim=True).values

    # print(j, "LAST TIME", last_time.shape, last_time.squeeze().cpu().tolist())
    if j > 5:
        break
    # else: # Redo sample
    #     print("REDOING SAMPLE")

    if adapt_dom_rate:  
        dynamic_dom_rates[:, k] = sample_intensities["total_intensity"].max(dim=1).values*100
        k = (k+1) % dynamic_dom_rates.shape[1]
        dominating_rate = torch.max(dynamic_dom_rates, dim=1, keepdim=True).values
    # print(j, "DOMINATING RATE", dominating_rate.shape, dominating_rate.squeeze().cpu().tolist())

    #new_times = last_time + dist.sample(sample_shape=(new_times.shape[0], proposal_batch_size)).cumsum(dim=-1)/dominating_rate
    new_times = last_time + torch.ones((new_times.shape[0], proposal_batch_size)).to(args.device).cumsum(dim=-1)/dominating_rate

print()
print()
#for i,l in enumerate(sample_lens):
resulting_times.append(timestamps)  #timestamps[i, :l])
resulting_marks.append(marks)       #marks[i, :l])
resulting_states.append(state_values)

# print(j+1, "FINAL ADDING TIMES", timestamps.shape, timestamps.squeeze().cpu().tolist())
# print(j+1, "FINAL ADDING MARKS", marks.shape, marks.squeeze().cpu().tolist())
#print(j+1, "FINAL ADDING STATES", state_values.shape, state_values.squeeze().cpu().tolist())

print(resulting_times)
print(resulting_marks)

assumption_violation = False
if not adapt_dom_rate:
    for _ in range(5):
        eval_times = torch.rand_like(timestamps).clamp(min=1e-8)*T
        sample_intensities = self.get_intensity(
            state_values=state_values,
            state_times=state_times,
            timestamps=eval_times,
            marks=None,
        )
        if (sample_intensities["total_intensity"] > dominating_rate).any().item():
            print("DR: {}".format(dominating_rate))
            print("IN: {}".format(sample_intensities["total_intensity"].max().item()))
            assumption_violation = True
            break

if assumption_violation:
    print("Violation in sampling assumption occurred. Redoing sample.")


    #     torch.cat((timestamps, torch.FloatTensor([sampled_times])), dim=-1), 
    #     torch.cat((marks, torch.LongTensor([sampled_marks])), dim=-1),
    # )
