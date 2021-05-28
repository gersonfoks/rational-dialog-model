''''
Important!
This analysis is based on the policy type rational models!
Does probably not work with the shared embedding type.
'''
from collections import Counter

import torch
import yaml
from tqdm import tqdm
import numpy as np

from modules.pytorch_lightning.LightingBaseRationalizedLanguageModel import LightingBaseRationalizedLanguageModel
from modules.pytorch_lightning.LightningLanguageModel import LightningLanguageModel
from modules.pytorch_lightning.LightningReinforceRationalizedLanguageModel import \
    LightingReinforceRationalizedLanguageModel
from utils.parse_config import get_tokenizer, get_datasets, get_language_model, get_rational_extractor, get_trainer
from utils.token_utils import get_token_id
from utils.utils import calc_perplexity

import matplotlib.pyplot as plt


def parse_config_for_analysis(config_ref):
    with open(config_ref, 'r') as f:
        config = yaml.load(f)

    result = {"config": config}

    # Make sure we use the pretrained
    config["language_model"]["pretrained"] = True
    config["rational_extractor"]["pretrained"] = True

    # First we load the tokenizer and the dataset
    tokenizer = get_tokenizer(config["tokenizer"])
    result["tokenizer"] = tokenizer
    datasets = get_datasets(config["dataset"], tokenizer, load_train=False)
    result = {**result, **datasets}

    # Get language model

    language_model_no_RE = get_language_model(config["language_model"], tokenizer)

    language_model_RE_config = config["language_model"].copy()

    result["language_model_no_RE"] = language_model_no_RE

    language_model_RE_config["load_location"] = language_model_RE_config["save_location"]
    language_model_RE = get_language_model(language_model_RE_config, tokenizer)
    embedding_size = language_model_RE.embedding_size
    result["language_model_RE"] = language_model_RE
    RE = get_rational_extractor(config["rational_extractor"], tokenizer, embedding_size=embedding_size)
    result["rational_extractor"] = RE

    # get loss module and hyper parameters for training
    hparams = config["hparams"]
    result["hparams"] = hparams

    if "policy" in config['rational_extractor']['type']:
        lightning_language_model_RE = LightingReinforceRationalizedLanguageModel(language_model_RE, RE, tokenizer,
                                                                              hparams=hparams,
                                                                              **config["rational_extractor"][
                                                                                  "parameters"])
    else:
        lightning_language_model_RE = LightingBaseRationalizedLanguageModel(language_model_RE, RE, tokenizer,
                                                                          hparams=hparams,
                                                                         **config["rational_extractor"]["parameters"])

    lightning_language_model = LightningLanguageModel(language_model_no_RE, tokenizer,
                                                      hparams=hparams)

    result["lightning_language_model_RE"] = lightning_language_model_RE
    result["lightning_language_model_no_RE"] = lightning_language_model

    trainer = get_trainer(result)
    result["trainer"] = trainer

    return result


def get_results(model, dataloader):
    mean_acc = 0
    mean_perplexity = 0
    mean_mask_percentage = 0
    total_samples = len(dataloader.dataset)
    for batch in tqdm(dataloader):
        contexts = batch[0].to(model.device)
        targets =  batch[1].to(model.device)
        results = model.batch_to_out((contexts, targets))
        samples_in_batch = batch[0].shape[0]
        mean_acc += results["acc"].item() * (samples_in_batch) / total_samples

        if "mask_mean" in results.keys():
            mean_mask_percentage += results["mask_mean"].item() * (samples_in_batch) / total_samples

        mean_perplexity += results["perplexity"].item() * (samples_in_batch) / total_samples
    return {"mean_acc": mean_acc, "mean_perplexity": mean_perplexity, "mean_mask_percentage": mean_mask_percentage}


def get_results_RE(model, dataloader, n_experiments):
    '''
    As RE is policy based we need to calculate average and std.

    '''
    total_results = []
    results = {}
    for i in range(n_experiments):
        total_results.append(get_results(model, dataloader))
    for key in total_results[0].keys():
        all_values = [r[key] for r in total_results]
        results[key] = {"mean": np.mean(all_values), "std": np.std(all_values)}
    return results


def calc_change_in_perplexity_experiment(model, dataloader, n_experiments=2, n_extra_mask=1):
    total_results = []
    results = {}
    for i in range(n_experiments):
        total_results.append(calc_change_in_perplexity(model, dataloader, n_extra_mask=n_extra_mask))
    for key in total_results[0].keys():
        all_values = [r[key] for r in total_results]
        results[key] = {"mean": np.mean(all_values), "std": np.std(all_values)}
    return results


def calc_change_in_perplexity(model, dataloader, n_extra_mask=1):
    total_diff_perplexity = 0
    tokenizer = model.tokenizer

    for (contexts, targets) in dataloader:
        # Make batch second
        contexts = contexts.permute(1, 0, ).to(model.device)
        targets = targets.permute(1, 0, ).to(model.device)
        n_targets = targets.shape[0]
        # Get the rational
        rational = model.get_rational(contexts)

        # Mask n extra tokens (in each rational)
        masked_input = rational["masked_input"]

        logits = model.forward_masked_input(masked_input, targets)[-(n_targets + 1):-1]

        mask_token_id = get_token_id(tokenizer, "mask_token")
        extra_masked = mask_extra(masked_input, n_extra_mask, mask_token_id)
        logits_altered = model.forward_masked_input(extra_masked, targets)[-(n_targets + 1):-1]

        # Get the perplexity
        perplexity_non_altered = calc_perplexity(logits, targets, tokenizer)
        perplexity_altered = calc_perplexity(logits_altered, targets, tokenizer)

        diff = (perplexity_non_altered - perplexity_altered).mean()
        total_diff_perplexity += diff

    mean_diff = total_diff_perplexity / len(dataloader)

    return {"mean_diff_perplexity": mean_diff}


def mask_extra(masked_input, n_extra_mask, mask_token_id):
    # First find all the locations in which there is no mask
    result = masked_input

    n_batches = masked_input.shape[1]
    non_mask_indices = (masked_input != mask_token_id).nonzero().cpu().numpy()

    # Next we extract for each batch extra tokens to mask
    for batch_number in range(n_batches):

        # Filter the possible locations
        # Get sequence index of the current batch where we do not have a mask
        mask = non_mask_indices[:, 1] == batch_number  # THe mask for the current batch
        possible_locations = non_mask_indices[mask][:, 0]  # The actual indices for this current batch

        # Pick a random_ones (make sure we do not pick more than there are available)
        n_mask = min(len(possible_locations), n_extra_mask)
        locations = np.random.choice(possible_locations, n_mask)

        # Mask those locations
        for location in locations:
            result[location, batch_number] = mask_token_id

    return result


def rational_analysis(model, dataloader, greedy=False):
    n = 1
    abs_averages = 0.0
    rel_averages = 0.0
    ## We also want to keep track of the distribution
    abs_pos_count = Counter()
    rel_pos_count = Counter()

    for (contexts, _) in dataloader:
        # This assumes a batch consists of a list of (context, response) pairs
        contexts = contexts.permute(1,0).to(model.device)

        #                print("Context: ", context)

        # Get the mask
        if greedy:
            rational = model.get_rational(contexts, greedy=greedy)
        else:
            rational = model.get_rational(contexts)
        mask = rational["mask"]

        #                print("Mask: ", mask, mask.size(), len(mask[0]))
        pad_id = get_token_id(model.tokenizer, "pad_token")
        abs_pos_count_batch, rel_pos_count_batch = get_abs_and_relative_positions(mask, contexts, pad_id=pad_id)
        abs_pos_count += abs_pos_count_batch
        rel_pos_count += rel_pos_count_batch


        # print("Average absolute: ", average_absolute)
        # print("Average relative: ", average_relative)
    abs_average = abs_averages / n
    rel_average = rel_averages / n
    return {"abs_average": abs_average, "rel_average": rel_average, "abs_pos_count": abs_pos_count , "rel_pos_count": rel_pos_count}


def get_abs_and_relative_positions(mask, tokens, pad_id, batch_first=False):

    #Make batch first:
    if not batch_first:
        mask = mask.permute(1,0)
        tokens = tokens.permute(1,0)

    non_paddings = (tokens != pad_id).bool()
    lengths = non_paddings.sum(dim=-1)

    length_copy = list(lengths.detach().cpu().numpy())
    mask = non_paddings * mask



    reverse_positions =  torch.tensor(list(range(mask.shape[1], 0, -1))).to(mask.device)




    mask_positions = torch.mul(mask, reverse_positions).float()
    #                print("Positions: ", mask_positions)


    lengths = lengths.reshape(-1, 1).repeat(1, mask_positions.shape[1])
    abs_pos = mask_positions.detach().cpu().numpy()

    rel_positions = (10 * torch.round(10 * (1/lengths) * mask_positions )).detach().cpu().numpy()

    mask = mask.detach().cpu().numpy()
    abs_pos = flatten([list([ p for p, keep in zip(pos[-l:], m[-l: ]) if keep ])for pos,m,  l in zip(abs_pos, mask,  length_copy)])
    rel_positions = flatten([list([ p for p, keep in zip(pos[-l:], m[-l: ]) if keep ])for pos,m,  l in zip(rel_positions, mask,  length_copy)])

    abs_pos_count = Counter(list(abs_pos))
    rel_pos_count = Counter(rel_positions)

    return abs_pos_count, rel_pos_count,

def flatten(l):
    result = []
    for r in l:
        result += r
    return result

def pretty_print_completed_dialogues(completed_dialogues):
    print("context ----> response")
    for dialogue in completed_dialogues:

        for rational, out in zip(dialogue["rationalized_input"], dialogue["response"]):
            print(rational, '------>', out)


import json
import pandas as pd
from os import path
def add_distribution_to_file(distribution, name, file):

    if not path.exists(file):
        with open(file, "w") as f:
            pass
    with open(file, "r") as f:
        if len(f.read()) == 0:
            print("empty")
            df = pd.DataFrame()
            df["percentages"] = [i* 10 for i in range( len(distribution.values()))]
            df[name] = list(distribution.values())
        else:
            df = pd.read_csv(file,  index_col=0)

            df[name] = list(distribution.values())
            print(df)
    df.to_csv(file)
    return df


def prepare_rel_pos_count(rel_pos_count):
    total = sum(rel_pos_count.values())
    info = {key: value/total for key, value in sorted(rel_pos_count.items(), key=lambda x: x[0])}
    return info


def plot_rel_pos(file):
    df = pd.read_csv(file, index_col=0)
    df.plot(x="percentages", xlabel="Relative Distance", ylabel="percentage")
    plt.show()