import torch
import yaml
from torch.utils.data import DataLoader

from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from daily_dialog.UtterancesDataset import UtterancesDataset
from modules.LanguageModels.LstmLanguageModel import LSTMLanguageModel
from modules.LanguageModels.PretrainedLanguageModel import PretrainedLanguageModel
from modules.RationalExtractors.KumaRationalExtractor import KumaRationalExtractor
from modules.RationalExtractors.PolicyBasedRationalExtractor import PolicyBasedRationalExtractor
from modules.RationalExtractors.PolicyBasedUtteranceRationalExtractor import PolicyBasedUtteranceRationalExtractor
from modules.RationalExtractors.RandomRationalExtractor import RandomRationalExtractor
from modules.pytorch_lightning.LightningLanguageModel import LightningLanguageModel
import pytorch_lightning as pl
from transformers import AutoTokenizer

from modules.pytorch_lightning.LightningReinforceRationalizedLanguageModel import \
    LightingReinforceRationalizedLanguageModel

from modules.pytorch_lightning.LightingBaseRationalizedLanguageModel import LightingBaseRationalizedLanguageModel

from modules.RationalExtractors.EmbeddingRationalExtractor import RationalExtractor
from utils.callbacks import FinishDialogueCallback, FinishDialogueRationalizedCallback, ReshuffleDatasetCallback
from tokenizers import Tokenizer
from utils.token_utils import get_token_id, get_vocab_size


def parse_config(config_ref):
    with open(config_ref, 'r') as f:
        config = yaml.load(f)

    result = {"config": config}

    # First we load the tokenizer and the dataset
    tokenizer = get_tokenizer(config["tokenizer"])
    result["tokenizer"] = tokenizer
    datasets = get_datasets(config["dataset"], tokenizer)
    result = {**result, **datasets}

    # Get language model and rationale extractor (if applicable)
    language_model = get_language_model(config["language_model"], tokenizer)
    result["language_model"] = language_model
    if "rational_extractor" in config.keys():
        if config['language_model']['type'] == "transformers":
            embedding_size = language_model.embedding_size
        else:
            embedding_size = language_model.embedding_dim
        RE = get_rational_extractor(config["rational_extractor"], tokenizer, embedding_size)
        result["rational_extractor"] = RE

    # get loss module and hyper parameters for training
    hparams = config["hparams"]
    result["hparams"] = hparams

    # Load the pytorch lightning module and the trainer
    if "rational_extractor" in config.keys():
        if "policy" in config['rational_extractor']['type']:
            lightning_language_model = LightingReinforceRationalizedLanguageModel(language_model, RE, tokenizer,
                                                                                  hparams=hparams,
                                                                                  **config["rational_extractor"][
                                                                                      "parameters"])
        else:
            lightning_language_model = LightingBaseRationalizedLanguageModel(language_model, RE, tokenizer,
                                                                             hparams=hparams,
                                                                             **config["rational_extractor"][
                                                                                 "parameters"])
    else:
        lightning_language_model = LightningLanguageModel(language_model, tokenizer,
                                                          hparams=hparams)

    result["lightning_language_model"] = lightning_language_model

    trainer = get_trainer(result)
    result["trainer"] = trainer

    return result


def get_tokenizer(tokenizer_config):
    if tokenizer_config["type"] == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["checkpoint"])
    elif tokenizer_config["type"] == "daily_dialogue":
        print(tokenizer_config["link"])
        tokenizer = get_daily_dialog_tokenizer(tokenizer_location=tokenizer_config["link"], )
    else:
        raise ValueError("type not found", tokenizer_config["type"])
    return tokenizer


def get_datasets(config, tokenizer, load_train=True):
    if config["type"] == 'daily_dialogue':
        if load_train:
            dataset_train = UtterancesDataset(
                tokenizer,
                subsets="start",
                split="train",
                size=config["size_train"],
                remove_top_n=config["remove_top_n"],
                max_length=config["max_length"] if "max_length" in config.keys() else 0,
            )

            dataloader_train = DataLoader(
                dataset_train,
                batch_size=config["batch_size"],
                collate_fn=UtterancesDataset.get_collate_fn(padding_value=get_token_id(tokenizer, "pad_token"))
            )
        else:
            dataloader_train = None
        dataset_test = UtterancesDataset(
            tokenizer,
            subsets="start",
            split="test",
            size=config["size_test"],
            remove_top_n=config["remove_top_n"],
            shuffle=False,
            max_length=config["max_length"] if "max_length" in config.keys() else 0,
        )  # We do not have to shuffle the test dataset.

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=config["batch_size"],
            collate_fn=UtterancesDataset.get_collate_fn(padding_value=get_token_id(tokenizer, "pad_token"))
        )
        return {"dataloader_train": dataloader_train, "dataloader_test": dataloader_test}
    else:
        raise ValueError("type not found", config["type"])


def get_language_model(config, tokenizer):
    if config["type"] == "LSTM":
        if config["pretrained"]:
            if "load_location" in config:
                model_location = config["load_location"]
            else:
                model_location = config["save_location"]
            print("load pretrained_model: ", model_location)
            language_model = LSTMLanguageModel.load(model_location)
        else:
            print("load fresh model")
            language_model = LSTMLanguageModel(
                tokenizer.get_vocab_size(),
                embedding_dim=config['embedding_dim'],
                num_layers=config['num_layers'],
                hidden_state_size=config['hidden_state_size']
            )
    elif config["type"] == "transformers":
        if config["pretrained"]:
            if "load_location" in config:
                model_location = config["load_location"]
            else:
                model_location = config["save_location"]
            print("load pretrained_model: ", model_location)
            language_model = PretrainedLanguageModel(
                pretrained_model=model_location,
                tokenizer=tokenizer
            )
        else:
            language_model = PretrainedLanguageModel(
                pretrained_model=config['checkpoint'],
                tokenizer=tokenizer
            )
    else:
        raise ValueError("type not found", config["type"])
    return language_model


def get_loss_module(config, tokenizer):
    pad_id = get_token_id(tokenizer, "pad_token")

    if type(tokenizer) == Tokenizer:
        weight = torch.ones(tokenizer.get_vocab_size())
    else:
        weight = torch.ones(len(tokenizer))
    weight[pad_id] = 0
    return torch.nn.CrossEntropyLoss(weight=weight)


def get_rational_extractor(config, tokenizer, embedding_size=32):
    if config["type"] == "policy_based":
        if config["pretrained"]:
            return PolicyBasedRationalExtractor.load(config["load_location"])
        else:
            return PolicyBasedRationalExtractor(get_vocab_size(tokenizer),
                                                mask_token=get_token_id(tokenizer, "mask_token"),
                                                )

    if config["type"] == "shared_embedding":
        if config["pretrained"]:
            return RationalExtractor.load(config["load_location"])
        else:
            return RationalExtractor(embedding_size)

    if config["type"] == "shared_embedding_kum":
        if config["pretrained"]:
            return KumaRationalExtractor.load(config["load_location"])
        else:
            return KumaRationalExtractor(embedding_size)

    if config["type"] == "policy_based_random":
        return RandomRationalExtractor(
            mask_token=get_token_id(tokenizer, "mask_token"),
            percentage=config["percentage"]
        )

    if config["type"] == "policy_utterance":
        return PolicyBasedUtteranceRationalExtractor(get_vocab_size(tokenizer),
                                                     mask_token=get_token_id(tokenizer, "mask_token"),
                                                     sep_token=get_token_id(tokenizer, "sep_token"),
                                                     )


def get_trainer(information):
    config = information["config"]["trainer"]

    if config["type"] == "normal":
        callbacks = [
            FinishDialogueCallback(["How are you doing today? [SEP]", "What are you upto? [SEP]"]),
            ReshuffleDatasetCallback(information["dataloader_test"].dataset),
        ]
        trainer = pl.Trainer(
            default_root_dir='logs',
            checkpoint_callback=False,
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks
        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        return trainer

    elif config["type"] == "policy":
        callbacks = [
            FinishDialogueRationalizedCallback(["How are you doing today?[SEP]", "What are you upto?[SEP]"]),
            FinishDialogueRationalizedCallback(["How are you doing today?[SEP]", "What are you upto?[SEP]"],
                                               greedy_policy=True),
            ReshuffleDatasetCallback(information["dataloader_test"].dataset),
            # ChangeInPerplexityCallback(information["dataloader_test"]) #TODO enable again
        ]
        trainer = pl.Trainer(
            default_root_dir='logs',
            checkpoint_callback=False,
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks

        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        return trainer

    elif config["type"] == "shared":
        callbacks = [
            ReshuffleDatasetCallback(information["dataloader_test"].dataset),
        ]
        trainer = pl.Trainer(
            default_root_dir='logs',
            checkpoint_callback=False,
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            progress_bar_refresh_rate=1,
            callbacks=callbacks

        )
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        return trainer
