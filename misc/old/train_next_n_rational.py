'''
Trains a language model with a rational on the daily dialog set.
'''

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from misc.old.NextNPredictionDataset import NextNPredictionDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer

from utils.callbacks import FinishDialogueRationalizedCallback
from modules.LanguageModels.LstmLanguageModel import LSTMLanguageModel
from modules.pytorch_lightning.LightingBaseRationalizedLanguageModel import LightingBaseRationalizedLanguageModel
from modules.RationalExtractors import RationalExtractor

save_path = '../../models/small_lm_pretrained.pt'
load_pretrained = True

teacher_forcing = True
size = int(5e2)
test_size = int(1e2)

max_epochs = 50
batch_size = 32
embedding_dim = 128
learning_rate = 1e-3

hparams = {
    "learning_rate": learning_rate,
    "teacher_forcing": teacher_forcing,
    "freeze_language_ml": True
}

my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )

train_dataset = NextNPredictionDataset(my_tokenizer, size=size)
dataloader_train = DataLoader(train_dataset, shuffle=True)

test_dataset = NextNPredictionDataset(my_tokenizer, size=test_size, split="test")
dataloader_test = DataLoader(test_dataset, )

device = "cuda"

if load_pretrained:
    print("load pretrained_model")

    language_model = LSTMLanguageModel.load(save_path).to(device)
else:
    print("load fresh model")
    language_model = LSTMLanguageModel(my_tokenizer.get_vocab_size(), embedding_dim=embedding_dim).to(device)

callbacks = [
    FinishDialogueRationalizedCallback(["[START] How ", "[START] What are you upto? "]),
]

loss_module = torch.nn.CrossEntropyLoss()

rational_extractor = RationalExtractor(embedding_dim)

model = LightingBaseRationalizedLanguageModel(language_model, rational_extractor, my_tokenizer, loss_module, hparams=hparams)

trainer = pl.Trainer(default_root_dir='../../logs',
                     checkpoint_callback=False,
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     callbacks=callbacks,
                     progress_bar_refresh_rate=1,
                     gradient_clip_val=2.0
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(model, dataloader_train, dataloader_test)
