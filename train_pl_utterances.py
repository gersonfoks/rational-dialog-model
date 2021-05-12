'''
Trains a language model on the daily dialog set.
'''
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from daily_dialog.UtterancesDataset import UtterancesDataset
from daily_dialog.DialogTokenizer import get_daily_dialog_tokenizer
from modules.LanguageModels.LstmLanguageModel import LSTMLM
from modules.LanguageModels.LanguageModelPL import LMPL
from daily_dialog.callbacks import FinishSentenceCallback, ReshuffleDatasetCallback

from utils import get_lastest_model_name, generate_model_name


save_path = r'./saved_models/'
load_pretrained = True
batch_first = True
max_epochs = 10
batch_size = 256
embedding_dim = 128
num_layers = 2
hidden_state_size = 128
learning_rate = 1e-3

print("get tokenizer")
my_tokenizer = get_daily_dialog_tokenizer(tokenizer_location='./daily_dialog/tokenizer.json', )
#my_tokenizer = get_daily_dialog_tokenizer()

dataset_train = UtterancesDataset(my_tokenizer, subsets="full", split="train")
dataset_test = UtterancesDataset(my_tokenizer, subsets="start", split="test")

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=UtterancesDataset.collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=UtterancesDataset.collate_fn)
device = "cuda" if torch.cuda.is_available() else "cpu"

if load_pretrained:
    model_name = get_lastest_model_name(save_path)
    print("load pretrained_model: ", model_name)
    language_model = LSTMLM.load(model_name).to(device)
else:
    print("load fresh model")
    language_model = LSTMLM(
        my_tokenizer.get_vocab_size(), 
        embedding_dim=embedding_dim, 
        num_layers=num_layers,
        hidden_state_size=hidden_state_size
    ).to(device)

callbacks = [
    FinishSentenceCallback(['How are you doing today?', 'What are you upto?', "Hi, long time no see!"]),
    ReshuffleDatasetCallback(dataset_train) # To reshuffle the dataset.
]

loss_module = torch.nn.CrossEntropyLoss(ignore_index=0)
hparams = {"learning_rate": learning_rate}
model = LMPL(language_model, my_tokenizer, loss_module, hparams=hparams)

print("training")
trainer = pl.Trainer(
    default_root_dir='logs',
    checkpoint_callback=False,
    gpus=1 if torch.cuda.is_available() else 0,
    max_epochs=max_epochs,
    log_every_n_steps=1,
    progress_bar_refresh_rate=0 if torch.cuda.is_available() else 1,
    callbacks=[] if torch.cuda.is_available() else callbacks,
    gradient_clip_val=2.0
)
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need


print("BEFORE")
print(torch.cuda.is_available(), device)
print(model.complete_sentence("Hi, what can I do for you?", 100))

trainer.fit(model, dataloader_train, dataloader_test)

name = generate_model_name(save_path)
print("save language model: ", name)
language_model.save(name)

print("AFTER")
print(model.complete_sentence("Hi, what can I do for you?", 100))
