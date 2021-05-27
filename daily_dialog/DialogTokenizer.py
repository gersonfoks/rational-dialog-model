'''
Code to create a tokenizer for the daily dialog.
'''
from tokenizers.pre_tokenizers import Whitespace
import datasets
from tokenizers import Tokenizer, normalizers

from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.trainers import WordPieceTrainer

from utils import token_utils
from utils.token_utils import special_tokens


def get_daily_dialog_tokenizer(tokenizer_location=None):
    '''
    Get the daily dialog tokenizer. Trains a new one if no location is provided
    :param tokenizer_location: Json containing information about the tokenizer.
    :return:
    '''
    if tokenizer_location:
        tokenizer = Tokenizer.from_file(tokenizer_location, )
        tokenizer.enable_padding()
        return tokenizer
    else:
        dataset_train = datasets.load_dataset("daily_dialog", split="train", )
        utterances = [special_tokens["sep_token"].join(dialogue["dialog"]) for dialogue in dataset_train]

        trainer = WordPieceTrainer(
            vocab_size = 2048, 
            special_tokens = token_utils.special_tokens.values()
        )

        custom_tokenizer = Tokenizer(WordPiece(unk_token=special_tokens["unk_token"], ))
        custom_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        custom_tokenizer.pre_tokenizer = Whitespace()
        custom_tokenizer.train_from_iterator(utterances, trainer, )
        custom_tokenizer.enable_padding()

        # Write every dialogue to file
        location = './daily_dialog/'
        custom_tokenizer.save(location + "tokenizer.json")

        return custom_tokenizer
