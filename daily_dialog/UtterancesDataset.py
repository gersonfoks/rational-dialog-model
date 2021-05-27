from torch.utils.data import Dataset
import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools
import random
from tokenizers import Tokenizer
from utils.token_utils import special_tokens, get_token, get_token_id


class UtterancesDataset(Dataset):

    def __init__(self, tokenizer, size=None, subsets="start", perturbation=None, split="train", remove_top_n=-1,
                 shuffle=True, max_length=0):
        '''
        remove_top_n: is used to remove the longest samples from the dataset (to prevent memory problems)
        size: is used to restrict the number of dialogues that is used
        '''
        assert subsets in ["start", "end", "single", "full"], "subsets should be 'start', 'end', 'single, or 'full'"
        assert perturbation in [None, "utterance_dialogue", "words_dialogue", "words_utterance"]
        self.shuffle = shuffle
        self.max_length = max_length
        print(self.max_length)
        self.original_dataset = datasets.load_dataset("daily_dialog", split=split, )
        self.dialogues = [item["dialog"] for item in self.original_dataset]
        self.size = len(self.dialogues)
        if (size != None) and (size > 0) and (size < self.size):
            self.dialogues = random.sample(self.dialogues, size)
            self.size = size
        self.tokenizer = tokenizer
        self.subsets = subsets
        self.remove_top_n = remove_top_n
        self.dataset = self.process_dataset(subsets, perturbation)


    def process_dataset(self, subsets, perturbation):
        '''
        Returns tokenized subsets of the dataset.
        Output is a list with (context, response) pairs, sorted on combined length
        '''

        # Split all the dialogues in subdialogues --> ([utterances in context], reponse)
        dialogue_samples = list(itertools.chain.from_iterable([
            self.subdialogues(dialogue, subsets)
            for dialogue in self.dialogues
        ]))

        # Perturn the context, based on specified perturbation option
        perturbed_samples = self.perturb_dataset(dialogue_samples, perturbation)

        # Tokenize the samples
        if type(self.tokenizer) == Tokenizer:

            # For Tokenizer class, you need to extract the id's
            sep_token = get_token(self.tokenizer, "sep_token")
            tokenized_samples = [
                (self.tokenizer.encode((sep_token).join(context) + sep_token).ids,
                 self.tokenizer.encode(response + sep_token).ids)
                for (context, response) in perturbed_samples
            ]
        else:

            # For the other models, the encode function already gives the id's
            self.tokenizer.add_special_tokens(
                {'sep_token': "[SEP]", 'pad_token': "[PAD]", 'mask_token': "[MASK]", 'eos_token': "[EOS]"})
            sep_token = self.tokenizer.sep_token
            tokenized_samples = [
                (self.tokenizer.encode((sep_token).join(context) + sep_token),
                 self.tokenizer.encode(response + sep_token))
                for (context, response) in perturbed_samples
            ]

        # Now shuffle samples and sort on length (to prevent amount of padding in batches)
        if self.max_length > 0:
            tokenized_samples = [(x,y) for (x,y) in tokenized_samples if len(x)  + len(y) < self.max_length]
        if self.shuffle:
            random.shuffle(tokenized_samples)
        sorted_samples = sorted(tokenized_samples, key=lambda x: len(x[0]) * 10 + len(x[1]) + self.shuffle * random.randint(0, 10))

        if self.remove_top_n > 0:
            sorted_samples = sorted_samples[:int(-1 * self.remove_top_n)]


        return sorted_samples

    def perturb_dataset(self, samples, perturbation):
        # print("DEBUG: Perturbation = ", perturbation)

        if perturbation == None:

            # No perturbation, so just return the dialogues in normal order
            result = samples

        elif perturbation == "utterance_dialogue":

            # Shuffle order of utterances within dialogue
            result = [(random.sample(context, len(context)), response) for (context, response) in samples]

        elif perturbation == "words_utterance":

            # Shuffle order of words within utterance, but keep order of utterances
            result = [
                ([self.shuffled_utterance(u) for u in context], response)
                for (context, response) in samples
            ]

        elif perturbation == "words_dialogue":

            # Shuffle order of words accross the whole dialogue
            result = [
                ([self.shuffled_utterance(" ".join(context))], response)
                for (context, response) in samples
            ]

        else:
            raise AssertionError("Unkown perturbation: " + perturbation)

        # print("DEBUG: result of perturbations")
        # print(result)
        return result

    def subdialogues(self, utterances, subsets):
        '''
        Create subsets of the dialogue.
        Input is a list of utterances.
        Output is a list with sub-dialogues with (context, reponse) pairs.
        If subsets == 'full', then return ALL subsets of utterances
        If subsets == 'start', then all subsets start from the first utterances
        If subsets == 'end', the all subsets END at utterance length-1
        If subsets == 'single', context is just a single utterance
        '''
        # print("DEBUG subdialogues ", subsets, '\n', utterances)
        l = len(utterances)
        if l < 2:
            return None
        if subsets == "single":
            results = [
                ([utterances[start]], utterances[start + 1])
                for start in range(0, l - 1)
            ]
        elif subsets == "start":
            results = [
                (utterances[0:end], utterances[end])
                for end in range(1, l)
            ]
        elif subsets == "end":
            results = [
                (utterances[start:l - 1], utterances[l - 1])
                for start in range(0, l - 1)
            ]
        elif subsets == "full":
            results = [
                (utterances[start:end], utterances[end])
                for start in range(0, l - 1)
                for end in range(start + 1, l)
            ]
        else:
            results = None

        # for i, s in enumerate(results):
        #     print(i, s)
        return results

    def shuffled_utterance(self, utterance):
        words = utterance.split()
        random.shuffle(words)
        return (' '.join(words))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def reshuffle_dataset(self):
        '''
        Reshuffles the dataset
        '''
        self.dataset = sorted(self.dataset, key=lambda x: len(x[0]) * 10 + len(x[1]) + random.randint(0, 10))

    @staticmethod
    def get_collate_fn(
            padding_value=None):  # make sure you provide the right id from the tokenizer, when get_collate_fn is called
        def collate_fn(items):
            '''
            Pads the context from the left and the response from the right.
            Results in batches with (contexts, responses), with batch-first.
            '''
            assert padding_value != None, "Please provide pad_token_id to use for padding"
            inputs = torch.fliplr(pad_sequence(
                [torch.tensor(list(reversed(sample[0]))) for sample in items],
                batch_first=True,
                padding_value=padding_value  # Padding code
            ))
            targets = pad_sequence(
                [torch.tensor(sample[1]) for sample in items],
                batch_first=True,
                padding_value=padding_value  # Padding code
            )

            return inputs, targets

        return collate_fn
