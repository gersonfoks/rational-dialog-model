import torch
import pytorch_lightning as pl
from tokenizers import Tokenizer
from transformers import AdamW
import torch.nn.functional as F
from utils.utils import calculate_mask_percentage, fussed_lasso, get_pad_id, calc_perplexity, calc_acc
from utils.token_utils import get_vocab_size, get_token_id, get_weights


class LightingBaseRationalizedLanguageModel(pl.LightningModule):
    '''
    PL wrapper for training a language model together with a rational extractor.
    '''

    def __init__(self, language_model, rational_extractor, tokenizer, hparams=None,
                 sparsity_weight=0.1,
                 fussed_lasso_weight=0.1, ):
        super().__init__()
        self.hparams = hparams
        self.language_model = language_model
        self.rational_extractor = rational_extractor
        self.tokenizer = tokenizer

        self.pad_token_id = get_token_id(tokenizer, "pad_token")
        self.mask_token = get_token_id(tokenizer, "mask_token")
        self.sparsity_weight = sparsity_weight
        self.fussed_lasso_weight = fussed_lasso_weight

        self.log_list = [
            "total_loss", "acc", "total_mask_loss", "mask_mean", "mask_fussed_lasso", "cross_entropy_loss",
            "perplexity",
        ]
        self.freeze_language_model = hparams["freeze_language_model"]
        self.hard = False

    def forward(self, x, targets, ):

        rational = self.get_rational(x)

        masked_embedding = rational['masked_embedding']

        ## Concatenate the two together and put through the lstm

        target_embedding = self.language_model.to_embedding(targets)
        lstm_in = torch.cat([masked_embedding, target_embedding])
        prediction = self.language_model.forward_embedding(lstm_in)

        return {"logits": prediction, **rational, "x": x}

    def get_rational(self, x):
        rational_embedding = self.language_model.to_embedding(x)
        rational = self.rational_extractor.forward(rational_embedding, hard=self.hard)
        mask = rational["mask"]
        ### Also apply the mask

        # We need to binarize
        binarized_mask = (mask > 0.5)
        masked_input = torch.mul(x, binarized_mask) + ~binarized_mask * self.mask_token

        return {masked_input: masked_input, **rational}

    def batch_to_out(self, batch):

        # Make batch second
        rational_in = batch[0].permute(1, 0)
        targets = batch[1].permute(1, 0)

        out = self.forward(rational_in, targets)

        targets = targets.long()

        n_targets = targets.shape[0]
        predictions = out["logits"][-(n_targets + 1):-1]
        h_loss = 0
        mask_mean = 0
        fussed_lasso_loss = 0

        if "mask" in out.keys():
            # batch first
            h = out["mask"].permute(1, 0).float()
            tokens = out["x"].permute(1, 0).float()
            mask_mean = calculate_mask_percentage(tokens, h, reduce=True,
                                                  pad_id=self.pad_token_id)
            fussed_lasso_loss = fussed_lasso(tokens, h, reduce=True,
                                             pad_id=self.pad_token_id)

            total_mask_loss = self.sparsity_weight * mask_mean + self.fussed_lasso_weight * fussed_lasso_loss

        weight = get_weights(self.tokenizer).to(targets.device)
        cross_entropy_loss = F.cross_entropy(predictions.view(-1, get_vocab_size(self.tokenizer)), targets.flatten(),
                                             weight=weight)
        total_loss = cross_entropy_loss + total_mask_loss
        perplexity = calc_perplexity(predictions,
                                     targets, self.tokenizer)
        vocab_size = get_vocab_size(self.tokenizer)


        acc = calc_acc(predictions.reshape(-1, vocab_size),
                       targets.flatten(), exclude=self.pad_token_id)


        return {"total_loss": total_loss, "acc": acc, "total_mask_loss": total_mask_loss, "mask_mean": mask_mean,
                "mask_fussed_lasso": fussed_lasso_loss, "cross_entropy_loss": cross_entropy_loss,
                "perplexity": perplexity}

    def training_step(self, batch, batch_idx):

        batch_out = self.batch_to_out(batch)

        self.log_results(batch_out)

        return batch_out["total_loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_out = self.batch_to_out(batch)

        self.log_results(batch_out, prepend="val ")

        return batch_out["total_loss"]

    def complete_dialogues(self, sentences, total_length, with_rational=True, greedy_rationals=True):
        return [
            self.complete_dialogue(sentence, total_length=total_length, with_rational=with_rational)
            for sentence in sentences]

    def complete_dialogue(self, input_sentence, n_rational=10, total_length=100, with_rational=True):

        if type(self.tokenizer) == Tokenizer:
            input_encoding = self.tokenizer.encode(input_sentence).ids
        else:
            input_encoding = self.tokenizer.encode(input_sentence)

        dialogue_tokens = input_encoding

        dialogue_tokens_ids_tensor = torch.tensor(dialogue_tokens).to(self.device).unsqueeze(1)
        rationals = []
        sentences = []
        rationalized_input = []
        while (len(dialogue_tokens_ids_tensor)) < total_length:

            # Extract rationals if needed.
            if with_rational and len(dialogue_tokens_ids_tensor) > n_rational:
                rational = self.get_rational(dialogue_tokens_ids_tensor)
                binary_mask = rational["mask"]
                rational_input = (dialogue_tokens_ids_tensor * binary_mask + ~binary_mask * self.mask_token).int().flatten().detach().cpu().numpy()
                rational_input = self.tokenizer.decode(rational_input, skip_special_tokens=False).replace(" #",
                                                                                                          "").replace(
                    "#", "")
                rationalized_input.append(rational_input)

                rationals.append(binary_mask.flatten())
                embedding = rational["masked_embedding"]
            else:
                rational_input = self.tokenizer.decode(
                    dialogue_tokens_ids_tensor.flatten().flatten().detach().cpu().numpy(),
                    skip_special_tokens=False).replace(" #", "").replace("#", "")
                rationalized_input.append(rational_input)
                rationals.append(torch.tensor([]))
                embedding = self.language_model.to_embedding(dialogue_tokens_ids_tensor)

            next_utterance = self.language_model.generate_next_utterance_from_embedding(embedding, get_token_id(self.tokenizer, "sep_token"))

            dialogue_tokens += next_utterance
            # dialogue_tokens = torch.cat([dialogue_tokens, next_ids])
            # dialogue_tokens.append(next_ids)
            sentences.append(
                self.tokenizer.decode(next_utterance, skip_special_tokens=False).replace(" #", "").replace("#", ""))
            dialogue_tokens_ids_tensor = torch.tensor(dialogue_tokens).to(self.device).unsqueeze(1)

        sentence = self.tokenizer.decode(dialogue_tokens, skip_special_tokens=False).replace(" #", "").replace("#", "")

        return {"completed_dialogue": sentence, "rationals": rationals, "rationalized_input": rationalized_input,
                "response": sentences}

    def configure_optimizers(
            self,
    ):
        if not self.freeze_language_model:
            parameters = list(self.language_model.parameters()) + list(self.rational_extractor.parameters())
        else:
            parameters = list(self.rational_extractor.parameters())

        optimizer = AdamW(
            parameters,
            lr=self.hparams['learning_rate'])
        return optimizer

    def log_results(self, result, prepend=""):

        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)

    def calc_acc(self, predictions, targets):
        indices = torch.argmax(predictions, dim=-1)

        correct = indices == targets
        return torch.mean(correct.float())
