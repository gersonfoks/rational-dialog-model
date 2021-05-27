import torch
import pytorch_lightning as pl
from tokenizers import Tokenizer
from transformers import AdamW

from utils.token_utils import get_vocab_size, get_token_id, get_weights
from utils.utils import fussed_lasso, calc_acc, calculate_mask_percentage, get_pad_id, calc_perplexity, \
    calc_cross_entropy_batch_wise, calc_policy_loss


class LightingReinforceRationalizedLanguageModel(pl.LightningModule):
    '''
    PL wrapper for training a language model together with a rational extractor.
    '''

    def __init__(self, language_model, rational_extractor, tokenizer, hparams=None,
                 sparsity_weight=0.0001,
                 fussed_lasso_weight=0.1):
        super().__init__()
        self.hparams = hparams
        self.language_model = language_model
        self.rational_extractor = rational_extractor
        self.tokenizer = tokenizer

        self.sparsity_weight = sparsity_weight
        self.fussed_lasso_weight = fussed_lasso_weight

        self.log_list = [
            "total_loss", "acc", "total_mask_loss", "mask_mean", "mask_fussed_lasso", "cross_entropy_loss",
            "perplexity",
        ]

        self.pad_token_id = get_pad_id(tokenizer)

        self.freeze_language_model = hparams["freeze_language_model"]

    def forward(self, x, targets, ):

        rational = self.get_rational(x)

        masked_input = rational["masked_input"]

        prediction = self.forward_masked_input(masked_input, targets)
        return {"logits": prediction, **rational, "x": x}

    def get_rational(self, x, greedy=False):
        if greedy:
            print(greedy)
            return self.rational_extractor(x, greedy=greedy)
        else:
            return self.rational_extractor(x)

    def forward_masked_input(self, masked_input, targets):
        ## Concatenate the two together and put through the lstm
        lm_in = torch.cat([masked_input, targets])

        prediction = self.language_model(lm_in)
        return prediction

    def get_scores(self, forward_dict, targets):
        """
        Gets a dict with multiply scores of the give prediction
        prediction
        forward_dict: {logits, mask, chosen_policy}
        """
        targets = targets.long()

        n_targets = targets.shape[0]
        predictions = forward_dict["logits"][-(n_targets + 1):-1]

        total_h_loss = 0
        h_mean = 0
        fussed_lasso_loss = 0

        if "mask" in forward_dict.keys():
            ### Make sure batch first.
            mask = forward_dict["mask"].permute(1, 0).float()
            tokens = forward_dict["x"].permute(1, 0).float()
            h_mean = calculate_mask_percentage(tokens, mask, reduce=False,
                                               pad_id=get_token_id(self.tokenizer, "pad_token"))
            fussed_lasso_loss = fussed_lasso(tokens, mask, reduce=False,
                                             pad_id=get_token_id(self.tokenizer, "pad_token"))
            total_h_loss = self.sparsity_weight * h_mean + self.fussed_lasso_weight * fussed_lasso_loss

        cross_entropy_loss = calc_cross_entropy_batch_wise(predictions, targets, self.tokenizer)
        ##Make sure it is mean per batch

        rewards = cross_entropy_loss + total_h_loss
        vocab_size = get_vocab_size(self.tokenizer)
        perplexity = calc_perplexity(predictions,
                                     targets, self.tokenizer)

        # Get the policy loss. (old one)
        # total_loss = -torch.mean(rewards.detach() * torch.log(forward_dict["chosen_policy"]))
        total_loss = calc_policy_loss(rewards, forward_dict["chosen_policy"])

        if not self.freeze_language_model:
            total_loss += torch.mean(rewards)

        acc = calc_acc(predictions.reshape(-1, vocab_size),
                       targets.flatten(), exclude=self.pad_token_id)

        return {"total_loss": total_loss, "acc": acc, "total_mask_loss": total_h_loss.mean(),
                "mask_mean": h_mean.mean(),
                "mask_fussed_lasso": fussed_lasso_loss.mean(), "cross_entropy_loss": cross_entropy_loss.mean(),
                "perplexity": perplexity}

    def batch_to_out(self, batch):
        rational_in = batch[0].permute(1, 0)
        targets = batch[1].permute(1, 0)

        out = self.forward(rational_in, targets)

        scores = self.get_scores(out, targets)

        return scores

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
        return [self.complete_dialogue(sentence, total_length=total_length, with_rational=with_rational,
                                       greedy_rationals=greedy_rationals) for sentence in
                sentences]

    def complete_dialogue(self, completed_dialogue, total_length=100, with_rational=True,
                          greedy_rationals=True):

        if type(self.tokenizer) == Tokenizer:
            encoding = self.tokenizer.encode(completed_dialogue).ids
        else:
            encoding = self.tokenizer.encode(completed_dialogue)

        all_tokens = torch.tensor(encoding).to(self.device).unsqueeze(1)

        ids_tensor = all_tokens
        rationals = []
        responses = []
        rationalized_input = []
        while (len(ids_tensor)) < total_length:

            if with_rational:
                # Get the rational
                rational = self.rational_extractor(ids_tensor, greedy=greedy_rationals)

                # Map back to tokens
                rational_input = self.tokenizer.decode(rational["masked_input"].long().view(-1).detach().cpu().numpy(),
                                                       skip_special_tokens=False).replace(" #",
                                                                                          "").replace(
                    "#", "")
                next_input = rational["masked_input"]
                # The mask
                rationals.append(rational["mask"].flatten())


            else:
                next_input = ids_tensor
                rational_input = self.tokenizer.decode(ids_tensor.long().view(-1).detach().cpu().numpy(),
                                                       skip_special_tokens=False).replace(" #",
                                                                                          "").replace(
                    "#", "")
            rationalized_input.append(rational_input)

            # Generate next ids based on the masked input
            # next_ids = self.language_model.generate_next_tokens(next_input, n_tokens=n_rational)
            next_ids = self.language_model.next_utterance(next_input.flatten(),
                                                          get_token_id(self.tokenizer, "sep_token"),
                                                          max_length=20).reshape(-1, 1)
            # next_ids = self.language_model.lm.generate(next_input.reshape(1,-1), 
            #                                             eos_token_id=self.tokenizer.sep_token_id,
            #                                             num_beams=5,
            #                                             min_length=5, 
            #                                             max_length=1000,#(20+len(next_input)),
            #                                             forced_eos_token_id=self.tokenizer.sep_token_id,
            #                                             ).reshape(-1,1) #TODO keep it to check if it will work in the future
            # Add to all tokens
            all_tokens = torch.cat([all_tokens, next_ids])

            # Map back to the sentence
            responses.append(
                # self.tokenizer.decode(next_ids.reshape(-1).detach().cpu().numpy(), skip_special_tokens=False).replace(" #", "").replace("#", "")
                self.tokenizer.decode(next_ids.reshape(-1).detach().cpu().numpy(), skip_special_tokens=False)
            )

            # Map back to tensor
            ids_tensor = all_tokens

        completed_dialogue = self.tokenizer.decode(all_tokens.reshape(-1).detach().cpu().numpy(),
                                                   skip_special_tokens=False).replace(" #", "").replace("#", "")

        result = {"completed_dialogue": completed_dialogue, "rationals": rationals,
                  "rationalized_input": rationalized_input,
                  "response": responses}

        return result

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
