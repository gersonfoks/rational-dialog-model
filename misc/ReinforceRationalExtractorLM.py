import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import Embedding, LSTM
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AdamW

from misc.old.NextNPredictionDataset import postprocess_dataloader_out
from utils.utils import fussed_lasso


class ReinforceRationalExtractorLM(nn.Module):

    def __init__(self, embedding_input=11, embedding_size=32, mask_token=0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = Embedding(embedding_input, embedding_size)

        # Layers for prediction
        self.prediction_LSTM = LSTM(embedding_size, hidden_size=int(embedding_size / 2), bidirectional=True)
        self.output_layer = nn.Linear(embedding_size, 2)
        self.mask_token = mask_token

    def forward(self, x, greedy=False):
        e = self.embedding(x)

        # Calculate mask
        h, (h_n, c_n) = self.prediction_LSTM(e)
        logits = self.output_layer(h)
        policy = F.softmax(logits, dim=-1)
        policy_reshaped = policy.view(-1, 2)

        if greedy:
            mask = torch.argmax(policy_reshaped, dim=-1).bool()
        else:
            mask = torch.multinomial(policy_reshaped, 1).bool()

        mask = mask.reshape(x.shape[0], -1)
        masked_input = torch.mul(x, mask) + ~mask * self.mask_token

        # Selects the probabilities of the actual chosen policy.
        mask_to_gather = mask.reshape(x.shape[0], -1, 1).long()
        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"mask_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": mask,
                "masked_input": masked_input}


class RELMPL(pl.LightningModule):
    '''
    PL wrapper for training a language model together with a rational extractor.
    '''

    def __init__(self, language_model, rational_extractor, tokenizer, loss_module, hparams=None,
                 sparsity_weight=0.1,
                 fussed_lasso_weight=0.1, ):
        super().__init__()
        self.hparams = hparams
        self.language_model = language_model
        self.rational_extractor = rational_extractor
        self.loss_module = loss_module
        self.tokenizer = tokenizer
        self.sparsity_weight = sparsity_weight
        self.fussed_lasso_weight = fussed_lasso_weight
        self.log_list = [
            "loss", "acc", "h_loss", "h_mean", "fussed_lasso", "cross_entropy_loss", "perplexity"
        ]
        self.teacher_forcing = hparams["teacher_forcing"]
        self.freeze_language_model = hparams["freeze_language_ml"]

    def forward(self, x, targets, ):

        rational = self.get_rational(x)
        masked_input = rational["masked_input"]
        prediction = self.forward_masked_input(masked_input, targets)
        return {"logits": prediction, **rational}

    def get_rational(self, x):
        return self.rational_extractor(x)

    def forward_masked_input(self, masked_input, targets):

        ## Concatenate the two together and put through the lstm
        lstm_in = torch.cat([masked_input, targets])
        prediction = self.language_model(lstm_in)
        return prediction

    def get_perplexity(self, prediction_logits, targets, reduce=False, weights=None):

        cross_entropy = F.cross_entropy(
            prediction_logits.view(-1, self.tokenizer.get_vocab_size()),
            targets.flatten(), 
            weight=weights, 
            reduce=reduce
        )
        perplexity = torch.exp(cross_entropy.reshape(prediction_logits.shape[0], -1).mean(dim=-1))
        return perplexity

    def get_scores(self, forward_dict, targets):
        """
        Gets a dict with multiply scores of the give prediction
        prediction
        forward_dict: {logits, mask, chosen_policy}
        """
        targets = targets.long()

        n_targets = targets.shape[0]
        predictions = forward_dict["logits"][-(n_targets + 1):-1]

        h_loss = 0
        h_mean = 0
        fussed_lasso_loss = 0

        if "mask" in forward_dict.keys():
            h = forward_dict["mask"].permute(1, 0).float()
            h_mean = torch.mean(h, dim=-1)
            fussed_lasso_loss = fussed_lasso(h, reduce=False)
            h_loss = self.sparsity_weight * h_mean + self.fussed_lasso_weight * fussed_lasso_loss

        if type(self.tokenizer) == Tokenizer:
            cross_entropy_loss = self.loss_module(
                predictions.view(-1, self.tokenizer.get_vocab_size()),
                targets.flatten()
            )
        else:
            cross_entropy_loss = self.loss_module(
                predictions.view(-1, len(self.tokenizer)), targets.flatten(),
                reduce=False
            )
        rewards = cross_entropy_loss + h_loss
        perplexity = torch.exp(cross_entropy_loss.mean(dim=-1)).mean()

        # Get the policy loss.
        total_loss = -torch.mean(rewards.detach() * torch.log(forward_dict["chosen_policy"]))
        if not self.freeze_language_model:
            total_loss += torch.mean(rewards)

        acc = self.calc_acc(predictions, targets)

        return {"loss": total_loss, "acc": acc, "h_loss": h_loss.mean(), "h_mean": h_mean.mean(),
                "fussed_lasso": fussed_lasso_loss.mean(), "cross_entropy_loss": cross_entropy_loss.mean(),
                "perplexity": perplexity}

    def batch_out(self, batch):

        rational_in, targets = postprocess_dataloader_out(batch)
        out = self.forward(rational_in, targets)
        scores = self.get_scores(out, targets)
        return scores

    def training_step(self, batch, batch_idx):

        batch_out = self.batch_out(batch)
        self.log_results(batch_out)
        return batch_out["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        batch_out = self.batch_out(batch)
        self.log_results(batch_out, prepend="val_")
        return batch_out["loss"]

    def complete_dialogues(self, sentences, total_length, with_rational=True, greedy_rationals=True):
        return [self.complete_dialogue(
            sentence, 
            total_length=total_length, 
            with_rational=with_rational,
            greedy_rationals=greedy_rationals
        ) for sentence in sentences]

    def complete_dialogue(self, completed_dialogue, n_rational=10, total_length=100, with_rational=True,
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
                rational_input = self.tokenizer.decode(
                    rational["masked_input"].long().view(-1).detach().cpu().numpy(),
                    skip_special_tokens=False
                ).replace(" #", "").replace("#", "")
                next_input = rational["masked_input"]

                # The mask
                rationals.append(rational["mask"].flatten())

            else:
                next_input = ids_tensor
                rational_input = self.tokenizer.decode(
                    ids_tensor.long().view(-1).detach().cpu().numpy(),
                    skip_special_tokens=False
                ).replace(" #", "").replace("#", "")
            rationalized_input.append(rational_input)

            # Generate next ids based on the masked input
            next_ids = self.language_model.generate_next_tokens(next_input, n_tokens=n_rational)

            # Add to all tokens
            all_tokens = torch.cat([all_tokens, next_ids])

            # Map back to the sentence
            responses.append(self.tokenizer.decode(
                    next_ids.reshape(-1).detach().cpu().numpy(), 
                    skip_special_tokens=False
                ).replace(" #", "").replace("#", "")
            )

            # Map back to tensor
            ids_tensor = all_tokens

        completed_dialogue = self.tokenizer.decode(
            all_tokens.reshape(-1).detach().cpu().numpy(),
            skip_special_tokens=False
        ).replace(" #", "").replace("#", "")

        result = {"completed_dialogue": completed_dialogue, "rationals": rationals,
                  "rationalized_input": rationalized_input,
                  "response": responses}
        return result

    def configure_optimizers(self, ):

        if not self.freeze_language_model:
            parameters = list(self.language_model.parameters()) + list(self.rational_extractor.parameters())
        else:
            parameters = list(self.rational_extractor.parameters())
        optimizer = AdamW(parameters, lr=self.hparams['learning_rate'])
        return optimizer

    def log_results(self, result, prepend=""):

        for k in self.log_list:
            self.log(prepend + k, result[k], on_step=True, on_epoch=True)

    def calc_acc(self, predictions, targets):

        indices = torch.argmax(predictions, dim=-1)
        correct = indices == targets
        return torch.mean(correct.float())
