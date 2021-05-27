import torch
from torch import nn
from torch.nn import Embedding, LSTM
import torch.nn.functional as F


class RandomRationalExtractor(nn.Module):

    def __init__(self, percentage=50, mask_token=0):
        '''
        Percentage: percentage that should be part of the rational.
        '''
        super().__init__()

        self.percentage = percentage
        self.mask_token = mask_token

    def forward(self, x, greedy=False, hard=False):
        '''
            Given a batch of sequences it outputs information about the policy, the actual chosen policy,
            the mask that was used and the masked x.
        '''

        # Just to make sure everything works:
        logits = torch.rand((x.shape[0], x.shape[1], 2)).to(x.device)
        ## Just filler
        policy = F.softmax(logits, dim=-1)




        mask = (torch.rand(x.shape) < self.percentage).to(x.device)

        masked_input = torch.mul(x, mask) + ~mask * self.mask_token

        # Selects the probabilities of the actual chosen policy.
        mask_to_gather = mask.reshape(x.shape[0], -1, 1).long()
        chosen_policy = torch.gather(policy, -1, mask_to_gather)

        return {"policy_logits": logits, "policy": policy, "chosen_policy": chosen_policy, "mask": mask,
                "masked_input": masked_input}

    def save(self, location):

        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                "mask_token": self.mask_token,
                "percentage": self.percentage
            }
        }, location)

    @classmethod
    def load(self, location):

        info = torch.load(location)
        model = RandomRationalExtractor(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model
