import math
from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PretrainedLanguageModel(BaseLanguageModel):
    ''''
    Load a pretrained DialoGPT language model
    '''

    def __init__(self, pretrained_model='microsoft/DialoGPT-small', tokenizer=None):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.embedding = self.lm.get_input_embeddings()
        self.layers = self.lm.get_output_embeddings()
        self.embedding_size = self.layers.in_features
        self.eos_token_id = tokenizer.eos_token_id

    def forward(self, tokenized_input_ids):
        # return self.model.generate(tokenized_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        return self.lm(tokenized_input_ids).logits

    def save(self, location):
        self.lm.save_pretrained(location)
        print("Model saved")

    def to_embedding(self, x):
        return self.embedding(x)

    def forward_embedding(self, embedding):
        return self.layers(embedding)

    # def complete_dialogue(self, context_tokens_ids, max_length=100):
    #     '''
    #     Complete the dialogue given the context
    #     '''
    #     context_tokens_ids = context_tokens_ids.unsqueeze(dim = 0)
    #     dialog = self.lm.generate(context_tokens_ids, max_length=max_length,
    #                               pad_token_id = self.eos_token_id)
    #     return dialog.squeeze()

    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):
        tokens = []
        tokens_embedding = embedding.clone()
        # ## Initialize:
        # logits = self.forward_embedding(embedding)
        # logits = logits[-1]
        # next_token = self.get_next_token_from_logits(logits)
        
        # # torch.cat(tokens, next_token)
        # tokens.append(next_token)
        # next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        # next_embedding = self.embedding(next_token_tensor)
        for i in range(n_tokens - 1):
            # next_embedding = next_embedding.reshape(1, 1, -1)

            logits = self.forward_embedding(tokens_embedding)[-1]

            next_token = self.get_next_token_from_logits(logits)
            # torch.cat(tokens, next_token)
            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)
            tokens_embedding = torch.cat((tokens_embedding, next_embedding),dim = 0)
        return tokens
        # return torch.tensor(tokens).to(embedding.device)


    @classmethod
    def load(self, location):
        self.lm = AutoModelForCausalLM.from_pretrained(location)
        self.embedding = self.lm.get_input_embeddings()
        self.layers = self.lm.get_output_embeddings()
        return self.lm