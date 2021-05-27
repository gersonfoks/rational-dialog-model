import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from modules.LanguageModels.BaseLanguageModel import BaseLanguageModel


class LSTMLanguageModel(BaseLanguageModel):
    ''''
    A simple lstm language model
    '''

    def __init__(self, num_embeddings, num_layers=2, embedding_dim=128, hidden_state_size=128, ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_state_size = hidden_state_size
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_state_size, num_layers=num_layers, )
        self.relu = F.relu
        self.classification_layer = nn.Linear(hidden_state_size, num_embeddings)
        self.embedding_size = embedding_dim

    def forward_embedding(self, embedding, teacher_forcing=True, n_to_predict=0):
        if teacher_forcing:
            out, hidden = self.lstm(embedding)
            out = self.relu(out)
            classification = self.classification_layer(out, )

        else:
            # Generate up to n tokens
            out, hidden = self.lstm(embedding)

            # hidden = (hidden[0][-1:, :, :].contiguous(), hidden[1][-1:, :, :].contiguous())
            out = F.relu(out[-1:, :, :])  # Get the latest token
            next_out = self.classification_layer(out)
            next_token = torch.argmax(next_out, dim=-1)
            next_embedding = self.embedding(next_token)
            outs = [next_out]

            for i in range(n_to_predict - 1):
                next_embedding = next_embedding.reshape(1, embedding.shape[1], -1)
                out, hidden = self.lstm(next_embedding, hidden)
                out = F.relu(out)
                next_out = self.classification_layer(out)
                next_token = torch.argmax(next_out, dim=-1)
                next_embedding = self.embedding(next_token)
                outs.append(next_out)

            classification = torch.cat(outs)

        return classification

    def forward(self, x):
        embedding = self.to_embedding(x)
        out, hidden = self.lstm(embedding)
        out = self.relu(out)
        classification = self.classification_layer(out, )
        return classification


    def generate_next_tokens_from_embedding(self, embedding, n_tokens=10):

        ## Initialize:
        tokens = []

        # Get the latest token
        out, hidden = self.lstm(embedding)
        out = F.relu(out[-1, 0])  

        logits = self.classification_layer(out)
        next_token = self.get_next_token_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)

        for i in range(n_tokens - 1):
            next_embedding = next_embedding.reshape(1, 1, -1)
            out, hidden = self.lstm(next_embedding, hidden)
            out = F.relu(out)
            logits = self.classification_layer(out)

            next_token = self.get_next_token_from_logits(logits)


            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)

        return tokens

    def generate_next_utterance_from_embedding(self, embedding, sep_token, max_length=100,):
        assert (type(sep_token) == int)
        ## Initialize:
        tokens = []
        embedding = embedding.clone()
        # Get the latest token
        out, hidden = self.lstm(embedding)
        out = F.relu(out[-1, 0])

        logits = self.classification_layer(out)
        next_token = self.get_next_token_from_logits(logits)

        tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
        next_embedding = self.embedding(next_token_tensor)

        while len(tokens) < max_length and next_token != sep_token:
            next_embedding = next_embedding.reshape(1, 1, -1)
            out, hidden = self.lstm(next_embedding, hidden)
            out = F.relu(out)
            logits = self.classification_layer(out)

            next_token = self.get_next_token_from_logits(logits)

            tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]]).to(embedding.device)
            next_embedding = self.embedding(next_token_tensor)

        return tokens

    def generate_next_token(self, tokens):
        tokens = tokens.view(-1, 1)
        logits = self.forward(tokens)[-1, 0, :]
        next_token = self.get_next_token_from_logits(logits)
        next_token = torch.tensor([next_token]).to(tokens.device)
        return next_token


    def save(self, location):
        print("save")
        torch.save({
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'num_embeddings': self.num_embeddings,
                'num_layers': self.num_layers,
                'embedding_dim': self.embedding_dim,
                'hidden_state_size': self.hidden_state_size,
            }
        }, location)

    @classmethod
    def load(self, location):
        info = torch.load(location)
        model = LSTMLanguageModel(**info['kwargs'])
        model.load_state_dict(info['model_state_dict'])
        model.train()
        return model
