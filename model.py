import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        # Set up the parent PyTorch class, nn.Module.
        # This lets PyTorch track our layers and their trainable values.
        super().__init__()#not sure about this line

        # Save the number of dimensions in every embedding vector.
        # Example: if d_model is 512, every token becomes 512 numbers.
        self.d_model = d_model

        # Save how many different token IDs exist.
        # Example: vocab_size=10000 means token IDs range from 0 to 9999.
        self.vocab_size = vocab_size

        # Create an embedding lookup table.
        #
        # Input: a token ID, such as 42
        # Output: a vector with `d_model` numbers
        #
        # It has one vector for every possible token ID.
        self.embedding = nn.Embedding(vocab_size, d_model)

    # This defines what happens when token IDs enter this layer.
    #
    # `x` should be a PyTorch tensor containing token IDs, for example:
    # x = torch.tensor([5, 12, 83])
    def forward(self, x):

        # Look up an embedding vector for every token ID in x.
        # Then multiply the vectors by sqrt(d_model), which is a standard
        # Transformer scaling step.
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout: float ) -> None:
        super().__init__()#setup the pytorch nn.module parent class
        self.d_model = d_model
        self.seq_len = seq_len
        #Saves these settings inside this particular model object.
        self.dropout = nn.Dropout(dropout)
        #Creates a dropout layer. For example, with dropout=0.1, it randomly sets roughly 10% of values to zero during training.

        #Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)#Creates a matrix full of zeroes
        #Create a vector of shape (Seq_Len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #This creates different frequency values for the positional patterns. The exact formula is from the original Transformer paper.
        #Apply the sin to even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_Len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)

    class LayerNormalization(nn.Module):

        def __init__(self, eps: float = 10** -6) -> None:
            super().__init__()
            self.eps = eps
            self.alpha = nn.Parameter(torch.ones(1)) #multiplied
            self.bias = nn.Parameter(torch.zeros(1)) #added

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std = x.std(dim = -1, keepdim = True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias