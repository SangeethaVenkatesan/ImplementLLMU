import torch 
import torch.nn as nn

# Implementation learned from https://www.youtube.com/watch?v=U0s0f995w14&t=1s # Alladin Persson #

class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # Heads - Number of heads to split the embed_size into # 
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * heads == embed_size), "Embedding size needs to be divisible by heads"

        # Linear layers for Query, Key and Value #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Performing the Linear transformation on the incoming data # 
        # y = x * W^T + b #
        self.query = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.key = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.value = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.fully_connected = nn.Linear(self.head_dimension * heads, embed_size)

    # Forward pass #
    def forward(self, value, key, query, mask):
        # piece of multi head attention #
        N = query.shape[0] # Number of training examples #
        # Depends on the source sentence length  and the target sentence length #
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        # split the embedding into self.heads pieces #
        value = value.reshape(N, value_len, self.heads, self.head_dimension)
        key = key.reshape(N, key_len, self.heads, self.head_dimension)
        query = query.reshape(N, query_len, self.heads, self.head_dimension)
        # query: what we are looking for #
        # key: what we have? #
        # value: what we want to output #
        # https://pytorch.org/docs/stable/generated/torch.einsum.html?highlight=einsum
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        """
        Einsum allows computing many common multi-dimensional linear algebraic array operations 
        by representing them in a short-hand format 
        based on the Einstein summation convention, given by equation.
        """
        # Queries shape - (N, query_len, heads, head_dimension) #
        # Keys shape - (N, key_len, heads, head_dimension) #
        # Energy shape - (N, heads, query_len, key_len) #

        # Masking #
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # Masking all the values which are 0 with a very small number #
        
        # Attention scores #
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        # Multiply the attention with the value #
        # Attention shape - (N, heads, query_len, key_len) #
        # Value shape - (N, value_len, heads, head_dimension) #
        # Output shape - (N, heads, query_len, head_dimension) #
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.heads * self.head_dimension)
        # Concatenating the heads #
        out = self.fully_connected(out)
        return out

class TransformerBlock(nn.Module):
    # Consists of Multi-Head Attention and a Feed Forward Network  with a normalization layer #
    def __init__(self,embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = selfAttention(embed_size, heads)
        # Normalization layer #
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm
        # per example 
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed Forward Network #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size),
                                            nn.ReLU(),
                                            nn.Linear(forward_expansion * embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout #

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # Add and Norm #
        x = self.dropout(self.norm1(attention + query)) # this is skip connection #
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # this is skip connection #
        return out
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        # max length specifies the maximum length of the source sentence - > positional embedding #
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Word Embedding #
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            # In Encoder - Value, Key and Query are all the same #
            out = layer(out, out, out, mask)
        return out




