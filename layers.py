import torch
import torch.nn as nn


class SubGraphLayer(nn.Module):
    def __init__(self, in_features: int = 7, hidden_size: int = 64):
        '''
        Layer of subgraph
        '''
        super(SubGraphLayer, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            bias=True
        )
        self.lm = nn.LayerNorm(normalized_shape=(self.hidden_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape (batch_size, num_of_seqs, max_seq_size, in_features)

        @input mask of shape (batch_size, num_of_seqs, max_seq_size)

        @return out of shape (batch_size, num_of_seqs, max_seq_size, hidden_size * 2): output features
        '''
        x = self.encode(x)
        ag = self.aggregate(x).repeat(1, 1, x.shape[2], 1)
        out = torch.cat([x, ag], dim=3)
        assert out.shape == (x.shape[0], x.shape[1], x.shape[2], self.hidden_size * 2)
        out = out * (mask.unsqueeze(dim=-1).repeat(1, 1, 1, self.hidden_size * 2))
        assert out.shape == (x.shape[0], x.shape[1], x.shape[2], self.hidden_size * 2)

        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''
        MLP + layer normalization + relu activation

        @input x of shape (batch_size, num_of_seqs, max_seq_size, in_features)

        @return x of shape (batch_size, num_of_seqs, max_seq_size, hidden_size)
        '''
        x = self.linear(x)
        x = self.lm(x)
        return torch.relu(x)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Use maxpooling to aggregate

        @input x of shape (batch_size, num_of_seqs, max_seq_size, hidden_size)

        @return x of shape (batch_size, num_of_seqs, hidden_size)
        '''
        y, _ = torch.max(x, dim=2)
        return y.unsqueeze(2)


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        '''
        Self-attention layer
        '''
        super(SelfAttentionLayer, self).__init__()
        self.Proj_Q = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )
        self.Proj_K = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )
        self.Proj_V = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, query: torch.Tensor, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        '''
        Do self-attention

        @input query of shape(batch_size, 1, 2 * hidden_size): agent features

        @input x of shape (batch_size, num_of_nodes, 2 * hidden_size): all polyline node features

        @input attention_mask of shape (batch_size, num_of_nodes): mask for self attention

        @return out of shape(batch_size, num_of_nodes, 2 * hidden_size): self-attention output
        '''
        P_q = self.Proj_Q(query)
        P_k = self.Proj_K(x)
        P_v = self.Proj_V(x)
        out = torch.bmm(P_q, P_k.transpose(1, 2))
        # mask for self attention
        if attention_mask is not None:
            out = out.masked_fill(attention_mask.unsqueeze(1).expand(-1, query.shape[1], -1) == 0, -1e9)
        out = torch.bmm(self.softmax(out), P_v)
        return out
