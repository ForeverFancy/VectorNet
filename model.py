import torch
import torch.nn as nn
from layers import SubGraphLayer
import numpy as np


class SubGraph(nn.Module):
    def __init__(self, in_features: int = 7, hidden_size: int = 64):
        '''
        Subgraph model
        '''
        super(SubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.sglayer1 = SubGraphLayer(
            in_features=self.in_features, hidden_size=self.hidden_size)
        self.sglayer2 = SubGraphLayer(
            in_features=self.hidden_size * 2, hidden_size=self.hidden_size)
        self.sglayer3 = SubGraphLayer(
            in_features=self.hidden_size * 2, hidden_size=self.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        TODO: input size not fixed, do padding?

        @input x of shape(batch_size, num_of_seqs, max_seq_size, in_features)

        @input mask of shape(batch_size, num_of_seqs, max_seq_size, in_features)

        @return out of shape(batch_size, num_of_seqs, hidden_size * 2): polyline level feature
        '''
        # x now is shape(*, hidden_size)
        x = self.sglayer1(x, mask)
        x = self.sglayer2(x, mask)
        out = self.sglayer3(x, mask)
        out = self.aggregate(out)
        return out

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(batch_size, num_of_seqs, max_seq_size, hidden_size * 2)

        @return x of shape(batch_size, num_of_seqs, hidden_size * 2)
        '''
        # TODO: change dim if input shape is not (*, hidden_size)
        y, _ = torch.max(x, dim=2)
        # print(y.shape)
        return y


class GlobalGraph(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 128):
        '''
        Global graph model
        '''
        super(GlobalGraph, self).__init__()
        self.Proj_Q = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.Proj_K = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.Proj_V = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query: torch.Tensor, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
        # print(attention_mask, attention_mask.shape)
        out = torch.bmm(P_q, P_k.transpose(1, 2))
        print(out.shape, attention_mask.unsqueeze(1).expand(-1, query.shape[1], -1).shape)
        # mask for self attention
        out = out.masked_fill(attention_mask.unsqueeze(1).expand(-1, query.shape[1], -1) == 0, -1e9)
        out = torch.bmm(self.softmax(out), P_v)
        return out


class TrajectoryDecoder(nn.Module):
    def __init__(self, in_features: int = 128, out_features: int = 2):
        '''
        Decode future trajectory
        '''
        super(TrajectoryDecoder, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(*, in_features)

        @return out of shape(*, out_features)
        '''
        out = self.linear(x)
        return out
