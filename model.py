import torch
import torch.nn as nn
from layers import SubGraphLayer


class SubGraph(nn.Module):
    def __init__(self, in_features: int = 7, hidden_size: int = 64):
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

        @input x of shape(num_of_seqs, max_seq_size, in_features)

        @input mask of shape(num_of_seqs, max_seq_size, in_features)

        @return out of shape(num_of_seqs, hidden_size * 2): polyline level feature
        '''
        # x now is shape(*, hidden_size)
        x = self.sglayer1(x, mask)
        x = self.sglayer2(x, mask)
        out = self.sglayer3(x, mask)
        out = self.aggregate(out)
        return out

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(num_of_seqs, max_seq_size, hidden_size * 2)

        @return x of shape(num_of_seqs, hidden_size * 2)
        '''
        # TODO: change dim if input shape is not (*, hidden_size)
        y, _ = torch.max(x, dim=1)
        return y


class GlobalGraph(nn.Module):
    def __init__(self,):
        super(GlobalGraph, self).__init__()
        self.Proj_Q = nn.Linear(in_features=128, out_features=128, bias=False)
        self.Proj_K = nn.Linear(in_features=128, out_features=128, bias=False)
        self.Proj_V = nn.Linear(in_features=128, out_features=128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(num_of_nodes, 2 * hidden_size): polyline node features

        @return out of shape(num_of_nodes, 2 * hidden_size): self-attention output
        '''
        P_q = self.Proj_Q(x)
        P_k = self.Proj_K(x)
        P_v = self.Proj_V(x)
        # print(P_q.shape, P_k.shape)
        out = torch.matmul((torch.matmul(P_q, P_k.t())).softmax(dim=1), P_v)
        return out


class TrajectoryDecoder(nn.Module):
    def __init__(self, out_features: int = 2):
        super(TrajectoryDecoder, self).__init__()
        self.linear = nn.Linear(in_features=128, out_features=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(*, 128)

        @return out of shape(*, 2)
        '''
        out = self.linear(x)
        return out
