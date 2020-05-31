import torch
import torch.nn as nn


class SubGraphLayer(nn.Module):
    def __init__(self, in_features: int = 7, hidden_size: int = 64):
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
        TODO: input size not fixed, do padding?

        @input x of shape(num_of_seqs, max_seq_size, in_features)

        @return out of shape(num_of_seqs, max_seq_size, hidden_size * 2): output features
        '''
        # x now is shape(*, hidden_size)
        x = self.encode(x)
        # print(x.shape)
        ag = self.aggregate(x).repeat(1, x.shape[1], 1)
        # print(ag.shape)
        out = torch.cat([x, ag], dim=2)
        # print(out.shape)
        out = out * mask.unsqueeze(dim=-1).repeat(1, 1, self.hidden_size * 2)
        assert out.shape == (x.shape[0], x.shape[1], self.hidden_size * 2)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(num_of_seqs, max_seq_size, in_features)

        @return x of shape(num_of_seqs, max_seq_size, hidden_size)
        '''
        x = self.linear(x)
        x = self.lm(x)
        return torch.relu(x)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        '''
        @input x of shape(num_of_seqs, max_seq_size, hidden_size)

        @return x of shape(num_of_seqs, hidden_size)
        '''
        # TODO: change dim if input shape is not (*, hidden_size)
        y, _ = torch.max(x, dim=1)
        return y.unsqueeze(1)
