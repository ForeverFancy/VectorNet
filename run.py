import torch
import torch.nn as nn
import argparse
from model import *


def train(args: argparse.Namespace):
    subgraph = SubGraph()
    globalgraph = SubGraph()
    subgraph.train()
    globalgraph.train()
    subgraph_optimizer = torch.optim.Adam(
        subgraph.parameters(), lr=args.subgraph_learning_rate)
    globalgraph_optimizer = torch.optim.Adam(
        globalgraph.parameters(), lr=args.globalgraph_learning_rate)
    loss = nn.MSELoss()
    for i in range(args.epochs):
        pass


def evaluate():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VectorNet training and evaluating")
    args = parser.parse_args()
    train(args)
