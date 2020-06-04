import torch
import torch.nn as nn
import argparse
from model import *
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm


def train(epochs, subgraph_learning_rate, globalgraph_learning_rate, dataset, batch_size=1, save_steps=20, device="cpu"):
    subgraph = SubGraph()
    globalgraph = SubGraph()
    subgraph.to(device)
    subgraph.train()
    subgraph.zero_grad()
    globalgraph.to(device)
    globalgraph.train()
    globalgraph.zero_grad()

    subgraph_optimizer = torch.optim.Adam(
        subgraph.parameters(), lr=subgraph_learning_rate)
    globalgraph_optimizer = torch.optim.Adam(
        globalgraph.parameters(), lr=globalgraph_learning_rate)

    mse_loss = nn.MSELoss()
    total_loss = 0.0

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, batch_sampler=train_sampler, batch_size=batch_size, shuffle=True)

    for i in range(epochs):
        print("-" * 80)
        print("*** Begin epoch {} ***".format(i + 1))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        subgraph.zero_grad()
        globalgraph.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            features, mask = batch
            out = subgraph.forward(features, mask)
            # print(out.shape)
            out = globalgraph.forward(out)
            loss = mse_loss(out)
            loss.backward()

            total_loss += loss.item()

            if (step + 1) % save_steps == 0:
                torch.save(subgraph.state_dict(), "./save/subgraph.pt")
                torch.save(subgraph_optimizer.state_dict(),
                           "./save/subgraph_optimizer.pt")
                torch.save(globalgraph.state_dict(), "./save/globalgraph.pt")
                torch.save(globalgraph_optimizer.state_dict(),
                           "./save/globalgraph_optimizer.pt")
            
            subgraph_optimizer.step()
            globalgraph_optimizer.step()
            subgraph.zero_grad()
            globalgraph.zero_grad()


def evaluate(models, dataset: torch.utils.data.TensorDataset, batch_size=1, device="cpu"):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=batch_size)
    
    subgraph, globalgraph = models
    subgraph.eval()
    globalgraph.eval()

    mse_loss = nn.MSELoss()
    total_loss = 0.0
    

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            features, mask = batch
            out = subgraph.forward(features, mask)
            # print(out.shape)
            out = globalgraph.forward(out)
            loss = mse_loss(out)
            total_loss += loss.item()
            
    print("-" * 80)
    print("Eval mse loss: {}".format(total_loss/len(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VectorNet training and evaluating")
    args = parser.parse_args()
    # train(args)
