import torch
import torch.nn as nn
import argparse
from model import *
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data_process import *
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import math
import os


def train(args, train_dataset, test_dataset, device):
    subgraph = SubGraph()
    globalgraph = GlobalGraph()
    decoder = TrajectoryDecoder(out_features=args.max_groundtruth_length * 4)

    subgraph.to(device)
    subgraph.train()
    subgraph.zero_grad()
    globalgraph.to(device)
    globalgraph.train()
    globalgraph.zero_grad()
    decoder.to(device)
    decoder.train()
    decoder.zero_grad()

    subgraph_optimizer = torch.optim.AdamW(
        subgraph.parameters(), lr=args.subgraph_learning_rate)
    globalgraph_optimizer = torch.optim.AdamW(
        globalgraph.parameters(), lr=args.globalgraph_learning_rate)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.decoder_learning_rate)
    
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) * args.epochs
    subgraph_scheduler = get_linear_schedule_with_warmup(subgraph_optimizer, num_warmup_steps=0, num_training_steps=t_total)
    globalgraph_scheduler = get_linear_schedule_with_warmup(globalgraph_optimizer, num_warmup_steps=0, num_training_steps=t_total)
    decoder_scheduler = get_linear_schedule_with_warmup(decoder_optimizer, num_warmup_steps=0, num_training_steps=t_total)

    if args.saving_path is not None:
        print("*** Loading model from {} ***".format(args.saving_path))
        if os.path.isfile(os.path.join(args.saving_path, "subgraph.pt")):
            subgraph.load_state_dict(torch.load(os.path.join(args.saving_path, "subgraph.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "globalgraph.pt")):
            globalgraph.load_state_dict(torch.load(os.path.join(args.saving_path, "globalgraph.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "decoder.pt")):
            decoder.load_state_dict(torch.load(os.path.join(args.saving_path, "decoder.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "subgraph.pt")):
            subgraph_optimizer.load_state_dict(torch.load(os.path.join(args.saving_path, "subgraph_optimizer.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "globalgraph.pt")):
            globalgraph_optimizer.load_state_dict(torch.load(os.path.join(args.saving_path, "globalgraph_optimizer.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "decoder.pt")):
            decoder_optimizer.load_state_dict(torch.load(os.path.join(args.saving_path, "decoder_optimizer.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "subgraph.pt")):
            subgraph_scheduler.load_state_dict(torch.load(os.path.join(args.saving_path, "subgraph_scheduler.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "globalgraph.pt")):
            globalgraph_scheduler.load_state_dict(torch.load(os.path.join(args.saving_path, "globalgraph_scheduler.pt")))
        if os.path.isfile(os.path.join(args.saving_path, "decoder.pt")):
            decoder_scheduler.load_state_dict(torch.load(os.path.join(args.saving_path, "decoder_scheduler.pt")))

    mse_loss = nn.MSELoss(reduction="mean")
    total_loss, logging_loss = 0.0, 0.0
    global_steps = 1
    print("-" * 80)
    print("*** Begin training ***" )

    for i in tqdm(range(args.epochs), desc='Epoch: '):
        subgraph.zero_grad()
        globalgraph.zero_grad()
        decoder.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        
        for step, batch in enumerate(epoch_iterator):
            subgraph.train()
            globalgraph.train()
            decoder.train()

            features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask = batch

            features = features.to(device)
            subgraph_mask = subgraph_mask.to(device)
            attention_mask = attention_mask.to(device)
            groundtruth = groundtruth.to(device)
            groundtruth_mask = groundtruth_mask.to(device)

            out = subgraph.forward(features, subgraph_mask)
            out = globalgraph.forward(out[:, 0, :].unsqueeze(dim=1), out, attention_mask)

            pred = decoder.forward(out).squeeze(1)
            loss = mse_loss.forward(pred * groundtruth_mask, groundtruth)
            loss.backward()

            total_loss += loss.item()
            if args.local_rank in [-1, 0] and args.enable_logging and global_steps % args.logging_steps == 0:
                print("\n\nLoss:\t {}".format(
                    (total_loss-logging_loss)/args.logging_steps))
                logging_loss = total_loss
            
            if args.local_rank in [-1, 0] and args.evaluate_during_training and global_steps % args.logging_steps == 0:
                evaluate(args, (subgraph, globalgraph, decoder), test_dataset, batch_size=args.eval_batch_size, device=device)

            if args.local_rank in [-1, 0] and global_steps % args.saving_steps == 0:
                save_model((subgraph, globalgraph, decoder, subgraph_optimizer, globalgraph_optimizer, decoder_optimizer, subgraph_scheduler, globalgraph_scheduler, decoder_scheduler))
            
            subgraph_optimizer.step()
            globalgraph_optimizer.step()
            decoder_optimizer.step()
            subgraph_scheduler.step()
            globalgraph_scheduler.step()
            decoder_scheduler.step()
            subgraph.zero_grad()
            globalgraph.zero_grad()
            decoder.zero_grad()

            global_steps += 1

    if test_dataset is not None:
        evaluate(args, (subgraph, globalgraph, decoder), test_dataset, device=device, batch_size=args.eval_batch_size)
    save_model((subgraph, globalgraph, decoder, subgraph_optimizer, globalgraph_optimizer,
                decoder_optimizer, subgraph_scheduler, globalgraph_scheduler, decoder_scheduler))


def evaluate(args, models, dataset: torch.utils.data.TensorDataset, device, batch_size=1):
    print("*** Evaluating ***")
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=batch_size)
    
    subgraph, globalgraph, decoder = models
    subgraph.eval()
    globalgraph.eval()
    decoder.eval()

    mse_loss = nn.MSELoss()
    total_loss = 0.0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask = batch

            features = features.to(device)
            subgraph_mask = subgraph_mask.to(device)
            attention_mask = attention_mask.to(device)
            groundtruth = groundtruth.to(device)
            groundtruth_mask = groundtruth_mask.to(device)

            out = subgraph.forward(features, subgraph_mask)
            out = globalgraph.forward(out[:, 0, :].unsqueeze(dim=1), out, attention_mask)

            pred = decoder.forward(out).squeeze(1)
            loss = mse_loss.forward(pred * groundtruth_mask, groundtruth)
            total_loss += loss.item()
  
    print("Eval mse loss (per point): {}".format(math.sqrt(total_loss / (len(dataset) // batch_size * args.max_groundtruth_length))))
    print("-" * 80)


def save_model(models: tuple):
    subgraph, globalgraph, decoder, subgraph_optimizer, globalgraph_optimizer, decoder_optimizer, subgraph_scheduler, globalgraph_scheduler, decoder_scheduler = models
    torch.save(subgraph.state_dict(), "./save/models/subgraph.pt")
    torch.save(globalgraph.state_dict(), "./save/models/globalgraph.pt")
    torch.save(decoder.state_dict(), "./save/models/decoder.pt")
    torch.save(subgraph_optimizer.state_dict(), "./save/models/subgraph_optimizer.pt")
    torch.save(globalgraph_optimizer.state_dict(), "./save/models/globalgraph_optimizer.pt")
    torch.save(decoder_optimizer.state_dict(), "./save/models/decoder_optimizer.pt")
    torch.save(subgraph_scheduler.state_dict(), "./save/models/subgraph_scheduler.pt")
    torch.save(globalgraph_scheduler.state_dict(), "./save/models/globalgraph_scheduler.pt")
    torch.save(decoder_scheduler.state_dict(), "./save/models/decoder_scheduler.pt")


def build_dataset(features: np.ndarray, subgraph_mask: np.ndarray, attention_mask: np.ndarray, groundtruth: np.ndarray, groundtruth_mask: np.ndarray):
    print("-" * 80)
    print("*** Building dataset ***")

    train_features, test_features, train_subgraph_mask, test_subgraph_mask, train_attention_mask, test_attention_mask, train_groundtruth, test_groundtruth, train_groundtruth_mask, test_groundtruth_mask = train_test_split(features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask, train_size=0.8)

    train_features = torch.from_numpy(train_features).to(dtype=torch.float)
    train_subgraph_mask = torch.from_numpy(
        train_subgraph_mask).to(dtype=torch.float)
    train_attention_mask = torch.from_numpy(
        train_attention_mask).to(dtype=torch.float)
    train_groundtruth = torch.from_numpy(
        train_groundtruth).to(dtype=torch.float)
    train_groundtruth_mask = torch.from_numpy(
        train_groundtruth_mask).to(dtype=torch.float)

    test_features = torch.from_numpy(test_features).to(dtype=torch.float)
    test_subgraph_mask = torch.from_numpy(
        test_subgraph_mask).to(dtype=torch.float)
    test_attention_mask = torch.from_numpy(
        test_attention_mask).to(dtype=torch.float)
    test_groundtruth = torch.from_numpy(test_groundtruth).to(dtype=torch.float)
    test_groundtruth_mask = torch.from_numpy(
        test_groundtruth_mask).to(dtype=torch.float)
    train_dataset = torch.utils.data.TensorDataset(
        train_features,
        train_subgraph_mask,
        train_attention_mask,
        train_groundtruth,
        train_groundtruth_mask
    )

    test_dataset = torch.utils.data.TensorDataset(
        test_features,
        test_subgraph_mask,
        test_attention_mask,
        test_groundtruth,
        test_groundtruth_mask
    )
    print("*** Finish building dataset ***")
    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run VectorNet training and evaluating")
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--subgraph_learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate for subgraph"
    )
    parser.add_argument(
        "--globalgraph_learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate for globalgraph"
    )
    parser.add_argument(
        "--decoder_learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate for decoder"
    )
    parser.add_argument(
        "--root_dir",
        default=None,
        required=True,
        type=str,
        help="Path to data root directory"
    )
    parser.add_argument(
        "--feature_path",
        default=None,
        type=str,
        help="Path to feature directory"
    )
    parser.add_argument(
        "--saving_path",
        default=None,
        type=str,
        help="Path to save model"
    )
    parser.add_argument(
        "--logging_steps",
        default=10,
        type=int,
        help="Number of logging steps"
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--saving_steps",
        default=100,
        type=int,
        help="Number of saving steps"
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true",
        help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--max_groundtruth_length",
        default=30,
        help="Maximum length of groundtruth"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="train batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="eval batch size"
    )
    parser.add_argument(
        "--enable_logging",
        action="store_true",
        help="whether enable logging"
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        help="local rank for distributed training"
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step"
    )

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        # Data parallel or CPU training
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    print("*** Process rank: {}, device: {}, n_gpu: {}, distributed training: {} ***".format(args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)))

    print("*** Loading features ***")
    features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask, max_groundtruth_length = load_features(root_dir=args.root_dir, feature_path=args.feature_path)
    args.max_groundtruth_length = max_groundtruth_length
    print("*** Finish loading features ***")

    train_dataset, test_dataset = build_dataset(
        features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask)

    train(args, train_dataset, test_dataset, device)


if __name__ == "__main__":
    main()
    
