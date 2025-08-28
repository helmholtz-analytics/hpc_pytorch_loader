import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..','..')))

from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from datasets.memmap.memmap_reader import MemmapReader

def train(rank, world_size, args):
    # Initialize distributed process group if in distributed mode
    if args.dist:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # Set device
    device = torch.device(f"cuda:{rank}" if args.dist else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Define dataset and DataLoader
    transform = Compose([
                        RandomResizedCrop(224),  # Randomly crop the image to 224x224
                        RandomHorizontalFlip(),  # Randomly flip the image horizontally
                        ToTensor(),              # Convert the image to a PyTorch tensor
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
                    ])
    dataset = MemmapReader(dataset_path=args.dataset_path, transform=transform)

    if args.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Define model, loss, and optimizer
    model = resnet18().to(device)
    if args.dist:
        model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Training loop
    for epoch in range(args.epochs):
        if args.dist:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0 and rank == 0:  # Print only from rank 0
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {running_loss / len(dataloader):.4f}")

    # Cleanup distributed process group
    if args.dist:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 with MemmapReader")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Memmap dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dist", action="store_true", help="Enable distributed training")
    args = parser.parse_args()

    if args.dist:
        # Launch distributed training
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        train(rank=rank, world_size=world_size, args=args)
    else:
        # Run single-process training
        train(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    main()
