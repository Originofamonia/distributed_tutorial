"""
https://github.com/Originofamonia/distributed_tutorial
"""
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex.parallel import DistributedDataParallel as DDP
from torch.cuda import amp


def cleanup():
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of server nodes')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--seed', default=444, type=int,
                        help='seed')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--resume', default=True, type=bool,
                        help='load saved model and resume training')
    parser.add_argument('--ckpt_path', default=f'model.pt', type=str,
                        help='ckpt path')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '14444'

    torch.manual_seed(args.seed)
    train_dataset = torchvision.datasets.CIFAR10(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    model = ConvNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    mp.spawn(train, nprocs=args.gpus, args=(args, train_dataset, model, loss_fn, optimizer))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args, train_dataset, model, loss_fn, optimizer):
    print(f'using GPU: {gpu}')
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    # Wrap the model
    model = DDP(model, device_ids=[gpu])
    # Data loading code
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas=args.world_size,
                                       rank=rank)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,  # must be False
                              num_workers=2,
                              pin_memory=True,
                              sampler=train_sampler)
    
    if args.resume:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(args.ckpt_path, map_location=map_location))

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for j, batch in enumerate(train_loader):
            batch = tuple(item.cuda() for item in batch)
            images, labels = batch
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j) % 100 == 0 and gpu == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{j}/{total_step}], Loss: {loss.item():.3f}')
    
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.state_dict(), args.ckpt_path)

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    
    cleanup()


if __name__ == '__main__':
    main()
