import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm
from datetime import datetime


from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    train_dataset = FoodDataset("./data", "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = FoodDataset("./data", "val", transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
    
    model.to(device)
    
    ##########################   fill here   ###########################
        
    # TODO : Training Loop을 작성해주세요
    # 1. logger, optimizer, criterion(loss function)을 정의합니다.
    # train loader는 training에 val loader는 epoch 성능 측정에 사용됩니다.
    # torch.save()를 이용해 epoch마다 model이 저장되도록 해 주세요
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger(__name__)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
   
    for epoch in range(1, args.epoch + 1):
        logger.info(f"Training epoch {epoch}")
        model.train()
        total_loss = 0.0

        for batch, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}')):
            inputs, targets = data['input'].to(device), data['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            
        # Calculate and log average loss for the epoch
        average_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} loss: {average_loss}")

        # Validation
        logger.info(f"Validating epoch {epoch}")
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f'Validating Epoch {epoch}'):
                inputs, targets = data['input'].to(device), data['target'].to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        logger.info(f"Epoch {epoch} accuracy: {accuracy}")

        # Save model checkpoint
        checkpoint_filename = f'{epoch}_score:{accuracy:.3f}.pth'
        checkpoint_path = os.path.join(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', checkpoint_filename)
        torch.save(model.state_dict(), checkpoint_path)
    ######################################################################