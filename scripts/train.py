import os
import time
import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from config import train_test
from dataset.custom_dataset import CustomDataset
from model.u_net import UNet
from utils import io_utils


def train():
    # logging
    logging.basicConfig(format='%(asctime)s \t %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    # directory
    os.makedirs(train_test.file_path['model_dir'], exist_ok=True)

    # device
    gpu_ids = [int(id) for id in train_test.train_utils['gpu_ids'].split(',')]
    DEVICE = torch.device("cuda:" + str(gpu_ids[0]) if torch.cuda.is_available() else "cpu")
    logging.info(f'device is {DEVICE}')
    
    # model
    logging.info(f'init model...')
    unet = UNet(in_channels=1, n_classes=2, bilinear=True).float()
    if torch.cuda.device_count() > 1:
        unet = nn.DataParallel(unet, device_ids=gpu_ids)
    unet = unet.to(DEVICE)
    unet.train()

    # optimizer
    logging.info(f'init optimizer...')
    optimizer = optim.Adam(unet.parameters(), lr=train_test_cfg.training_utils['learning_rate'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

    # loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10], device=DEVICE))

    # dataloader
    logging.info(f'init dataloader...')
    train_set = CustomDataset(is_train=True)
    train_loader = DataLoader(
        train_set,
        batch_size=train_test.train_utils['batch_size'],
        shuffle=True,
        num_workers=train_test.train_utils['num_of_workers'],
    )

    # main
    start_time = time.time()
    for epoch in range(train_test.train_utils['num_of_epochs']):
        for batch, info in enumerate(train_loader):
            optimizer.zero_grad()
            data, label = info
            pred = unet(data.float().cuda())
            loss = criterion(pred, label.float().cuda())
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                logging.info(
                    f'Time: {(time.time() - start_time) // 60} min,\t Epoch: {epoch},\t Batch: {batch},\t Loss: {loss.item()}'
                )

        scheduler.step()
        if epoch % 1 == 0:
            save_checkpoint_path = os.path.join(train_test.file_path['model_dir'],
                                                f'epoch{epoch}_batch{batch}.tar')
            io_utils.save_model(unet, optimizer, epoch, batch, loss.item(), save_checkpoint_path)


if __name__ == '__main__':
    train()
