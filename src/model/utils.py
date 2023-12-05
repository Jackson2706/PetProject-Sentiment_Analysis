import time

import torch
from tqdm import tqdm


def train(
    model, optimizer, criterion, train_dataloader, epochs=0, log_interval=50
):
    model.train()
    epoch_acc_list, epoch_loss_list = [], []
    total_acc, total_count = 0, 0
    for epoch in tqdm(range(epochs)):
        losses = []
        start_time = time.time()
        idx = 0
        for inputs, offsets, labels in train_dataloader:
            idx += 1
            optimizer.zero_grad()
            predictions = model(inputs, offsets)

            # compute loss
            loss = criterion(predictions, labels)
            losses.append(loss.item())

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch,
                        idx,
                        len(train_dataloader),
                        total_acc / total_count,
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()

        epoch_acc = total_acc / total_count
        epoch_loss = sum(losses) / len(losses)
    epoch_acc = sum(epoch_acc_list) / len(epoch_acc_list)
    epoch_loss = sum(epoch_loss_list) / len(epoch_loss_list)
    return epoch_acc, epoch_loss


def evaluate(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for inputs, offsets, labels in tqdm(valid_dataloader):
            predictions = model(inputs, offsets)
            loss = criterion(predictions, labels)
            losses.append(loss)
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss
