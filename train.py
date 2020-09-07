import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scorer import cer, wer
from processing import GreedyDecoder
from pathlib import Path


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)
    running_train_loss = 0
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        running_train_loss += loss.item() * spectrograms.shape[0]  # this doesn't match valid()

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print(f'Train Epoch: {epoch} [{batch_idx * len(spectrograms)}/{data_len} '
                  f'({100. * batch_idx / len(train_loader)}%)]\t'
                  f'Loss: {loss.item()}')

    running_train_loss /= len(train_loader)  # should be length of the dataset
    print(f'Running train loss = {running_train_loss}')


def test(model, device, test_loader, criterion, epoch, iter_meter):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(f'Test set: Average loss: {test_loss}, Average CER: {avg_cer} Average WER: {avg_wer}\n')


def main(train_loader, test_loader, hparams, model):

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])  # can't use AdamW for some reason
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    iter_meter = IterMeter()
    for epoch in range(1, hparams['epochs'] + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        test(model, device, test_loader, criterion, epoch, iter_meter)
