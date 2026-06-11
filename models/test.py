#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
try:
    import torch_npu
except ImportError:
    pass
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, num_workers=args.num_workers)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data = data.to(args.device)
        target = target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.item(), test_loss

def test_text(net, dataset, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size_text,
        shuffle=False,
        num_workers=0
    )
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch.get('attention_mask')
            token_type_ids = batch.get('token_type_ids')
            labels = batch['labels'].to(args.device)
            forward_kwargs = {
                'input_ids': input_ids,
                'labels': labels,
            }
            if attention_mask is not None:
                forward_kwargs['attention_mask'] = attention_mask.to(args.device)
            if token_type_ids is not None:
                forward_kwargs['token_type_ids'] = token_type_ids.to(args.device)

            outputs = net(**forward_kwargs)
            
            test_loss += outputs.loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100. * correct / max(1, total)
    avg_loss = test_loss / max(1, total)
    
    return accuracy, avg_loss
