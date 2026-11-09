#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import math

import torch
try:
    import torch_npu
except ImportError:
    pass
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, return_diagnostics=False):
    net_g.eval()
    test_loss = 0.0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, num_workers=args.num_workers)
    dataset_size = len(data_loader.dataset)
    if dataset_size <= 0:
        raise ValueError('test_img requires a non-empty evaluation dataset')
    diagnostics = {
        'eval_batches': 0,
        'eval_nonfinite_input_batches': 0,
        'eval_nonfinite_logit_batches': 0,
        'eval_nonfinite_loss_batches': 0,
        'eval_nonfinite_prediction_batches': 0,
        'eval_loss_finite': True,
        'eval_logits_finite': True,
        'eval_inputs_finite': True,
        'eval_predictions_valid': True,
    }
    with torch.no_grad():
        for data, target in data_loader:
            diagnostics['eval_batches'] += 1
            data = data.to(args.device)
            target = target.to(args.device)
            inputs_finite = bool(torch.isfinite(data).all().item())
            diagnostics['eval_inputs_finite'] &= inputs_finite
            if not inputs_finite:
                diagnostics['eval_nonfinite_input_batches'] += 1

            log_probs = net_g(data)
            logits_finite = bool(torch.isfinite(log_probs).all().item())
            diagnostics['eval_logits_finite'] &= logits_finite
            if not logits_finite:
                diagnostics['eval_nonfinite_logit_batches'] += 1

            batch_loss = F.cross_entropy(log_probs, target, reduction='sum')
            loss_finite = bool(torch.isfinite(batch_loss).item())
            diagnostics['eval_loss_finite'] &= loss_finite
            if not loss_finite:
                diagnostics['eval_nonfinite_loss_batches'] += 1
            test_loss += float(batch_loss.item())

            # 非有限 logits 的 argmax 仍可能给出整数，不能据此认定 accuracy 有效。
            if not logits_finite:
                diagnostics['eval_predictions_valid'] = False
                diagnostics['eval_nonfinite_prediction_batches'] += 1
            y_pred = log_probs.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.view_as(y_pred)).long().cpu().sum()

    test_loss /= float(dataset_size)
    accuracy = 100.00 * correct / dataset_size
    diagnostics['eval_loss_finite'] &= math.isfinite(test_loss)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, dataset_size, accuracy))
    if return_diagnostics:
        return accuracy.item(), test_loss, diagnostics
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
