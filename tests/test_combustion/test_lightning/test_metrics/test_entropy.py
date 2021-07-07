#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from combustion.lightning.metrics import Entropy


class TestEntropy:
    def test_binary_entropy(self, cuda):
        logits = torch.tensor(
            [
                [2.0, 1.3, 0.0],
                [0.7, -1.3, -0.1],
            ]
        )
        metric = Entropy(dim=None)
        if cuda:
            logits = logits.cuda()
            metric = metric.cuda()

        p = logits.sigmoid()
        expected = (p.log() * p + (1 - p).log() * (1 - p)).sum().div_(logits.numel()).neg_()

        entropy = metric(logits)  # type: ignore
        assert torch.allclose(entropy, expected)

    def test_categorical_entropy(self, cuda):
        dim = 0
        logits = torch.tensor(
            [
                [2.0, 1.3, 0.2],
                [1.0, 1.0, 1.0],
                [0.7, -1.3, -0.1],
            ]
        )
        metric = Entropy(dim=dim)
        if cuda:
            logits = logits.cuda()
            metric = metric.cuda()

        N = logits.shape[dim]
        divisor = logits.numel() / N * logits.new_tensor(N).log_()
        p = logits.softmax(dim=dim)
        expected = (p.log() * p).sum(dim=dim).neg_().sum().div_(divisor)
        entropy = metric(logits)  # type: ignore
        assert torch.allclose(entropy, expected)

        if cuda:
            logits = logits.cuda()
