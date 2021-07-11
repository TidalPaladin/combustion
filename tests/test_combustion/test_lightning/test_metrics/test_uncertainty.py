#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

from pathlib import Path
from combustion.lightning.metrics import ECE, UCE, ErrorAtUncertainty


class TestECE:

    # https://jamesmccaffrey.wordpress.com/2021/01/06/calculating-expected-calibration-error-for-binary-classification/
    def test_binary_ece_real(self, cuda):
        probs = torch.tensor([0.61, 0.39, 0.31, 0.76, 0.22, 0.59, 0.92, 0.83, 0.57, 0.41])
        true = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
        logits = probs.logit()

        metric = ECE(num_bins=3, from_logits=True)
        if cuda:
            logits = logits.cuda()
            metric = metric.cuda()
            true = true.cuda()

        ece = metric(logits, true)  # type: ignore
        assert ece == 0.241

    # https://jamesmccaffrey.wordpress.com/2021/01/22/how-to-calculate-expected-calibration-error-for-multi-class-classification/
    def test_categorical_ece_real(self, cuda):
        probs = torch.tensor(
            [
                [0.25, 0.2, 0.22, 0.18, 0.15],
                [0.16, 0.06, 0.50, 0.07, 0.21],
                [0.06, 0.03, 0.8, 0.07, 0.04],
                [0.02, 0.03, 0.01, 0.04, 0.9],
                [0.4, 0.15, 0.16, 0.14, 0.15],
                [0.15, 0.28, 0.18, 0.17, 0.22],
                [0.07, 0.8, 0.03, 0.06, 0.04],
                [0.1, 0.05, 0.03, 0.75, 0.07],
                [0.25, 0.22, 0.05, 0.3, 0.18],
                [0.12, 0.09, 0.02, 0.17, 0.6],
            ]
        )
        true = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])

        metric = ECE(num_bins=3, from_logits=False)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            true = true.cuda()

        ece = metric(probs, true)  # type: ignore
        assert torch.allclose(ece, ece.new_tensor(0.192))

    def test_categorical_ece_classwise(self, cuda):
        probs = torch.tensor(
            [
                [0.25, 0.2, 0.22, 0.18, 0.15],
                [0.16, 0.06, 0.50, 0.07, 0.21],
                [0.06, 0.03, 0.8, 0.07, 0.04],
                [0.02, 0.03, 0.01, 0.04, 0.9],
                [0.4, 0.15, 0.16, 0.14, 0.15],
                [0.15, 0.28, 0.18, 0.17, 0.22],
                [0.07, 0.8, 0.03, 0.06, 0.04],
                [0.1, 0.05, 0.03, 0.75, 0.07],
                [0.25, 0.22, 0.05, 0.3, 0.18],
                [0.12, 0.09, 0.02, 0.17, 0.6],
            ]
        )
        true = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])
        N = 5

        metric = ECE(num_bins=3, from_logits=False, classwise=True, num_classes=N)
        metric_base = ECE(num_bins=3, from_logits=False, classwise=False, num_classes=N)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            metric_base = metric_base.cuda()
            true = true.cuda()

        out = probs.new_zeros(1)
        for i in range(N):
            keep = true == i
            _probs = probs[keep]
            _true = true[keep]
            out.add_(metric_base(_probs, _true))  # type: ignore
        expected = out / N

        ece = metric(probs, true)  # type: ignore
        assert torch.allclose(ece, expected)


class TestUCE:
    def test_binary(self, cuda):
        probs = torch.tensor([0.61, 0.39, 0.31, 0.76, 0.22, 0.59, 0.92, 0.83, 0.57, 0.41])
        true = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
        logits = probs.logit()

        metric = UCE(num_bins=3, from_logits=True)
        if cuda:
            logits = logits.cuda()
            metric = metric.cuda()
            true = true.cuda()

        uce = metric(logits, true)  # type: ignore
        # TODO check this
        assert torch.allclose(uce, torch.tensor(0.4248734))

    def test_categorical(self, cuda):
        probs = torch.tensor(
            [
                [0.25, 0.2, 0.22, 0.18, 0.15],
                [0.16, 0.06, 0.50, 0.07, 0.21],
                [0.06, 0.03, 0.8, 0.07, 0.04],
                [0.02, 0.03, 0.01, 0.04, 0.9],
                [0.4, 0.15, 0.16, 0.14, 0.15],
                [0.15, 0.28, 0.18, 0.17, 0.22],
                [0.07, 0.8, 0.03, 0.06, 0.04],
                [0.1, 0.05, 0.03, 0.75, 0.07],
                [0.25, 0.22, 0.05, 0.3, 0.18],
                [0.12, 0.09, 0.02, 0.17, 0.6],
            ]
        )
        true = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])

        metric = UCE(num_bins=3, from_logits=False)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            true = true.cuda()

        uce = metric(probs, true)  # type: ignore
        # TODO check this
        assert torch.allclose(uce, uce.new_tensor(0.3168278))

    def test_categorical_classwise(self, cuda):
        probs = torch.tensor(
            [
                [0.25, 0.2, 0.22, 0.18, 0.15],
                [0.16, 0.06, 0.50, 0.07, 0.21],
                [0.06, 0.03, 0.8, 0.07, 0.04],
                [0.02, 0.03, 0.01, 0.04, 0.9],
                [0.4, 0.15, 0.16, 0.14, 0.15],
                [0.15, 0.28, 0.18, 0.17, 0.22],
                [0.07, 0.8, 0.03, 0.06, 0.04],
                [0.1, 0.05, 0.03, 0.75, 0.07],
                [0.25, 0.22, 0.05, 0.3, 0.18],
                [0.12, 0.09, 0.02, 0.17, 0.6],
            ]
        )
        true = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])
        N = 5

        metric = UCE(num_bins=3, from_logits=False, classwise=True, num_classes=N)
        metric_base = UCE(num_bins=3, from_logits=False, classwise=False, num_classes=N)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            metric_base = metric_base.cuda()
            true = true.cuda()

        out = probs.new_zeros(1)
        for i in range(N):
            keep = true == i
            _probs = probs[keep]
            _true = true[keep]
            out.add_(metric_base(_probs, _true))  # type: ignore
        expected = out / N

        uce = metric(probs, true)  # type: ignore
        # TODO check this
        assert torch.allclose(uce, expected)

    def test_plot(self, cuda):
        probs = torch.tensor(
            [
                [0.25, 0.2, 0.22, 0.18, 0.15],
                [0.16, 0.06, 0.50, 0.07, 0.21],
                [0.06, 0.03, 0.8, 0.07, 0.04],
                [0.02, 0.03, 0.01, 0.04, 0.9],
                [0.4, 0.15, 0.16, 0.14, 0.15],
                [0.15, 0.28, 0.18, 0.17, 0.22],
                [0.07, 0.8, 0.03, 0.06, 0.04],
                [0.1, 0.05, 0.03, 0.75, 0.07],
                [0.25, 0.22, 0.05, 0.3, 0.18],
                [0.12, 0.09, 0.02, 0.17, 0.6],
            ]
        )
        true = torch.tensor([0, 2, 3, 4, 2, 0, 1, 3, 3, 2])
        N = 5

        metric = ErrorAtUncertainty(num_bins=5, from_logits=False, classwise=True, num_classes=N)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            true = true.cuda()

        entropy, err, has_items = metric(probs, true)  # type: ignore
        fig = metric.plot(entropy[has_items], err[has_items])
        dest = Path("/home/tidal/test_imgs")
        if dest.is_dir():
            dest = Path(dest, "TestUCE")
            fig.savefig(dest)
            plt.close(fig)

    def test_plot2(self, cuda):
        probs = torch.randn(10000, 10).div(0.1)
        true = torch.randint(0, 10, (10000,))
        N = 10

        metric = ErrorAtUncertainty(num_bins=10, from_logits=True, classwise=False, num_classes=N)
        if cuda:
            probs = probs.cuda()
            metric = metric.cuda()
            true = true.cuda()

        entropy, err, has_items = metric(probs, true)  # type: ignore
        fig = metric.plot(entropy[has_items], err[has_items])
        dest = Path("/home/tidal/test_imgs")
        if dest.is_dir():
            dest = Path(dest, "TestUCE2")
            fig.savefig(dest)
            plt.close(fig)
