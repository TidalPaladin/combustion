#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def cuda_available():
    r"""Checks if CUDA is available and device is ready"""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if isinstance(capability, tuple):
        capability = f"sm_{''.join(str(x) for x in capability)}"

    if capability not in arch_list:
        return False

    return True
