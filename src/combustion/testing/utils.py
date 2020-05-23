#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import torch


cuda_or_skip = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
