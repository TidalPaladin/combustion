#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest


# from combustion.train import *


@pytest.mark.skip
class TestIgniteTrainFunc:
    @pytest.fixture
    def func(self, mock_model, optimizer, criterion):
        return IgniteTrainFunc(mock_model, optimizer, criterion)

    @pytest.fixture
    def engine(self, ignite, func):
        return ignite.engine.Engine(func)

    @pytest.mark.usefixtures("backward_pass")
    def test_sets_model_to_train_mode(self, mocker, func, engine, batch, mock_model):
        func(engine, batch)
        mock_model.train.assert_called_once()

    @pytest.fixture
    def backward_pass(self, torch, mocker, optimizer, criterion):
        parent = mocker.Mock()
        mocker.patch.object(optimizer, "zero_grad")
        mocker.patch.object(optimizer, "step")
        mocker.patch.object(criterion, "forward")
        criterion.forward.return_value = mocker.Mock(spec_set=torch.Tensor, name="loss")
        parent.attach_mock(optimizer.zero_grad, "zero_grad")
        parent.attach_mock(optimizer.step, "step")
        parent.attach_mock(criterion.forward, "criterion")
        parent.attach_mock(criterion.forward.return_value, "loss")
        return parent

    def test_backward_pass(self, mocker, func, engine, backward_pass, mock_model, batch):
        func(engine, batch)
        criterion_input = mock_model.return_value.squeeze(-3)
        criterion_target = batch.labels
        print(criterion_input.shape)
        print(criterion_target.shape)

        # order is zero grads, forward, loss, backward, opt.step(), returns loss.item()
        assert backward_pass.method_calls[0] == mocker.call.zero_grad()
        assert backward_pass.method_calls[2] == mocker.call.loss.backward()
        assert backward_pass.method_calls[3] == mocker.call.step()
        assert backward_pass.method_calls[4] == mocker.call.loss.item()

    def test_returns_loss(self, mocker, func, engine, backward_pass, mock_model, batch):
        result = func(engine, batch)
        assert result == backward_pass.loss.item.return_value

    def test_call_no_mocks(self, mocker, func, engine, batch):
        loss = func(engine, batch)
        assert loss >= 0


@pytest.mark.skip
class TestIgniteEvalFunc:
    @pytest.fixture
    def func(self, mock_model):
        return IgniteEvalFunc(mock_model)

    @pytest.fixture
    def engine(self, ignite, func):
        return ignite.engine.Engine(func)

    def test_sets_model_to_eval_mode(self, mocker, func, engine, batch, mock_model):
        func(engine, batch)
        mock_model.eval.assert_called_once()

    def test_returns_predictions(self, torch, mocker, func, engine, mock_model, batch):
        result = func(engine, batch)
        assert torch.allclose(result[0], mock_model.return_value.squeeze(-3))
        assert torch.allclose(result[1], batch.labels)

    def test_visualize(self, torch, ignite, mocker, mock_model, batch, tmpdir):
        trainer = ignite.engine.Engine(lambda x, y: None)
        func = IgniteEvalFunc(mock_model, results_path=tmpdir, trainer=trainer)
        engine = ignite.engine.Engine(func)
        engine.state = ignite.engine.State()
        trainer.state = ignite.engine.State()
        func(engine, batch)
        subdir = "epoch_%d" % trainer.state.epoch
        filepath = "output_%d.png" % trainer.state.iteration
        assert subdir in os.listdir(tmpdir), "should produce epoch_x subdir"
        assert filepath in os.listdir(os.path.join(tmpdir, subdir)), "should produce epoch_x/output_y.png"
