#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest

from combustion.train import *


class TestGetOptimizer:
    @pytest.mark.parametrize("opt", ["adam", "rmsprop"])
    def test_get_valid_optimizer(self, mocker, mock_args, opt, torch):
        model = torch.nn.Linear(10, 10)
        spy = mocker.spy(model, "parameters")
        mock_args.optim = opt
        mock_args.lr = 0.001
        mock_args.beta1 = 0.9
        mock_args.beta2 = 0.99
        mock_args.epsilon = 1e-6
        mock_args.momentum = 0.5

        result = get_optim_from_args(mock_args, model)
        if opt == "adam":
            assert isinstance(result, torch.optim.Adam)
            assert result.defaults["lr"] == mock_args.lr
            assert result.defaults["eps"] == mock_args.epsilon
            assert result.defaults["betas"] == (mock_args.beta1, mock_args.beta2)
            spy.assert_called()
        elif opt == "rmsprop":
            assert isinstance(result, torch.optim.RMSprop)
            assert result.defaults["lr"] == mock_args.lr
            assert result.defaults["eps"] == mock_args.epsilon
            assert result.defaults["momentum"] == mock_args.momentum
            spy.assert_called()

    def test_get_invalid_optimizer(self, mocker, mock_args, torch):
        model = torch.nn.Linear(10, 10)
        mock_args.optim = "foobar"
        with pytest.raises(ValueError):
            get_optim_from_args(mock_args, model)


@pytest.mark.skip
class TestGetCheckpointHandler:
    @pytest.fixture(autouse=True)
    def checkpoint_dir(self, mock_args, tmp_path):
        path = os.path.join(tmp_path, "checkpoint")
        mock_args.model_path = path
        return path

    @pytest.fixture(autouse=True)
    def prefix(self, mock_args):
        prefix = "foo"
        mock_args.checkpoint_prefix = prefix
        return prefix

    @pytest.fixture(
        autouse=True,
        params=[pytest.param(1, id="saved=1"), pytest.param(2, id="saved=2"), pytest.param(5, id="saved=5"),],
    )
    def n_saved(self, request, mock_args):
        n_saved = request.param
        mock_args.n_saved = n_saved
        return n_saved

    @pytest.fixture
    def to_save(self, trainer, model, optimizer):
        to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
        return to_save

    @pytest.fixture(params=["step", "epoch"])
    def event_type(self, request):
        return request.param

    @pytest.fixture
    def event(self, event_type, ignite, mocker, request, mock_args):
        if event_type == "step":
            steps = 4
            mock_args.checkpoint_steps = steps
            mock_args.checkpoint_epochs = None
            return ignite.engine.Events.ITERATION_COMPLETED(every=steps)
        elif event_type == "epoch":
            epochs = 1
            mock_args.checkpoint_steps = None
            mock_args.checkpoint_epochs = epochs
            return ignite.engine.Events.EPOCH_COMPLETED(every=epochs)
        raise pytest.UsageError("unexpected param")

    def test_returns_added_handler(self, trainer, mock_args, to_save, mocker, event, ignite):
        spy = mocker.spy(trainer, "add_event_handler")
        handler = attach_checkpoint_handler(mock_args, trainer, to_save)
        assert isinstance(handler, ignite.handlers.ModelCheckpoint)

    def test_return_none_if_interval_not_set(self, trainer, mock_args, to_save, mocker):
        mock_args.checkpoint_steps = None
        mock_args.checkpoint_epochs = None
        spy = mocker.spy(trainer, "add_event_handler")
        handler = attach_checkpoint_handler(mock_args, trainer, to_save)
        assert handler is None, "returns None if no handler added"

    def test_no_handler_added_if_step_interval_not_set(self, trainer, mock_args, to_save, mocker):
        mock_args.checkpoint_steps = None
        mock_args.checkpoint_epochs = None
        spy = mocker.spy(trainer, "add_event_handler")
        attach_checkpoint_handler(mock_args, trainer, to_save)
        spy.assert_not_called()

    @pytest.mark.usefixtures("event")
    @pytest.mark.parametrize(
        "epochs", [pytest.param(1, id="epochs=1"), pytest.param(2, id="epochs=2"), pytest.param(3, id="epochs=3"),]
    )
    def test_num_checkpoints_saved(
        self, event_type, trainer, mock_args, to_save, data, checkpoint_dir, n_saved, epochs
    ):
        attach_checkpoint_handler(mock_args, trainer, to_save)
        state = trainer.run(data, epochs)
        if event_type == "step":
            num_expected = min(state.iteration // mock_args.checkpoint_steps, n_saved)
        else:
            num_expected = min(state.epoch, n_saved)
        # check n_saved if failing with only 1 output checkpoint
        assert len(os.listdir(checkpoint_dir)) == num_expected

    @pytest.mark.usefixtures("event")
    def test_saved_checkpoint_prefix_from_args(self, trainer, mock_args, to_save, data, checkpoint_dir, prefix):
        attach_checkpoint_handler(mock_args, trainer, to_save)
        state = trainer.run(data, 1)
        saved = os.listdir(checkpoint_dir)
        assert saved and all([prefix in c for c in saved])

    @pytest.mark.usefixtures("event")
    def test_saved_with_pth_file_ext(self, trainer, mock_args, to_save, data, checkpoint_dir):
        suffix = ".pth"
        attach_checkpoint_handler(mock_args, trainer, to_save)
        state = trainer.run(data, 1)
        saved = os.listdir(checkpoint_dir)
        assert saved and all([suffix in c for c in saved])

    @pytest.mark.usefixtures("event")
    def test_to_save_dict_consumed(self, mocker, trainer, mock_args, to_save, data):
        spy = mocker.spy(trainer, "add_event_handler")
        attach_checkpoint_handler(mock_args, trainer, to_save)
        spy.assert_called_once()
        args = spy.call_args[0]
        # to_save passed to add_event_handler
        assert to_save in args


@pytest.mark.skip
class TestLoadCheckpoint:
    @pytest.fixture(autouse=True)
    def checkpoint_dir(self, mock_args, tmp_path):
        path = os.path.join(tmp_path, "checkpoint")
        mock_args.model_path = path
        os.makedirs(path)
        return path

    @pytest.fixture(autouse=True)
    def lr_scheduler(self, torch, optimizer, ignite):
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, 0.001)
        scheduler = ignite.contrib.handlers.param_scheduler.LRScheduler(schedule)
        return scheduler

    @pytest.fixture
    def to_save(self, trainer, model, optimizer, lr_scheduler):
        trainer_state = {"seed": 0, "epoch": 3, "max_epochs": 100, "epoch_length": 10}
        trainer.load_state_dict(trainer_state)
        to_save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "trainer": trainer_state,
        }
        return to_save

    @pytest.fixture
    def checkpoint(self, torch, checkpoint_dir, to_save, mock_args):
        filename = os.path.join(checkpoint_dir, "foo_checkpoint_10.pth")
        mock_args.load_model = filename
        torch.save(to_save, filename)
        return filename

    @pytest.fixture(params=["model", "optimizer", "lr_scheduler", "trainer"])
    def state_dict_target(self, request, model, optimizer, lr_scheduler, trainer):
        if request.param == "model":
            target = model
        elif request.param == "optimizer":
            target = optimizer
        elif request.param == "lr_scheduler":
            target = lr_scheduler
        elif request.param == "trainer":
            target = trainer
        else:
            raise pytest.UsageError("unknown param")
        return request.param, target

    def test_load_model_arg_target(
        self, mocker, mock_args, model, optimizer, lr_scheduler, trainer, checkpoint, torch
    ):
        torch_load_spy = mocker.spy(torch, "load")
        restore_checkpoint(mock_args, model, optimizer, lr_scheduler, trainer)
        torch_load_spy.assert_called_once_with(mock_args.load_model)

    def test_loads_state_dict(
        self, mocker, mock_args, model, optimizer, lr_scheduler, trainer, checkpoint, state_dict_target, torch
    ):
        name, obj = state_dict_target
        load_dict_spy = mocker.spy(obj, "load_state_dict")
        torch_load_spy = mocker.spy(torch, "load")
        restore_checkpoint(mock_args, model, optimizer, lr_scheduler, trainer)
        loaded_state_dict = torch_load_spy.spy_return[name]
        load_dict_spy.assert_called_once_with(loaded_state_dict)

    def test_returns_path_state_dict_tuple(
        self, mock_args, model, optimizer, lr_scheduler, trainer, checkpoint, to_save
    ):
        path, state_dict = restore_checkpoint(mock_args, model, optimizer, lr_scheduler, trainer)
        assert path == checkpoint
        # best we can do here is check keys to avoid ambiguous tensor comparisons
        assert state_dict.keys() == to_save.keys()


@pytest.mark.skip
class TestProgbarHandler:
    @pytest.fixture(params=[pytest.param(True, id="progbar=True"), pytest.param(False, id="progbar=False")])
    def progbar(self, request, mock_args):
        mock_args.progbar = request.param
        return mock_args

    @pytest.fixture(params=[pytest.param(True, id="gpuinfo=True"), pytest.param(False, id="gpuinfo=False")])
    def gpuinfo(self, mocker, request, mock_args, torch):
        if request.param:
            if not torch.cuda.is_available():
                pytest.skip("test requires GPU available")
            pytest.importorskip("pynvml", reason="test requires pynvml")
        mock_args.gpuinfo = request.param
        mock_args.gpu_format = "gpu:0 mem(%)"
        return mock_args

    @pytest.fixture
    def args(self, mock_args, progbar, gpuinfo):
        mock_args.progbar_format = "foobar"
        return mock_args

    @pytest.fixture
    def spy_progbar_attach(self, mocker, ignite):
        progbar = ignite.contrib.handlers.tqdm_logger.ProgressBar
        spy = mocker.spy(progbar, "attach")
        return spy

    @pytest.fixture
    def spy_gpuinfo_attach(self, mocker, ignite):
        info = ignite.contrib.metrics.GpuInfo
        spy = mocker.spy(info, "attach")
        return spy

    def test_gpu_info_attached_to_trainer(self, args, trainer, spy_gpuinfo_attach):
        attach_progbar_handler(args, trainer)
        if args.gpuinfo and args.progbar:
            spy_gpuinfo_attach.assert_called_once()
        else:
            spy_gpuinfo_attach.assert_not_called()

    @pytest.mark.parametrize("lr", [pytest.param(None, id="lr=None"), pytest.param("one_cycle", id="lr=one_cycle"),])
    def test_progbar_attached_to_trainer(self, mocker, args, trainer, spy_progbar_attach, lr):
        args.lr_decay = lr
        attach_progbar_handler(args, trainer)
        if args.progbar:
            spy_progbar_attach.assert_called_once()
            assert trainer in spy_progbar_attach.call_args[0]
            metrics = ["loss"]
            if args.gpuinfo:
                metrics.append(args.gpu_format)
            if lr is not None:
                metrics.append("lr")
            assert metrics in spy_progbar_attach.call_args[0]
        else:
            spy_progbar_attach.assert_not_called()

    def test_returns_progbar_handler_if_attached(self, args, trainer, ignite):
        returned = attach_progbar_handler(args, trainer)
        if args.progbar:
            assert isinstance(returned, ignite.contrib.handlers.tqdm_logger.ProgressBar)
        else:
            assert returned is None

    def test_progbar_format_arg(self, mocker, args, trainer, ignite):
        mock = ignite.contrib.handlers.tqdm_logger.ProgressBar
        spy = mocker.spy(mock, "__init__")
        attach_progbar_handler(args, trainer)
        if args.progbar:
            spy.assert_called_once()
            assert "bar_format" in spy.call_args[1]
            assert spy.call_args[1]["bar_format"] == args.progbar_format
        else:
            spy.assert_not_called()

    def test_default_progbar_format_arg(self, mocker, args, trainer, ignite):
        args.progbar_format = None
        mock = ignite.contrib.handlers.tqdm_logger.ProgressBar
        spy = mocker.spy(mock, "__init__")
        attach_progbar_handler(args, trainer)
        if args.progbar:
            spy.assert_called_once()
            assert "bar_format" not in spy.call_args[1]
        else:
            spy.assert_not_called()


@pytest.mark.skip
class TestLrScheduleHandler:
    @pytest.fixture
    def args(self, mock_args):
        mock_args.max_lr = 0.01
        mock_args.warmup_percent = 0.2
        mock_args.div_factor = 1e3
        mock_args.final_div_factor = 1e6
        mock_args.steps = 10
        mock_args.epochs = 2
        mock_args.steps_per_epoch = 12
        return mock_args

    @pytest.fixture(
        params=[pytest.param(None, id="lr_decay=None"), pytest.param("one_cycle", id="lr_decay=one_cycle")]
    )
    def lr_decay(self, request, args):
        args.lr_decay = request.param
        return request.param

    def test_return_val(self, args, trainer, optimizer, lr_decay, ignite, torch):
        handler = attach_lr_schedule_handler(args, trainer, optimizer)
        handler_type = ignite.contrib.handlers.param_scheduler.LRScheduler
        scheduler_type = torch.optim.lr_scheduler.OneCycleLR
        if lr_decay == "one_cycle":
            assert isinstance(handler, handler_type)
            assert isinstance(handler.lr_scheduler, scheduler_type)
        else:
            assert handler is None

    def test_attached_to_trainer(self, mocker, args, trainer, optimizer, lr_decay, ignite):
        spy = mocker.spy(trainer, "add_event_handler")
        attached_type = ignite.contrib.handlers.param_scheduler.LRScheduler
        attach_lr_schedule_handler(args, trainer, optimizer)
        if lr_decay is not None:
            spy.assert_called_once()
            assert isinstance(spy.call_args[0][1], attached_type)
        else:
            spy.assert_not_called()

    def test_one_cycle_args(self, mocker, args, trainer, optimizer, torch):
        args.lr_decay = "one_cycle"
        spy = mocker.spy(torch.optim.lr_scheduler.OneCycleLR, "__init__")
        attach_lr_schedule_handler(args, trainer, optimizer)
        spy.assert_called_once()
        assert optimizer in spy.call_args[0]

        self.check_call_kwarg(spy, "max_lr", args.max_lr)
        self.check_call_kwarg(spy, "pct_start", args.warmup_percent)
        self.check_call_kwarg(spy, "div_factor", args.div_factor)
        self.check_call_kwarg(spy, "final_div_factor", args.final_div_factor)
        self.check_call_kwarg(spy, "total_steps", args.steps)
        self.check_call_kwarg(spy, "epochs", args.epochs)
        self.check_call_kwarg(spy, "steps_per_epoch", args.steps_per_epoch)

    def check_call_kwarg(self, mock, key, value):
        __tracebackhide__ = True
        if key not in mock.call_args[1].keys():
            pytest.fail("missing kwarg: {}".format(key))
        if not mock.call_args[1][key] == value:
            pytest.fail("expected {}={}, got {}={}".format(key, value, key, mock.call_args[1][key]))
