# Copyright (c) 2026 Guy's and St Thomas' NHS Foundation Trust & King's College London
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from unittest.mock import MagicMock, patch

from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from nvflare.apis.dxo import DataKind
from nvflare.app_common.app_constant import AppConstants

from flip.nvflare.executors.monai_fl_adapter import RUN_MONAI_FL_EVALUATOR, RUN_MONAI_FL_TRAINER, RUN_MONAI_FL_VALIDATOR


def _make_fl_ctx():
    fl_ctx = MagicMock()
    fl_ctx.get_peer_context.return_value = None
    fl_ctx.get_job_id.return_value = "test_job"
    fl_ctx.get_identity_name.return_value = "test_client"
    return fl_ctx


def _make_shareable_with_weights(weights=None, current_round=0):
    """Build a MagicMock Shareable that looks like it carries WEIGHTS."""
    from nvflare.apis.dxo import DXO

    if weights is None:
        weights = {"layer.weight": [1.0, 2.0]}
    dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
    shareable = dxo.to_shareable()
    shareable.set_header(AppConstants.CURRENT_ROUND, current_round)
    return shareable


# ---------------------------------------------------------------------------
# RUN_MONAI_FL_TRAINER
# ---------------------------------------------------------------------------


class TestRunMonaiFlTrainer:
    def test_init_defaults(self):
        adapter = RUN_MONAI_FL_TRAINER()
        assert adapter._train_task_name == AppConstants.TASK_TRAIN
        assert adapter._submit_model_task_name == AppConstants.TASK_SUBMIT_MODEL
        assert adapter._project_id == ""
        assert adapter._query == ""
        assert adapter._algo is None

    def test_init_custom_values(self):
        adapter = RUN_MONAI_FL_TRAINER(
            train_task_name="train_ae",
            submit_model_task_name="submit_model",
            project_id="proj-1",
            query="SELECT * FROM t",
        )
        assert adapter._train_task_name == "train_ae"
        assert adapter._project_id == "proj-1"

    def test_execute_train_task_calls_lifecycle(self):
        """train task drives initialize → train → get_weights pipeline."""
        adapter = RUN_MONAI_FL_TRAINER(project_id="p", query="q")

        mock_algo = MagicMock()
        weight_diff = {"layer.weight": [0.1, 0.2]}
        mock_algo.get_weights.return_value = ExchangeObject(
            weights=weight_diff,
            weight_type=WeightType.WEIGHT_DIFF,
            statistics={"num_steps": 10},
        )
        mock_trainer_class = MagicMock(return_value=mock_algo)

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        abort_signal.triggered = False
        shareable = _make_shareable_with_weights(current_round=1)

        with patch.dict("sys.modules", {"trainer": MagicMock(FLIP_TRAINER=mock_trainer_class)}):
            result = adapter.execute(AppConstants.TASK_TRAIN, shareable, fl_ctx, abort_signal)

        mock_algo.initialize.assert_called_once()
        mock_algo.train.assert_called_once()
        mock_algo.get_weights.assert_called_once()
        # Result should be a Shareable (not make_reply error)
        assert result is not None

    def test_execute_submit_model_calls_get_weights_with_full_type(self):
        """submit_model task requests full weights (WeightType.WEIGHTS)."""
        adapter = RUN_MONAI_FL_TRAINER()

        mock_algo = MagicMock()
        full_weights = {"layer.weight": [1.0, 2.0]}
        mock_algo.get_weights.return_value = ExchangeObject(
            weights=full_weights,
            weight_type=WeightType.WEIGHTS,
        )
        adapter._algo = mock_algo  # pre-initialize

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        shareable = _make_shareable_with_weights()

        adapter.execute(AppConstants.TASK_SUBMIT_MODEL, shareable, fl_ctx, abort_signal)

        call_extra = mock_algo.get_weights.call_args.kwargs.get("extra") or mock_algo.get_weights.call_args[1].get(
            "extra"
        )
        assert call_extra is not None
        assert call_extra.get("weight_type") == WeightType.WEIGHTS

    def test_execute_unknown_task_returns_task_unknown(self):
        adapter = RUN_MONAI_FL_TRAINER()

        mock_algo = MagicMock()
        adapter._algo = mock_algo

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        shareable = _make_shareable_with_weights()

        result = adapter.execute("unknown_task", shareable, fl_ctx, abort_signal)
        from nvflare.apis.fl_constant import ReturnCode

        assert result.get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_execute_reuses_initialized_algo(self):
        """The algo is initialized only once across multiple train calls."""
        adapter = RUN_MONAI_FL_TRAINER()

        mock_algo = MagicMock()
        mock_algo.get_weights.return_value = ExchangeObject(weights={}, weight_type=WeightType.WEIGHT_DIFF)
        adapter._algo = mock_algo  # already initialized

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        abort_signal.triggered = False

        shareable = _make_shareable_with_weights()
        adapter.execute(AppConstants.TASK_TRAIN, shareable, fl_ctx, abort_signal)
        adapter.execute(AppConstants.TASK_TRAIN, shareable, fl_ctx, abort_signal)

        # initialize should NOT be called again
        mock_algo.initialize.assert_not_called()


# ---------------------------------------------------------------------------
# RUN_MONAI_FL_VALIDATOR
# ---------------------------------------------------------------------------


class TestRunMonaiFlValidator:
    def test_init_defaults(self):
        adapter = RUN_MONAI_FL_VALIDATOR()
        assert adapter._validate_task_name == AppConstants.TASK_VALIDATION
        assert adapter._algo is None

    def test_execute_validate_task_calls_evaluate(self):
        adapter = RUN_MONAI_FL_VALIDATOR(project_id="p", query="q")

        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = ExchangeObject(metrics={"val_acc": 0.85})
        mock_validator_class = MagicMock(return_value=mock_algo)

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        abort_signal.triggered = False
        shareable = _make_shareable_with_weights()

        with patch.dict("sys.modules", {"validator": MagicMock(FLIP_VALIDATOR=mock_validator_class)}):
            result = adapter.execute(AppConstants.TASK_VALIDATION, shareable, fl_ctx, abort_signal)

        mock_algo.initialize.assert_called_once()
        mock_algo.evaluate.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# RUN_MONAI_FL_EVALUATOR
# ---------------------------------------------------------------------------


class TestRunMonaiFlEvaluator:
    def test_init_defaults(self):
        adapter = RUN_MONAI_FL_EVALUATOR()
        assert adapter._evaluate_task_name == "evaluation"
        assert adapter._algo is None

    def test_execute_evaluation_task_calls_evaluate(self):
        adapter = RUN_MONAI_FL_EVALUATOR(project_id="p", query="q")

        mock_algo = MagicMock()
        mock_algo.evaluate.return_value = ExchangeObject(metrics={"model_a": {"spleen": {"mean_dice": 0.9}}})
        mock_evaluator_class = MagicMock(return_value=mock_algo)

        fl_ctx = _make_fl_ctx()
        abort_signal = MagicMock()
        abort_signal.triggered = False
        shareable = _make_shareable_with_weights()

        with patch.dict("sys.modules", {"evaluator": MagicMock(FLIP_EVALUATOR=mock_evaluator_class)}):
            result = adapter.execute("evaluation", shareable, fl_ctx, abort_signal)

        mock_algo.initialize.assert_called_once()
        mock_algo.evaluate.assert_called_once()
        assert result is not None
