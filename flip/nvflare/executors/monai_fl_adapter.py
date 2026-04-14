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

"""
MONAI FL adapter Executors.

This module provides thin NVFLARE Executor wrappers that drive user-provided
``ClientAlgo`` (and ``MonaiAlgo``) subclasses via the MONAI FL lifecycle
interface, translating the NVFLARE ``Shareable``/``DXO`` protocol to MONAI
FL's ``ExchangeObject`` and back.

The three adapters mirror the existing ``RUN_TRAINER`` / ``RUN_VALIDATOR`` /
``RUN_EVALUATOR`` executors but require tutorial classes to implement the
``monai.fl.client.client_algo.ClientAlgo`` interface instead of the raw
``nvflare.apis.executor.Executor`` interface.

Classes:
    RUN_MONAI_FL_TRAINER:   handles ``train`` and ``submit_model`` tasks.
    RUN_MONAI_FL_VALIDATOR: handles ``validate`` tasks.
    RUN_MONAI_FL_EVALUATOR: handles ``evaluation`` tasks.
"""

from __future__ import annotations

from monai.fl.utils.constants import WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_traceback


def _shareable_to_exchange_object(shareable: Shareable) -> tuple[ExchangeObject, int]:
    """Extract weights from a Shareable and wrap in an ExchangeObject.

    Args:
        shareable: NVFLARE Shareable containing a DXO with ``DataKind.WEIGHTS``.

    Returns:
        Tuple of (ExchangeObject with weights, current round number).

    Raises:
        ValueError: if the DXO data kind is not ``WEIGHTS``.
    """
    dxo = from_shareable(shareable)
    if dxo.data_kind not in (DataKind.WEIGHTS, DataKind.WEIGHT_DIFF):
        raise ValueError(f"Expected DXO DataKind WEIGHTS, got {dxo.data_kind!r}")
    current_round = shareable.get_header(AppConstants.CURRENT_ROUND, 0) or 0
    return (
        ExchangeObject(
            weights=dxo.data,
            weight_type=WeightType.WEIGHTS,
            statistics={"current_round": current_round},
        ),
        current_round,
    )


def _exchange_object_to_shareable(exchange_obj: ExchangeObject) -> Shareable:
    """Pack an ExchangeObject's weights into a NVFLARE Shareable.

    Chooses ``DataKind.WEIGHT_DIFF`` when ``weight_type`` is
    ``WeightType.WEIGHT_DIFF``, otherwise ``DataKind.WEIGHTS``.

    Args:
        exchange_obj: ExchangeObject returned by ``algo.get_weights()``.

    Returns:
        NVFLARE Shareable ready to be returned from an executor.
    """
    data_kind = DataKind.WEIGHT_DIFF if exchange_obj.weight_type == WeightType.WEIGHT_DIFF else DataKind.WEIGHTS
    n_steps = 1
    if exchange_obj.statistics:
        n_steps = exchange_obj.statistics.get("num_steps", n_steps)
    dxo = DXO(
        data_kind=data_kind,
        data=exchange_obj.weights,
        meta={MetaKey.NUM_STEPS_CURRENT_ROUND: n_steps},
    )
    return dxo.to_shareable()


class RUN_MONAI_FL_TRAINER(Executor):
    """NVFLARE adapter executor that drives a ``ClientAlgo``-compatible ``FLIP_TRAINER``.

    Translates the NVFLARE ``train`` / ``submit_model`` task protocol to the
    MONAI FL ``ClientAlgo`` lifecycle::

        initialize(extra) → train(data) → get_weights() → get_weights(full)

    The user-provided ``FLIP_TRAINER`` class is imported lazily from the job's
    ``trainer`` module (present in the NVFLARE Python path) when the first task
    arrives.  Initialization state is preserved across tasks.

    Args:
        train_task_name: NVFLARE task name for local training.
        submit_model_task_name: NVFLARE task name for model submission.
        project_id: FLIP project identifier passed to ``FLIP_TRAINER.__init__``.
        query: Cohort query string passed to ``FLIP_TRAINER.__init__``.
    """

    def __init__(
        self,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        project_id: str = "",
        query: str = "",
    ) -> None:
        super().__init__()
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._project_id = project_id
        self._query = query
        self._algo = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if self._algo is None:
                import trainer as trainer_module
                from trainer import FLIP_TRAINER

                trainer_file = getattr(trainer_module, "__file__", "unknown")
                self.log_info(fl_ctx, f"[RUN_MONAI_FL_TRAINER] Imported FLIP_TRAINER from: {trainer_file}")
                self._algo = FLIP_TRAINER(
                    project_id=self._project_id,
                    query=self._query,
                    train_task_name=self._train_task_name,
                )
                self._algo.initialize(extra={"fl_ctx": fl_ctx})

            if task_name == self._train_task_name:
                exchange_in, current_round = _shareable_to_exchange_object(shareable)
                self._algo.train(
                    data=exchange_in,
                    extra={
                        "fl_ctx": fl_ctx,
                        "abort_signal": abort_signal,
                        "current_round": current_round,
                    },
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                exchange_out = self._algo.get_weights(extra={"fl_ctx": fl_ctx, "weight_type": WeightType.WEIGHT_DIFF})
                return _exchange_object_to_shareable(exchange_out)

            elif task_name == self._submit_model_task_name:
                exchange_out = self._algo.get_weights(extra={"fl_ctx": fl_ctx, "weight_type": WeightType.WEIGHTS})
                return _exchange_object_to_shareable(exchange_out)

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception:
            self.log_info(fl_ctx, "An exception has been caught in RUN_MONAI_FL_TRAINER")
            self.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


class RUN_MONAI_FL_VALIDATOR(Executor):
    """NVFLARE adapter executor that drives a ``ClientAlgo``-compatible ``FLIP_VALIDATOR``.

    Translates the NVFLARE ``validate`` task protocol to the MONAI FL
    ``ClientAlgo`` lifecycle::

        initialize(extra) → evaluate(data) → ExchangeObject(metrics)

    The user-provided ``FLIP_VALIDATOR`` class is imported lazily from the
    job's ``validator`` module on the first task.

    Args:
        validate_task_name: NVFLARE task name for validation.
        project_id: FLIP project identifier passed to ``FLIP_VALIDATOR.__init__``.
        query: Cohort query string passed to ``FLIP_VALIDATOR.__init__``.
    """

    def __init__(
        self,
        validate_task_name: str = AppConstants.TASK_VALIDATION,
        project_id: str = "",
        query: str = "",
    ) -> None:
        super().__init__()
        self._validate_task_name = validate_task_name
        self._project_id = project_id
        self._query = query
        self._algo = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if self._algo is None:
                import validator as validator_module
                from validator import FLIP_VALIDATOR

                validator_file = getattr(validator_module, "__file__", "unknown")
                self.log_info(fl_ctx, f"[RUN_MONAI_FL_VALIDATOR] Imported FLIP_VALIDATOR from: {validator_file}")
                self._algo = FLIP_VALIDATOR(
                    project_id=self._project_id,
                    query=self._query,
                    validate_task_name=self._validate_task_name,
                )
                self._algo.initialize(extra={"fl_ctx": fl_ctx})

            if task_name == self._validate_task_name:
                exchange_in, _ = _shareable_to_exchange_object(shareable)
                exchange_out = self._algo.evaluate(
                    data=exchange_in,
                    extra={"fl_ctx": fl_ctx, "abort_signal": abort_signal},
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                dxo = DXO(data_kind=DataKind.METRICS, data=exchange_out.metrics or {})
                return dxo.to_shareable()

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception:
            self.log_info(fl_ctx, "An exception has been caught in RUN_MONAI_FL_VALIDATOR")
            self.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


class RUN_MONAI_FL_EVALUATOR(Executor):
    """NVFLARE adapter executor that drives a ``ClientAlgo``-compatible ``FLIP_EVALUATOR``.

    Translates the NVFLARE ``evaluation`` task protocol to the MONAI FL
    ``ClientAlgo`` lifecycle::

        initialize(extra) → evaluate(data) → ExchangeObject(metrics)

    The evaluator receives multi-model weights packed in the Shareable DXO.
    The ``ExchangeObject`` passed to ``evaluate()`` carries the full DXO data
    dict under ``weights`` so that tutorial evaluators can unpack per-model
    weights themselves.

    The user-provided ``FLIP_EVALUATOR`` class is imported lazily from the
    job's ``evaluator`` module on the first task.

    Args:
        evaluate_task_name: NVFLARE task name for evaluation.
        project_id: FLIP project identifier passed to ``FLIP_EVALUATOR.__init__``.
        query: Cohort query string passed to ``FLIP_EVALUATOR.__init__``.
    """

    def __init__(
        self,
        evaluate_task_name: str = "evaluation",
        project_id: str = "",
        query: str = "",
    ) -> None:
        super().__init__()
        self._evaluate_task_name = evaluate_task_name
        self._project_id = project_id
        self._query = query
        self._algo = None

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if self._algo is None:
                import evaluator as evaluator_module
                from evaluator import FLIP_EVALUATOR

                evaluator_file = getattr(evaluator_module, "__file__", "unknown")
                self.log_info(fl_ctx, f"[RUN_MONAI_FL_EVALUATOR] Imported FLIP_EVALUATOR from: {evaluator_file}")
                self._algo = FLIP_EVALUATOR(
                    project_id=self._project_id,
                    query=self._query,
                    evaluate_task_name=self._evaluate_task_name,
                )
                self._algo.initialize(extra={"fl_ctx": fl_ctx})

            if task_name == self._evaluate_task_name:
                # Pass the raw DXO data to the evaluator so it can unpack
                # per-model weights; the ExchangeObject.weights carries the
                # full DXO data dict.
                dxo = from_shareable(shareable)
                exchange_in = ExchangeObject(
                    weights=dxo.data,
                    weight_type=WeightType.WEIGHTS,
                )
                exchange_out = self._algo.evaluate(
                    data=exchange_in,
                    extra={"fl_ctx": fl_ctx, "abort_signal": abort_signal},
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                out_dxo = DXO(data_kind=DataKind.METRICS, data=exchange_out.metrics or {})
                return out_dxo.to_shareable()

            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)

        except Exception:
            self.log_info(fl_ctx, "An exception has been caught in RUN_MONAI_FL_EVALUATOR")
            self.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
