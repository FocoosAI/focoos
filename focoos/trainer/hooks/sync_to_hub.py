import os
import sys
from datetime import datetime
from typing import List, Optional

from focoos.hub.focoos_hub import FocoosHUB
from focoos.ports import ArtifactName, ModelInfo, ModelStatus, StatusTransition
from focoos.trainer.hooks.base import HookBase
from focoos.utils.logger import get_logger

logger = get_logger("SyncToHubHook")


class SyncToHubHook(HookBase):
    def __init__(
        self,
        hub: FocoosHUB,
        model_info: ModelInfo,
        output_dir: str,
        sync_period: int = 100,
        eval_period: int = 100,
    ):
        self.hub = hub
        self.model_info = model_info
        self.output_dir = output_dir
        self.sync_period = sync_period
        self.eval_period = eval_period

    @property
    def iteration(self):
        try:
            _iter = self.trainer.iter
        except Exception:
            _iter = 1
        return _iter

    def before_train(self):
        """
        Called before the first iteration.
        """
        self._sync_train_job()

    def after_step(self):
        if (self.iteration % self.sync_period == 0) and self.iteration > 0:
            self._sync_train_job()

        elif (self.iteration % (self.eval_period + 3) == 0) and self.iteration > 0:
            self._sync_train_job(upload_artifacts=[ArtifactName.WEIGHTS])

    def after_train(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is not None:
            logger.error(
                f"Exception during training, status set to TRAINING_ERROR: {str(exc_type.__name__)} {str(exc_value)}"
            )
            status = ModelStatus.TRAINING_ERROR
            self.model_info.status = status
            if self.model_info.training_info is not None:
                self.model_info.training_info.main_status = status
                self.model_info.training_info.failure_reason = str(exc_value)
                self.model_info.training_info.end_time = datetime.now().isoformat()
                if self.model_info.training_info.status_transitions is None:
                    self.model_info.training_info.status_transitions = []
                self.model_info.training_info.status_transitions.append(
                    StatusTransition(
                        status=status,
                        timestamp=datetime.now().isoformat(),
                        detail=f"{str(exc_type.__name__)}:  {str(exc_value)}",
                    )
                )

        self.model_info.dump_json(os.path.join(self.output_dir, ArtifactName.INFO))
        self._sync_train_job(
            upload_artifacts=[
                ArtifactName.WEIGHTS,
                ArtifactName.LOGS,
                ArtifactName.PT,
                ArtifactName.ONNX,
                ArtifactName.INFO,
                ArtifactName.METRICS,
            ]
        )

    def _sync_train_job(self, upload_artifacts: Optional[List[ArtifactName]] = None):
        try:
            self.hub.sync_training_job(self.output_dir, upload_artifacts)
            # logger.debug(f"Sync: {self.iteration} {self.model_info.name} ref: {self.model_info.ref}")
        except Exception as e:
            logger.error(f"[sync_train_job] failed to sync train job: {str(e)}")
