import os
import time
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from supervision import BoxAnnotator, Detections, LabelAnnotator, MaskAnnotator
from tqdm import tqdm

from focoos.ports import (
    DeploymentMode,
    FocoosDet,
    FocoosDetections,
    FocoosTask,
    Hyperparameters,
    LatencyMetrics,
    ModelMetadata,
    ModelStatus,
    OnnxEngineOpts,
    TrainInstance,
)
from focoos.runtime import ONNXRuntime
from focoos.utils.logger import get_logger
from focoos.utils.system import HttpClient
from focoos.utils.vision import (
    focoos_detections_to_supervision,
    image_loader,
    image_preprocess,
    scale_detections,
    sv_to_focoos_detections,
)


class FocoosModel:
    def __init__(self, model_ref: str, http_client: HttpClient):
        self.model_ref = model_ref
        self.logger = get_logger()
        self.http_client = http_client
        self.max_deploy_wait = 10
        self.metadata: ModelMetadata = self.get_info()
        self.runtime = None
        self.model_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "focoos", self.model_ref
        )
        self.logger.info(
            f"[RemoteModel]: ref: {self.model_ref} name: {self.metadata.name} description: {self.metadata.description} status: {self.metadata.status}"
        )
        self.label_annotator = LabelAnnotator(text_padding=10, border_radius=10)
        self.box_annotator = BoxAnnotator()
        self.mask_annotator = MaskAnnotator()

    def get_info(self) -> ModelMetadata:
        res = self.http_client.get(f"models/{self.model_ref}")
        if res.status_code == 200:
            return ModelMetadata(**res.json())
        else:
            self.logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}")

    def remote_train(
        self,
        dataset_ref: str,
        hyperparameters: Hyperparameters,
        anyma_version: str = "anyma-sagemaker-cu12-torch22-0111",
        instance_type: TrainInstance = TrainInstance.ML_G4DN_XLARGE,
        volume_size: int = 50,
        max_runtime_in_seconds: int = 36000,
    ):
        res = self.http_client.post(
            f"models/{self.model_ref}/train",
            data={
                "dataset_ref": dataset_ref,
                "anyma_version": anyma_version,
                "instance_type": instance_type,
                "volume_size": volume_size,
                "max_runtime_in_seconds": max_runtime_in_seconds,
                "hyperparameters": hyperparameters.model_dump(),
            },
        )
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.warning(f"Failed to train model: {res.status_code} {res.text}")
            return None

    def train_status(self):
        res = self.http_client.get(f"models/{self.model_ref}/train/status")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get train status: {res.status_code} {res.text}"
            )
            raise ValueError(
                f"Failed to get train status: {res.status_code} {res.text}"
            )

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        if self.runtime is None:
            raise ValueError("Model is not deployed (locally)")
        return self.runtime.benchmark(iterations=iterations, size=size)

    def deploy(
        self, deployment_mode: DeploymentMode = DeploymentMode.REMOTE, wait: bool = True
    ):
        self.get_info()
        if self.metadata.status not in [
            ModelStatus.DEPLOYED,
            ModelStatus.TRAINING_COMPLETED,
        ]:
            raise ValueError(
                f"Model {self.model_ref} is not in a valid state to be deployed. Current status: {self.metadata.status}, expected: {ModelStatus.TRAINING_COMPLETED}"
            )
        if deployment_mode == DeploymentMode.LOCAL:
            model_dir = self._download_model()
            self.runtime = ONNXRuntime(
                model_path=model_dir,
                opts=OnnxEngineOpts(
                    cuda=True,
                    coreml=True,
                    warmup_iter=0,
                    trt=False,
                    verbose=False,
                    fp16=False,
                ),
                model_metadata=self.metadata,
            )
            return
        if self.metadata.status == ModelStatus.DEPLOYED:
            deployment_info = self._deployment_info()
            self.logger.debug(
                f"Model {self.model_ref} is already deployed, deployment info: {deployment_info}"
            )
            return deployment_info

        self.logger.info(
            f"🚀 Deploying model {self.model_ref} to inference endpoint... this might take a while."
        )
        res = self.http_client.post(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 201, 409]:
            if res.status_code == 409:
                self.logger.info(f"Status code 409, model is already deployed")

            if wait:
                for i in range(self.max_deploy_wait):
                    self.logger.info(
                        f"⏱️ Waiting for model {self.model_ref} to be ready... {i+1} of {self.max_deploy_wait}"
                    )
                    if self._deployment_info()["status"] == "READY":
                        self.logger.info(
                            f"✅ Model {self.model_ref} deployed successfully"
                        )
                        return
                    time.sleep(1 + i)
                self.logger.error(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
                raise ValueError(
                    f"Model {self.model_ref} deployment timed out after {self.max_deploy_wait} attempts."
                )
            return res.json()
        else:
            self.logger.error(f"Failed to deploy model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to deploy model: {res.status_code} {res.text}")

    def unload(self):
        res = self.http_client.delete(f"models/{self.model_ref}/deploy")
        if res.status_code in [200, 204, 409]:
            return res.json()
        else:
            self.logger.error(f"Failed to unload model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to unload model: {res.status_code} {res.text}")

    def train_logs(self) -> list[str]:
        res = self.http_client.get(f"models/{self.model_ref}/train/logs")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.warning(
                f"Failed to get train logs: {res.status_code} {res.text}"
            )
            return []

    def _deployment_info(self):
        res = self.http_client.get(f"models/{self.model_ref}/deploy")
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get deployment info: {res.status_code} {res.text}"
            )
            raise ValueError(
                f"Failed to get deployment info: {res.status_code} {res.text}"
            )

    def _annotate(self, im: np.ndarray, detections: Detections) -> np.ndarray:
        classes = self.metadata.classes
        if classes is not None:
            labels = [
                f"{classes[int(class_id)]}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
            ]
        else:
            labels = [
                f"{str(class_id)}: {confid*100:.0f}%"
                for class_id, confid in zip(detections.class_id, detections.confidence)
            ]
        if self.metadata.task == FocoosTask.DETECTION:
            annotated_im = self.box_annotator.annotate(
                scene=im.copy(), detections=detections
            )

            annotated_im = self.label_annotator.annotate(
                scene=annotated_im, detections=detections, labels=labels
            )
        elif self.metadata.task == FocoosTask.SEMSEG:
            annotated_im = self.mask_annotator.annotate(
                scene=im.copy(), detections=detections
            )
        return annotated_im

    def infer(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        if self.runtime is None:
            raise ValueError("Model is not deployed (locally)")
        resize = None  #!TODO remove this, and take it from the metadata
        if self.metadata.task == FocoosTask.DETECTION:
            resize = 640  #!TODO remove this, and take it from the metadata
        self.logger.debug(f"Resize: {resize}")
        t0 = perf_counter()
        im1, im0 = image_preprocess(image_path, resize=resize)
        t1 = perf_counter()
        detections = self.runtime(im1.astype(np.float32), threshold)
        t2 = perf_counter()
        if resize:
            detections = scale_detections(
                detections, (resize, resize), (im0.shape[1], im0.shape[0])
            )

        self.logger.debug(f"Inference time: {t2-t1:.3f} seconds")
        im = None
        if annotate:
            im = self._annotate(im0, detections)

        out = sv_to_focoos_detections(detections, classes=self.metadata.classes)
        t3 = perf_counter()
        out.latency = {
            "inference": round(t2 - t1, 3),
            "preprocess": round(t1 - t0, 3),
            "postprocess": round(t3 - t2, 3),
        }
        return out, im

    def remote_infer(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5,
        annotate: bool = False,
    ) -> Tuple[FocoosDetections, Optional[np.ndarray]]:
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        files = {"file": open(image_path, "rb")}
        t0 = time.time()
        res = self.http_client.post(
            f"models/{self.model_ref}/inference?confidence_threshold={threshold}",
            files=files,
        )
        t1 = time.time()
        if res.status_code == 200:
            self.logger.debug(f"Inference time: {t1-t0:.3f} seconds")
            detections = FocoosDetections(
                detections=[
                    FocoosDet.model_validate(d)
                    for d in res.json().get("detections", [])
                ],
                latency=res.json().get("latency", None),
            )
            preview = None
            if annotate:
                im0 = image_loader(image_path)
                sv_detections = focoos_detections_to_supervision(detections)
                preview = self._annotate(im0, sv_detections)
            return detections, preview
        else:
            self.logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")

    def _download_model(
        self,
    ):
        model_path = os.path.join(self.model_dir, "model.onnx")
        metadata_path = os.path.join(self.model_dir, "focoos_metadata.json")
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            self.logger.info(f"📥 Model already downloaded")
            return model_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        presigned_url = self.http_client.get(
            f"models/{self.model_ref}/download?format=onnx"
        )
        if presigned_url.status_code == 200:
            with open(metadata_path, "w") as f:
                f.write(self.metadata.model_dump_json())
            model_uri = presigned_url.json()
            self.logger.debug(f"Model URI: {model_uri}")
            self.logger.info(f"📥 Downloading model from Focoos Cloud.. ")
            response = self.http_client.get_external_url(model_uri, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                self.logger.info(f"📥 Size: {total_size / (1024**2):.2f} MB")
                with open(model_path, "wb") as f, tqdm(
                    desc=str(model_path).split("/")[-1],
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                self.logger.info(f"📥 File downloaded: {model_path}")
                return model_path
        else:
            self.logger.error(
                f"Failed to download model: {presigned_url.status_code} {presigned_url.text}"
            )
            raise ValueError(
                f"Failed to download model: {presigned_url.status_code} {presigned_url.text}"
            )
