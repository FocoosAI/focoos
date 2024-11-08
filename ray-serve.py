import os

import starlette
import torch
from ray import serve

from focoos import Focoos
from focoos.ports import DeploymentMode, FocoosEnvHostUrl

NUM_REPLICAS = 1
NUM_CPUS = 1
NUM_GPUS = 0.5
NUM_MODELS_PER_REPLICA = os.getenv("NUM_MODELS_PER_REPLICA", None)
print(f"1NUM_MODELS_PER_REPLICA: {NUM_MODELS_PER_REPLICA}")


@serve.deployment(
    name="multi-model",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": NUM_CPUS, "num_gpus": NUM_GPUS},
)
class ModelInferencer:
    def __init__(self):
        self.focoos_client = Focoos(
            api_key=os.getenv("FOCOOS_API_KEY"), host_url=FocoosEnvHostUrl.DEV
        )
        print("MULTI MODEL INFERENCER INITIALIZED")
        print(f"2NUM_MODELS_PER_REPLICA: {NUM_MODELS_PER_REPLICA}")

    @serve.multiplexed(max_num_models_per_replica=int(NUM_MODELS_PER_REPLICA))
    async def get_model(self, model_id: str):
        print(f"GETTING MODEL {model_id}")
        model = self.focoos_client.get_model(model_id)
        model.deploy(deployment_mode=DeploymentMode.LOCAL)
        return model

    async def __call__(self, request: starlette.requests.Request):
        model_id = serve.get_multiplexed_model_id()
        model = await self.get_model(model_id)
        return model.infer(await request.body())[0]


app = ModelInferencer.bind()
