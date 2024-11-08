import argparse
import os

from focoos import Focoos
from focoos.ports import DeploymentMode

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
if __name__ == "__main__":
    args = parser.parse_args()

    focoos = Focoos(api_key=os.environ["FOCOOS_API_KEY"])
    m = focoos.get_model(args.model)
    m.deploy(deployment_mode=DeploymentMode.LOCAL)
    metrics = m.benchmark(iterations=10, size=640)
    print(f"{args.model} {metrics}")
