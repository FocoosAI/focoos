from focoos import ModelManager

RESOLUTION = 224
DEVICE = "mps"
NUM_ITERATIONS = 10_000


def main():
    model = ModelManager.get("fai-cls-n-coco")
    model.benchmark(iterations=NUM_ITERATIONS, size=(RESOLUTION, RESOLUTION), device=DEVICE)


if __name__ == "__main__":
    main()
