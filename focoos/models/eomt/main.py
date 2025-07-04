from focoos.model_manager import ModelManager
from focoos.ports import Task
from focoos.utils.vision import annotate_image, image_loader

if __name__ == "__main__":
    model = ModelManager.get("eomt-m-coco-ins")
    image = image_loader("https://public.focoos.ai/samples/image1.png")

    results = model(image, threshold=0.0, top_k=300, predict_all_pixels=True)
    print(results)
    annotate_image(image, results, task=Task.INSTANCE_SEGMENTATION, classes=model.model_info.classes)
