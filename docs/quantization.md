# Quantization (Beta)

The quantization of Focoos models is currently in working in progress stage.

currently tested and working for **classification models**.

## Example

```python
from focoos import ModelManager,ASSETS_DIR, MODELS_DIR, RuntimeType
from focoos.infer.quantizer import OnnxQuantizer, QuantizationCfg
import os

image_size = 224  # 224px input size
model_name = "fai-cls-m-coco" # you can also take model from focoos hub with "hub://YOUR_MODEL_REF"
im = ASSETS_DIR / "federer.jpg"

model = ModelManager.get(model_name)

exported_model = model.export(runtime_type=RuntimeType.ONNX_CPU, # optimized for edge or cpu
    image_size=image_size,
    dynamic_axes=False,  # quantization need static axes!
    simplify_onnx=False, # simplify and optimize onnx model graph
    onnx_opset=18,
    out_dir=os.path.join(MODELS_DIR, "my_edge_model")) # save to models dir

# benchmark onnx model
exported_model.benchmark(iterations=100)

# test onnx model

result = exported_model.infer(im,annotate=True)
Image.fromarray(result.image)


quantization_cfg = QuantizationCfg(
  size = image_size, # input size: must be same as exported model
  calibration_images_folder = str(ASSETS_DIR), # Calibration images folder: It is strongly recommended
                                               # to use the dataset validation split on which the model was trained.
                                               # Here, for example, we will use the assets folder.
  format="QDQ", # QO (QOperator): All the quantized operators have their own ONNX definitions, like QLinearConv, MatMulInteger etc.
                # QDQ (Quantize-DeQuantize): inserts DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators to simulate the quantization and dequantization process.
  per_channel=True,      # Per-channel quantization: each channel has its own scale/zero-point → more accurate,
                         # especially for convolutions, at the cost of extra memory and computation.
  normalize_images=True, # normalize images during preprocessing: some models have normalization outside of model forward
)

quantizer = OnnxQuantizer(
    input_model_path=exported_model.model_path,
    cfg=quantization_cfg
)
model_path = quantizer.quantize(
  benchmark=True # benchmark bot fp32 and int8 models
)

quantized_model = InferModel(model_path, runtime_type=RuntimeType.ONNX_CPU)

res = quantized_model.infer(im,annotate=True)
Image.fromarray(res.image)

```
