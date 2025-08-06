import torch

from focoos.models.yoloxpose.config import YOLOXPoseConfig
from focoos.models.yoloxpose.modelling import YOLOXPose
from focoos.nn.backbone.darknet import C2fDarkNetConfig

# Script to test the YOLOXPose model from modelling.py with various configurations


def run_test(config_kwargs, image_shape=(2, 3, 640, 640)):
    print("\nTesting with config:")
    for k, v in config_kwargs.items():
        print(f"  {k}: {v}")
    config = YOLOXPoseConfig(**config_kwargs)
    dummy_images = torch.randn(*image_shape)
    model = YOLOXPose(config)
    model.eval()
    print("YOLOXPose model created successfully")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Input shape: {dummy_images.shape}")
    with torch.no_grad():
        outputs = model(dummy_images)
    print("YOLOXPose outputs:")
    print(f"Output type: {type(outputs)}")
    if hasattr(outputs, "outputs"):
        print(f"Outputs type: {type(outputs.outputs)}")
        if hasattr(outputs.outputs, "scores"):
            print(f"Scores shape: {outputs.outputs.scores.shape}")
        if hasattr(outputs.outputs, "pred_bboxes"):
            print(f"Pred bboxes shape: {outputs.outputs.pred_bboxes.shape}")
        if hasattr(outputs.outputs, "pred_keypoints"):
            print(f"Pred keypoints shape: {outputs.outputs.pred_keypoints.shape}")
    if hasattr(outputs, "loss"):
        print(f"Loss: {outputs.loss}")
    print("Test completed successfully!\n" + "-" * 40)


if __name__ == "__main__":
    # List of different parameter sets to test
    test_configs = [
        # Default config
        dict(
            backbone_config=C2fDarkNetConfig(size="s"),
            neck_feat_dim=256,
            neck_out_dim=256,
            c2f_depth=1,
            num_keypoints=17,
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            stacked_convs=2,
            featmap_strides=[32, 16, 8],
            norm="BN",
            activation="relu",
            use_aux_loss=False,
            overlaps_power=1.0,
            score_thr=0.01,
            nms_topk=100000,
            nms_thr=1.0,
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=True,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        ),
        # Different backbone size and activation
        dict(
            backbone_config=C2fDarkNetConfig(size="m"),
            neck_feat_dim=128,
            neck_out_dim=128,
            c2f_depth=2,
            num_keypoints=17,
            num_classes=2,
            in_channels=128,
            feat_channels=128,
            stacked_convs=1,
            featmap_strides=[32, 16, 8],
            norm="BN",
            activation="silu",
            use_aux_loss=True,
            overlaps_power=2.0,
            score_thr=0.05,
            nms_topk=5000,
            nms_thr=0.5,
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=False,
            pixel_mean=[0.0, 0.0, 0.0],
            pixel_std=[1.0, 1.0, 1.0],
        ),
        # Different number of keypoints and classes, different normalization
        dict(
            backbone_config=C2fDarkNetConfig(
                size="l",
            ),
            neck_feat_dim=64,
            neck_out_dim=64,
            c2f_depth=1,
            num_keypoints=5,
            num_classes=3,
            in_channels=64,
            feat_channels=64,
            stacked_convs=3,
            featmap_strides=[32, 16, 8],
            norm=None,
            activation="leaky_relu",
            use_aux_loss=False,
            overlaps_power=1.5,
            score_thr=0.1,
            nms_topk=2000,
            nms_thr=0.3,
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=True,
            pixel_mean=[128.0, 128.0, 128.0],
            pixel_std=[64.0, 64.0, 64.0],
        ),
        # Test with different image size and pixel normalization
        dict(
            backbone_config=C2fDarkNetConfig(
                size="x",
            ),
            neck_feat_dim=32,
            neck_out_dim=32,
            c2f_depth=2,
            num_keypoints=10,
            num_classes=5,
            in_channels=32,
            feat_channels=32,
            stacked_convs=2,
            featmap_strides=[20, 10, 5],
            norm="BN",
            activation="gelu",
            use_aux_loss=True,
            overlaps_power=0.5,
            score_thr=0.2,
            nms_topk=100,
            nms_thr=0.9,
            featmap_strides_pointgenerator=[(20, 20), (10, 10), (5, 5)],
            centralize_points_pointgenerator=False,
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
        ),
    ]

    image_shapes = [
        (2, 3, 640, 640),
        (1, 3, 320, 320),
        (4, 3, 128, 128),
        (2, 3, 256, 512),
    ]

    for i, (cfg, img_shape) in enumerate(zip(test_configs, image_shapes)):
        print(f"\n=== Running test {i + 1} ===")
        run_test(cfg, image_shape=img_shape)
