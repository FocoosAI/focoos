import torch

from focoos.models.rtmo.config import RTMOConfig
from focoos.models.rtmo.modelling import RTMO
from focoos.nn.backbone.darknet import DarkNetConfig

# Script to test the RTMO model from modelling.py with various configurations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_test(config_kwargs, image_shape=(2, 3, 640, 640)):
    print("\nTesting with config:")
    for k, v in config_kwargs.items():
        print(f"  {k}: {v}")
    config = RTMOConfig(**config_kwargs)
    dummy_images = torch.randn(*image_shape)
    model = RTMO(config)
    print(f"# params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"# train`params: {sum(p.numel() for p in model.head.parameters()):,}")
    model.to(DEVICE)
    model.eval()
    print("RTMO model created successfully")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Input shape: {dummy_images.shape}")
    with torch.no_grad():
        dummy_images = dummy_images.to(DEVICE)
        outputs = model(dummy_images)
    print("RTMO outputs:")
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
            backbone_config=DarkNetConfig(size="s"),
            num_classes=1,
            num_keypoints=17,
            in_channels=256,
            pose_vec_channels=256,
            feat_channels=256,
            stacked_convs=2,
            activation="relu",
            featmap_strides=[32, 16, 8],
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=True,
            bbox_padding=1.25,
            nms_pre=100000,
            nms_thr=1.0,
            score_thr=0.01,
            norm="BN",
            use_aux_loss=False,
            overlaps_power=1.0,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            neck_feat_dim=256,
            neck_out_dim=256,
            c2f_depth=1,
            feat_channels_dcc=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_s=128,
            gau_expansion_factor=2,
            gau_dropout_rate=0.0,
        ),
        # Different backbone size and activation
        dict(
            backbone_config=DarkNetConfig(size="m"),
            num_classes=2,
            num_keypoints=17,
            in_channels=128,
            pose_vec_channels=128,
            feat_channels=128,
            stacked_convs=1,
            activation="silu",
            featmap_strides=[32, 16, 8],
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=False,
            bbox_padding=1.0,
            nms_pre=5000,
            nms_thr=0.5,
            score_thr=0.05,
            norm="BN",
            use_aux_loss=True,
            overlaps_power=2.0,
            pixel_mean=[0.0, 0.0, 0.0],
            pixel_std=[1.0, 1.0, 1.0],
            neck_feat_dim=128,
            neck_out_dim=128,
            c2f_depth=2,
            feat_channels_dcc=64,
            num_bins=(96, 128),
            spe_channels=64,
            gau_s=64,
            gau_expansion_factor=1,
            gau_dropout_rate=0.1,
        ),
        # Different number of keypoints and classes, different normalization
        dict(
            backbone_config=DarkNetConfig(size="l"),
            num_classes=3,
            num_keypoints=5,
            in_channels=64,
            pose_vec_channels=64,
            feat_channels=64,
            stacked_convs=3,
            activation="leaky_relu",
            featmap_strides=[32, 16, 8],
            featmap_strides_pointgenerator=[(32, 32), (16, 16), (8, 8)],
            centralize_points_pointgenerator=True,
            bbox_padding=1.5,
            nms_pre=2000,
            nms_thr=0.3,
            score_thr=0.1,
            norm=None,
            use_aux_loss=False,
            overlaps_power=1.5,
            pixel_mean=[128.0, 128.0, 128.0],
            pixel_std=[64.0, 64.0, 64.0],
            neck_feat_dim=64,
            neck_out_dim=64,
            c2f_depth=1,
            feat_channels_dcc=32,
            num_bins=(48, 64),
            spe_channels=32,
            gau_s=32,
            gau_expansion_factor=1,
            gau_dropout_rate=0.2,
        ),
        # Test with different image size and pixel normalization
        dict(
            backbone_config=DarkNetConfig(size="x"),
            num_classes=5,
            num_keypoints=10,
            in_channels=32,
            pose_vec_channels=32,
            feat_channels=32,
            stacked_convs=2,
            activation="gelu",
            featmap_strides=[20, 10, 5],
            featmap_strides_pointgenerator=[(20, 20), (10, 10), (5, 5)],
            centralize_points_pointgenerator=False,
            bbox_padding=1.1,
            nms_pre=100,
            nms_thr=0.9,
            score_thr=0.2,
            norm="BN",
            use_aux_loss=True,
            overlaps_power=0.5,
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
            neck_feat_dim=32,
            neck_out_dim=32,
            c2f_depth=2,
            feat_channels_dcc=16,
            num_bins=(24, 32),
            spe_channels=16,
            gau_s=16,
            gau_expansion_factor=2,
            gau_dropout_rate=0.0,
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
