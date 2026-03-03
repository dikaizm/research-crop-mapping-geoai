"""
SegFormer wrapper using segmentation-models-pytorch.

Usage:
    from src.models import build_segformer

    model = build_segformer(
        encoder_name="mit_b2",
        in_channels=27,
        num_classes=12,
    )
"""
import segmentation_models_pytorch as smp


def build_segformer(
    encoder_name: str = "mit_b2",
    encoder_weights: str = "imagenet",
    in_channels: int = 27,
    num_classes: int = 12,
) -> smp.Segformer:
    """
    Build a SegFormer model with Mix Transformer (MiT) encoder.

    SegFormer already contains Efficient Self-Attention in every encoder
    block — no additional attention module is required.

    Args:
        encoder_name:     MiT variant: "mit_b0" … "mit_b5" (default "mit_b2").
        encoder_weights:  Pretrained weights (default "imagenet").
        in_channels:      Number of input channels (spectral-temporal bands).
        num_classes:      Number of output segmentation classes.

    Returns:
        smp.Segformer model instance.
    """
    return smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
