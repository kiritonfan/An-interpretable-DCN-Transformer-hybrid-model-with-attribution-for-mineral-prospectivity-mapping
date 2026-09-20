"""Define the deformable CNN-Transformer comparison model."""

import torch.nn as nn

from dcnv2_model import DCNv2
from Dense_transformers import Encoder, LegacyCLSDTransformer


class DCNTransformerModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        feature_dim=32,
        transformer_depth=1,
        transformer_heads=2,
        dropout=0.15,
    ):
        super().__init__()
        self.dcn = DCNv2(num_classes=num_classes, output_dim=feature_dim)
        encoder = Encoder(
            dim=feature_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=feature_dim // transformer_heads,
            # Historical CSV checkpoint used the legacy 4x FFN width.
            mlp_dim=feature_dim * 4,
            dropout=dropout,
        )
        self.transformer = LegacyCLSDTransformer(
            image_size=9,
            patch_size=1,
            attn_layers=encoder,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x, return_attention=False):
        features = self.dcn(x, extract_features=True)
        return self.transformer(features, return_attention=return_attention)


def create_dcn_transformer_model(
    num_classes=2,
    feature_dim=32,
    transformer_depth=1,
    transformer_heads=2,
    dropout=0.15,
):
    return DCNTransformerModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        dropout=dropout,
    )


if __name__ == "__main__":
    from Dense_transformers import run_model_pipeline

    run_model_pipeline(
        create_dcn_transformer_model,
        "dcn_transformer",
        pipeline_defaults={"epochs": 200, "patience": 50},
    )
