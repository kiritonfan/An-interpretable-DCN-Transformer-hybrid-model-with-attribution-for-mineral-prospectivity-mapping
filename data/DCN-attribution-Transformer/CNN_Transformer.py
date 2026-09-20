"""Define the standard CNN-Transformer comparison model."""

import torch
import torch.nn as nn

from CNN import CNN
from Dense_transformers import ConstrainedBranchFusion, DTransformer, Encoder


class CNNTransformerModel(nn.Module):
    """Standard CNN feature extractor followed by the unchanged Transformer."""

    def __init__(
        self, num_classes=2, feature_dim=32, transformer_depth=1, transformer_heads=2
    ):
        super().__init__()
        self.cnn = CNN(num_classes=num_classes, output_dim=feature_dim)
        encoder = Encoder(
            dim=feature_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=feature_dim // transformer_heads,
            mlp_dim=feature_dim * 2,
            dropout=0.15,
        )
        self.transformer = DTransformer(
            image_size=9,
            patch_size=1,
            attn_layers=encoder,
            num_classes=num_classes,
            dropout=0.15,
        )
        # Both classifiers keep a meaningful share throughout training.  The
        # former zero-initialized residual gate collapsed to |gate| ~= 0.016.
        self.logit_fusion = ConstrainedBranchFusion(
            initial_weights=(0.45, 0.55), minimum_weight=0.25
        )

    def forward(self, x, return_branch_logits=False):
        features = self.cnn(x, extract_features=True)
        base_logits = self.cnn.classifier(features)
        transformer_logits = self.transformer(features)
        output = self.logit_fusion(base_logits, transformer_logits)
        if return_branch_logits:
            return output, {
                "base": base_logits,
                "transformer": transformer_logits,
                "fusion_weights": self.logit_fusion.weights(),
            }
        return output

    def fusion_diagnostics(self):
        weights = self.logit_fusion.weights().detach().cpu().tolist()
        return {"base_weight": weights[0], "transformer_weight": weights[1]}


def create_cnn_transformer_model(
    num_classes=2, feature_dim=32, transformer_depth=1, transformer_heads=2
):
    return CNNTransformerModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
    )


if __name__ == "__main__":
    from Dense_transformers import run_model_pipeline

    run_model_pipeline(create_cnn_transformer_model, "cnn_transformer")
