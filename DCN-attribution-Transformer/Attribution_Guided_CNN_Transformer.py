"""Define the attribution-guided CNN-Transformer ablation model."""

import torch
import torch.nn as nn

from CNN import CNN
from attribution_guided_model import (
    AttributionGuidedAttention,
    DotProductAttributionNetwork,
)
from Dense_transformers import ConstrainedBranchFusion, DTransformer, Encoder


class AttributionGuidedCNNTransformer(nn.Module):
    """Fuse DeepLIFT-derived spatial guidance with standard CNN features."""

    def __init__(
        self,
        num_classes=2,
        feature_dim=512,
        transformer_depth=16,
        transformer_heads=8,
        factor_pooling="mean",
        factor_branch_type="mlp",
        dropout=0.15,
    ):
        super().__init__()
        if feature_dim % transformer_heads != 0:
            raise ValueError("feature_dim must be divisible by transformer_heads")

        self.dcn = CNN(num_classes=num_classes, output_dim=feature_dim)
        self.attribution_network = DotProductAttributionNetwork(
            input_channels=42,
            hidden_dim=max(64, feature_dim),
            output_dim=feature_dim,
            factor_pooling=factor_pooling,
            factor_branch_type=factor_branch_type,
        )
        self.attribution_attention = AttributionGuidedAttention(
            feature_dim=feature_dim,
            num_heads=transformer_heads,
            dropout=dropout,
        )
        encoder = Encoder(
            dim=feature_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=feature_dim // transformer_heads,
            mlp_dim=feature_dim * 2,
            dropout=dropout,
        )
        self.transformer = DTransformer(
            image_size=9,
            patch_size=1,
            attn_layers=encoder,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.logit_fusion = ConstrainedBranchFusion(
            initial_weights=(0.25, 0.55, 0.20), minimum_weight=0.15
        )

    def forward(
        self,
        x,
        return_attribution=False,
        return_attention=False,
        return_factor_attribution=False,
        return_branch_logits=False,
        disable_attribution_guidance=False,
    ):
        attribution_key, attribution_scores, factor_info = self.attribution_network(
            x, return_factor_attribution=True
        )
        cnn_features = self.dcn(x, extract_features=True)
        fused_features = (
            cnn_features
            if disable_attribution_guidance
            else (
                self.attribution_attention(
                    cnn_features, attribution_key, attribution_scores
                )
            )
        )

        if return_attention:
            transformer_logits, attention_maps = self.transformer(
                fused_features, return_attention=True
            )
        else:
            transformer_logits = self.transformer(fused_features)
            attention_maps = None

        base_logits = self.dcn.classifier(cnn_features)
        factor_logit = factor_info["branch_logit"]
        factor_logits = torch.cat((-0.5 * factor_logit, 0.5 * factor_logit), dim=1)
        logits = self.logit_fusion(base_logits, transformer_logits, factor_logits)
        branch_info = {
            "base": base_logits,
            "transformer": transformer_logits,
            "factor": factor_logits,
            "fusion_weights": self.logit_fusion.weights(),
        }

        results = [logits]
        if return_attribution:
            results.append(attribution_scores)
        if return_attention:
            results.append(attention_maps)
        if return_factor_attribution:
            results.append(factor_info)
        if return_branch_logits:
            results.append(branch_info)
        return results[0] if len(results) == 1 else tuple(results)

    def get_attribution_map(self, x):
        with torch.no_grad():
            _, scores = self.attribution_network(x)
        return scores

    def get_factor_attribution(self, x):
        self.eval()
        factors = self.attribution_network._get_factor_input(x)
        with torch.no_grad():
            (
                _,
                contributions,
                completeness_delta,
            ) = self.attribution_network.factor_branch.explain(factors)
        return contributions, completeness_delta

    def fusion_diagnostics(self):
        weights = self.logit_fusion.weights().detach().cpu().tolist()
        return {
            "base_weight": weights[0],
            "transformer_weight": weights[1],
            "factor_attribution_weight": weights[2],
            **self.attribution_attention.fusion_diagnostics(),
        }


def create_attribution_guided_cnn_model(
    num_classes=2,
    feature_dim=512,
    transformer_depth=16,
    transformer_heads=8,
    factor_pooling="mean",
    factor_branch_type="mlp",
    dropout=0.15,
):
    return AttributionGuidedCNNTransformer(
        num_classes=num_classes,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        factor_pooling=factor_pooling,
        factor_branch_type=factor_branch_type,
        dropout=dropout,
    )


if __name__ == "__main__":
    from Dense_transformers import run_model_pipeline

    def workflow_model():
        return create_attribution_guided_cnn_model(
            feature_dim=32,
            transformer_depth=1,
            transformer_heads=2,
            factor_branch_type="mlp",
        )

    run_model_pipeline(workflow_model, "attribution_guided_cnn_transformer")
