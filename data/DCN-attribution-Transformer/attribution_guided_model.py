"""Implement attribution-guided attention and classification components."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dcnv2_model import DCNv2
from Dense_transformers import ConstrainedBranchFusion, DTransformer, Encoder


def _bounded_logit_initial(value, lower, upper):
    """Return the unconstrained logit for a bounded sigmoid parameter."""
    if not lower < value < upper:
        raise ValueError("initial value must lie strictly inside its bounds")
    proportion = (value - lower) / (upper - lower)
    return torch.logit(torch.tensor([proportion], dtype=torch.float32))


class FactorDeepLIFTBranch(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.register_buffer("reference_input", torch.zeros(1, input_dim))
        self.register_buffer("reference_initialized", torch.tensor(False))

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        return self.fc3(a2), (z1, a1, z2, a2)

    @staticmethod
    def _rescale_multiplier(delta_out, delta_in, eps=1e-12):

        safe_delta = torch.where(
            delta_in.abs() > eps, delta_in, torch.ones_like(delta_in)
        )
        ratio = delta_out / safe_delta
        return torch.where(delta_in.abs() > eps, ratio, torch.zeros_like(ratio))

    def set_reference(self, reference):
        reference = reference.detach().to(
            device=self.reference_input.device, dtype=self.reference_input.dtype
        )
        if reference.dim() == 1:
            reference = reference.unsqueeze(0)
        if reference.shape != self.reference_input.shape:
            raise ValueError(
                f"因子基线形状错误，期望 {tuple(self.reference_input.shape)}，"
                f"实际 {tuple(reference.shape)}"
            )
        self.reference_input.copy_(reference)
        self.reference_initialized.fill_(True)

    def explain(self, x):

        if x.dim() != 2:
            raise ValueError(f"因子归因输入应为 (B,42)，实际为 {tuple(x.shape)}")

        branch_logit, _ = self.forward(x)

        with torch.no_grad():
            reference = self.reference_input.to(device=x.device, dtype=x.dtype)
            _, (z1, a1, z2, a2) = self.forward(x)
            ref_logit, (ref_z1, ref_a1, ref_z2, ref_a2) = self.forward(
                reference.expand(x.size(0), -1)
            )

            multiplier_a2 = self.fc3.weight.expand(x.size(0), -1)
            multiplier_z2 = multiplier_a2 * self._rescale_multiplier(
                a2 - ref_a2, z2 - ref_z2
            )
            multiplier_a1 = multiplier_z2 @ self.fc2.weight
            multiplier_z1 = multiplier_a1 * self._rescale_multiplier(
                a1 - ref_a1, z1 - ref_z1
            )
            multiplier_x = multiplier_z1 @ self.fc1.weight

            contributions = multiplier_x * (x - reference)
            delta_output = (branch_logit.detach() - ref_logit).squeeze(-1)
            completeness_delta = contributions.sum(dim=1) - delta_output
        return branch_logit, contributions, completeness_delta


class LinearFactorDeepLIFTBranch(nn.Module):
    def __init__(self, input_dim=42):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer("reference_input", torch.zeros(1, input_dim))
        self.register_buffer("reference_initialized", torch.tensor(False))

    def forward(self, x):
        return self.linear(x), ()

    def set_reference(self, reference):
        reference = reference.detach().to(
            device=self.reference_input.device, dtype=self.reference_input.dtype
        )
        if reference.dim() == 1:
            reference = reference.unsqueeze(0)
        if reference.shape != self.reference_input.shape:
            raise ValueError(
                f"因子基线形状错误，期望 {tuple(self.reference_input.shape)}，"
                f"实际 {tuple(reference.shape)}"
            )
        self.reference_input.copy_(reference)
        self.reference_initialized.fill_(True)

    def explain(self, x):
        if x.dim() != 2:
            raise ValueError(f"因子归因输入应为 (B,42)，实际为 {tuple(x.shape)}")
        branch_logit = self.linear(x)
        reference = self.reference_input.to(device=x.device, dtype=x.dtype)
        contributions = (x - reference) * self.linear.weight
        ref_logit = self.linear(reference.expand(x.size(0), -1)).squeeze(-1)
        completeness_delta = contributions.sum(dim=1) - (
            branch_logit.squeeze(-1) - ref_logit
        )
        return branch_logit, contributions, completeness_delta


class DotProductAttributionNetwork(nn.Module):
    def __init__(
        self,
        input_channels=42,
        hidden_dim=256,
        output_dim=512,
        factor_pooling="mean",
        factor_branch_type="linear",
    ):
        super(DotProductAttributionNetwork, self).__init__()

        if factor_pooling not in ("mean", "center"):
            raise ValueError("factor_pooling 必须是 'mean' 或 'center'")
        self.factor_pooling = factor_pooling
        self.factor_branch_type = factor_branch_type

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.input_channels = input_channels

        self.register_buffer("reference_input", torch.zeros(1, input_channels, 9, 9))
        self.register_buffer("reference_initialized", torch.tensor(False))

        self.output_projection = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        if factor_branch_type == "linear":
            self.factor_branch = LinearFactorDeepLIFTBranch(input_dim=input_channels)
        elif factor_branch_type == "mlp":
            self.factor_branch = FactorDeepLIFTBranch(
                input_dim=input_channels, hidden_dim=64
            )
        else:
            raise ValueError("factor_branch_type 必须为 'mlp' 或 'linear'")
        self._initialize_weights()

    def update_reference_features(self, train_data_loader, device):

        self.eval()
        input_accumulated = None
        factor_accumulated = None
        count = 0
        with torch.no_grad():
            for data, labels in train_data_loader:
                negative = labels.eq(0)
                if not negative.any():
                    continue
                data = data[negative].to(device)
                batch_sum = data.sum(dim=0, keepdim=True)
                input_accumulated = (
                    batch_sum
                    if input_accumulated is None
                    else input_accumulated + batch_sum
                )
                factor_input = self._get_factor_input(data)  # (B, 42)
                factor_sum = factor_input.sum(dim=0, keepdim=True)
                factor_accumulated = (
                    factor_sum
                    if factor_accumulated is None
                    else factor_accumulated + factor_sum
                )
                count += data.size(0)

        if count == 0:
            raise RuntimeError("当前训练折没有无矿样本，无法初始化归因基线")
        mean_input = input_accumulated / count
        if mean_input.shape[-2:] != self.reference_input.shape[-2:]:
            mean_input = F.interpolate(
                mean_input,
                size=self.reference_input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        self.reference_input.copy_(mean_input)
        self.reference_initialized.fill_(True)
        self.factor_branch.set_reference(factor_accumulated / count)
        print(f"[归因基线] 已用当前训练折 {count} 个无矿窗口固定初始化")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_factor_input(self, x):

        if x.dim() != 4 or x.size(1) != 42:
            raise ValueError(f"输入应为 (B,42,H,W)，实际为 {tuple(x.shape)}")
        if self.factor_pooling == "center":
            return x[:, :, x.size(2) // 2, x.size(3) // 2]
        return x.mean(dim=(2, 3))

    def compute_attribution_scores(self, x, factor_contributions, eps=1e-6):

        if not self.reference_initialized.item():
            raise RuntimeError("归因参考基线尚未用当前训练折的无矿样本初始化")
        reference = self.reference_input
        if reference.shape[-2:] != x.shape[-2:]:
            reference = F.interpolate(
                reference, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        factor_delta = self._get_factor_input(x) - self.factor_branch.reference_input
        nonzero_delta = factor_delta.abs() > eps
        safe_factor_delta = torch.where(
            nonzero_delta, factor_delta, torch.ones_like(factor_delta)
        )
        multipliers = torch.where(
            nonzero_delta,
            factor_contributions / safe_factor_delta,
            torch.zeros_like(factor_contributions),
        )
        local_contributions = multipliers.unsqueeze(-1).unsqueeze(-1) * (x - reference)
        raw_scores = local_contributions.abs().sum(dim=1, keepdim=True)
        flat = torch.log1p(raw_scores).flatten(1)
        standardized = (flat - flat.mean(dim=1, keepdim=True)) / flat.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(eps)
        scores = torch.softmax(standardized, dim=1) * flat.size(1)
        return scores.view_as(raw_scores), local_contributions

    def forward(self, x, return_factor_attribution=False):
        """
        Args:
            x: (B, 42, 9, 9)

        Returns:
            key_features      : (B, output_dim, 9, 9)
            attribution_scores: (B, 1,          9, 9)
            factor_info: optional dict containing (B,42) DeepLIFT contributions
        """
        factor_input = self._get_factor_input(x)
        (
            factor_logit,
            factor_contributions,
            completeness_delta,
        ) = self.factor_branch.explain(factor_input)
        attribution_scores, local_contributions = self.compute_attribution_scores(
            x, factor_contributions
        )
        encoded_features = self.feature_encoder(x)
        key_features = self.output_projection(encoded_features * attribution_scores)

        if return_factor_attribution:
            factor_info = {
                "factor_input": factor_input.detach(),
                "branch_logit": factor_logit,
                "contributions": factor_contributions,
                "completeness_delta": completeness_delta,
                "spatial_contributions": local_contributions.detach(),
            }
            return key_features, attribution_scores, factor_info
        return key_features, attribution_scores


class AttributionGuidedAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.15):
        super(AttributionGuidedAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim 必须能被 num_heads 整除"

        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        self.attribution_strength_bounds = (0.75, 2.50)
        self.guided_mix_bounds = (0.35, 0.85)
        self.attribution_strength_logit = nn.Parameter(
            _bounded_logit_initial(1.50, *self.attribution_strength_bounds)
        )
        self.guided_mix_logit = nn.Parameter(
            _bounded_logit_initial(0.60, *self.guided_mix_bounds)
        )

        self.dropout = nn.Dropout(dropout)

        num_groups = min(8, feature_dim // 4)
        self.norm = nn.GroupNorm(num_groups, feature_dim)

    @staticmethod
    def _bounded_value(parameter, bounds):
        lower, upper = bounds
        return lower + (upper - lower) * torch.sigmoid(parameter)

    def attribution_strength(self):
        return self._bounded_value(
            self.attribution_strength_logit, self.attribution_strength_bounds
        )

    def guided_mix(self):
        return self._bounded_value(self.guided_mix_logit, self.guided_mix_bounds)

    def forward(self, dcn_features, attribution_key, attribution_scores):
        batch_size, _, H, W = dcn_features.size()
        num_spatial = H * W  # 9×9=81

        query = self.query_proj(dcn_features)
        key = self.key_proj(attribution_key)
        value = self.value_proj(dcn_features)

        query = query.view(batch_size, self.num_heads, self.head_dim, -1)
        key = key.view(batch_size, self.num_heads, self.head_dim, -1)
        value = value.view(batch_size, self.num_heads, self.head_dim, -1)

        attention_weights = torch.matmul(query.transpose(-2, -1), key) / (
            self.head_dim**0.5
        )

        if attribution_scores.size(-1) != W or attribution_scores.size(-2) != H:
            attribution_scores = F.adaptive_avg_pool2d(attribution_scores, (H, W))
        attr_flat = attribution_scores.view(batch_size, 1, 1, -1).expand(
            -1, self.num_heads, num_spatial, -1
        )
        attention_weights = attention_weights + self.attribution_strength() * (
            attr_flat - 1.0
        )

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(value, attention_weights.transpose(-2, -1))
        attended = attended.view(batch_size, self.feature_dim, H, W)

        guided_features = self.norm(dcn_features + self.output_proj(attended))
        mix = self.guided_mix()
        output = (1.0 - mix) * dcn_features + mix * guided_features

        return output

    def fusion_diagnostics(self):
        return {
            "attribution_strength": float(self.attribution_strength().detach().cpu()),
            "guided_feature_weight": float(self.guided_mix().detach().cpu()),
        }


class AttributionGuidedDCNTransformer(nn.Module):
    def __init__(
        self,
        num_classes=2,
        feature_dim=512,
        transformer_depth=16,
        transformer_heads=8,
        factor_pooling="mean",
        factor_branch_type="linear",
        dropout=0.15,
        backbone_type="dcn",
    ):
        super(AttributionGuidedDCNTransformer, self).__init__()

        if backbone_type == "dcn":
            self.dcn = DCNv2(num_classes=num_classes, output_dim=feature_dim)
        elif backbone_type == "cnn":
            from CNN import CNN

            self.dcn = CNN(num_classes=num_classes, output_dim=feature_dim)
        else:
            raise ValueError("backbone_type 必须为 'dcn' 或 'cnn'")

        self.attribution_network = DotProductAttributionNetwork(
            input_channels=42,
            hidden_dim=max(64, feature_dim),
            output_dim=feature_dim,
            factor_pooling=factor_pooling,
            factor_branch_type=factor_branch_type,
        )

        self.attribution_attention = AttributionGuidedAttention(
            feature_dim=feature_dim, num_heads=transformer_heads, dropout=dropout
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

        print(f"[OK] 归因引导DCN-Transformer模型已创建（小样本门控版）")
        print(f"- 特征维度 (d_model): {feature_dim}")
        print(f"- DCNv2层数: 3层 (从10层简化)")
        print(f"- 归因网络层数: 3层 (从6层简化)")
        print(f"- Transformer层数: {transformer_depth}")
        print(f"- 注意力头数: {transformer_heads}")
        print(f"- MLP维度: {feature_dim * 2}")
        print(f"- Dropout: {dropout:.2f}")
        print(f"- 输入分辨率: 9x9 = 81个Token")
        print(f"- DeepLIFT因子输入: {factor_pooling}池化后的42维向量")
        print(f"- DeepLIFT因子分支: {factor_branch_type}")

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

        dcn_features = self.dcn(x, extract_features=True)  # (B, feature_dim, 9, 9)

        if disable_attribution_guidance:
            fused_features = dcn_features
        else:
            fused_features = self.attribution_attention(
                dcn_features, attribution_key, attribution_scores
            )

        if return_attention:
            transformer_logits, attention_maps = self.transformer(
                fused_features, return_attention=True
            )
        else:
            transformer_logits = self.transformer(fused_features)
            attention_maps = None
        base_logits = self.dcn.classifier(dcn_features)
        factor_logit = factor_info["branch_logit"]
        factor_class_logits = torch.cat(
            (-0.5 * factor_logit, 0.5 * factor_logit), dim=1
        )
        output = self.logit_fusion(base_logits, transformer_logits, factor_class_logits)
        branch_info = {
            "base": base_logits,
            "transformer": transformer_logits,
            "factor": factor_class_logits,
            "fusion_weights": self.logit_fusion.weights(),
        }

        results = [output]
        if return_attribution:
            results.append(attribution_scores)
        if return_attention:
            results.append(attention_maps)
        if return_factor_attribution:
            results.append(factor_info)
        if return_branch_logits:
            results.append(branch_info)
        return results[0] if len(results) == 1 else tuple(results)

    def fusion_diagnostics(self):
        weights = self.logit_fusion.weights().detach().cpu().tolist()
        return {
            "base_weight": weights[0],
            "transformer_weight": weights[1],
            "factor_attribution_weight": weights[2],
            **self.attribution_attention.fusion_diagnostics(),
        }

    def get_attribution_map(self, x):

        with torch.no_grad():
            _, attribution_scores = self.attribution_network(x)
        return attribution_scores

    def get_factor_attribution(self, x):

        self.eval()
        factor_input = self.attribution_network._get_factor_input(x)
        with torch.no_grad():
            (
                _,
                contributions,
                completeness_delta,
            ) = self.attribution_network.factor_branch.explain(factor_input)
        return contributions, completeness_delta


def create_attribution_guided_model(
    num_classes=2,
    feature_dim=512,
    transformer_depth=16,
    transformer_heads=8,
    factor_pooling="mean",
    factor_branch_type="linear",
    dropout=0.15,
):

    return AttributionGuidedDCNTransformer(
        num_classes=num_classes,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        factor_pooling=factor_pooling,
        factor_branch_type=factor_branch_type,
        dropout=dropout,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_attribution_guided_model(
        num_classes=2, feature_dim=32, transformer_depth=1, transformer_heads=2
    ).to(device)

    test_input = torch.randn(4, 42, 9, 9).to(device)
    output, attr_scores = model(test_input, return_attribution=True)

    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"归因分数形状: {attr_scores.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
