"""This module defines the attribution-guided DCN-Transformer model used in the study."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dcnv2_model import DCNv2
from Dense_transformers import DTransformer, Encoder


class DotProductAttributionNetwork(nn.Module):

    def __init__(self, input_channels=42, hidden_dim=256, output_dim=512,
                 ema_momentum=0.005):
        super(DotProductAttributionNetwork, self).__init__()

                                 
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

                                           
        self.register_buffer('reference_features', torch.zeros(1, hidden_dim, 9, 9))
        self.register_buffer('reference_initialized', torch.tensor(False))

                                                      
                                                                 
        self.ema_momentum = ema_momentum

                                  
        reduced_dim = max(32, hidden_dim // 2)
        self.dot_product_weight = nn.Parameter(torch.randn(reduced_dim, hidden_dim))

                                             
        self.output_projection = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        self._initialize_weights()

    def update_reference_features(self, train_data_loader, device):
        self.eval()
        accumulated = None
        count = 0
        with torch.no_grad():
            for data, _ in train_data_loader:
                data = data.to(device)
                encoded = self.feature_encoder(data)                                 
                batch_sum = encoded.sum(dim=0, keepdim=True)                
                accumulated = batch_sum if accumulated is None else accumulated + batch_sum
                count += encoded.size(0)

        if count > 0:
            mean_features = accumulated / count
            self.reference_features.copy_(mean_features)
            self.reference_initialized.fill_(True)
            print(f"[Attribution Baseline] Initialized with encoded mean features from Training Set {count} samples, "
                  f"mean range: [{mean_features.min().item():.4f}, "
                  f"{mean_features.max().item():.4f}]")
        else:
            print("[Attribution Baseline] Warning: dataset is empty，baseline remains all zeros")

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.dot_product_weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def compute_attribution_scores(self, features):
        batch_size = features.size(0)

        ref_features = self.reference_features.expand(batch_size, -1, -1, -1)
        delta_features = features - ref_features                         

        b, c, h, w = delta_features.shape
        features_flat = delta_features.view(b, c, -1)                           

                           
        weighted_features = torch.matmul(
            self.dot_product_weight, features_flat                                
        )
        attribution_scores = torch.norm(
            weighted_features, dim=1, keepdim=True                      
        ).view(b, 1, h, w)

        attribution_scores = torch.sigmoid(attribution_scores)                  
        attributed_features = delta_features * attribution_scores         

        return attributed_features, attribution_scores

    def forward(self, x):
        encoded_features = self.feature_encoder(x)

                                    
                                                   
                                                           
        if self.training and self.reference_initialized.item():
            with torch.no_grad():
                batch_mean = encoded_features.detach().mean(dim=0, keepdim=True)
                self.reference_features.mul_(1.0 - self.ema_momentum).add_(
                    self.ema_momentum * batch_mean
                )

        attributed_features, attribution_scores = self.compute_attribution_scores(encoded_features)
        key_features = self.output_projection(attributed_features)
        return key_features, attribution_scores


class AttributionGuidedAttention(nn.Module):

    def __init__(self, feature_dim=512, num_heads=8):
        super(AttributionGuidedAttention, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_proj   = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

                              
        self.attribution_fusion = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(0.15)

                                         
        num_groups = min(8, feature_dim // 4)
        self.norm = nn.GroupNorm(num_groups, feature_dim)

    def forward(self, dcn_features, attribution_key, attribution_scores):
        batch_size, _, H, W = dcn_features.size()
        num_spatial = H * W          

                                              
        query = self.query_proj(dcn_features)
        key   = self.key_proj(attribution_key)
        value = self.value_proj(dcn_features)

                                                
        query = query.view(batch_size, self.num_heads, self.head_dim, -1)
        key   = key.view(batch_size, self.num_heads, self.head_dim, -1)
        value = value.view(batch_size, self.num_heads, self.head_dim, -1)

                                           
        attention_weights = torch.matmul(
            query.transpose(-2, -1), key
        ) / (self.head_dim ** 0.5)

                           
        if attribution_scores.size(-1) != W or attribution_scores.size(-2) != H:
            attribution_scores = F.adaptive_avg_pool2d(attribution_scores, (H, W))
        attr_flat = attribution_scores.view(batch_size, 1, 1, -1).expand(
            -1, self.num_heads, num_spatial, -1
        )
        attention_weights = attention_weights + self.attribution_fusion * attr_flat

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

                                             
        attended = torch.matmul(value, attention_weights.transpose(-2, -1))
        attended = attended.view(batch_size, self.feature_dim, H, W)

                         
        output = self.output_proj(attended)
        output = output + dcn_features
        output = self.norm(output)

        return output


class AttributionGuidedDCNTransformer(nn.Module):

    def __init__(self, num_classes=2, feature_dim=512,
                 transformer_depth=16, transformer_heads=8):
        super(AttributionGuidedDCNTransformer, self).__init__()

                            
        self.attribution_network = DotProductAttributionNetwork(
            input_channels=42,
            hidden_dim=max(64, feature_dim),
            output_dim=feature_dim
        )

                               
        self.dcn = DCNv2(num_classes=num_classes, output_dim=feature_dim)

                        
        self.attribution_attention = AttributionGuidedAttention(
            feature_dim=feature_dim,
            num_heads=transformer_heads
        )

                                 
        encoder = Encoder(
            dim=feature_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=feature_dim // transformer_heads,
            mlp_dim=feature_dim * 2,                        
            dropout=0.15
        )
        self.transformer = DTransformer(
            image_size=9,
            patch_size=1,
            attn_layers=encoder,
            num_classes=num_classes,
            dropout=0.15
        )

        print(f"[OK] Attribution-guided DCN-Transformer model created (optimized version - for 314 samples）")
        print(f"- Feature dimension (d_model): {feature_dim}")
        print(f"- DCNv2 layers: 3 layers (reduced from 10)")
        print(f"- Attribution network layers: 3 layers (reduced from 6)")
        print(f"- Transformer layers: {transformer_depth}")
        print(f"- Attention heads: {transformer_heads}")
        print(f"- MLP dimension: {feature_dim * 2}")
        print(f"- Dropout: 0.15 (reduced from 0.25)")
        print(f"- Input resolution: 9x9 = 81 tokens")

    def forward(self, x, return_attribution=False, return_attention=False):
                                  
        attribution_key, attribution_scores = self.attribution_network(x)

                                          
        dcn_features = self.dcn(x, extract_features=True)                          

                          
        fused_features = self.attribution_attention(
            dcn_features, attribution_key, attribution_scores
        )

                               
        if return_attention:
            output, attention_maps = self.transformer(fused_features, return_attention=True)
            if return_attribution:
                return output, attribution_scores, attention_maps
            return output, attention_maps
        else:
            output = self.transformer(fused_features)
            if return_attribution:
                return output, attribution_scores
            return output

    def get_attribution_map(self, x):
        with torch.no_grad():
            _, attribution_scores = self.attribution_network(x)
        return attribution_scores


def create_attribution_guided_model(num_classes=2,
                                    feature_dim=512,
                                    transformer_depth=16,
                                    transformer_heads=8):
    return AttributionGuidedDCNTransformer(
        num_classes=num_classes,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_attribution_guided_model(
        num_classes=2, feature_dim=32,
        transformer_depth=1, transformer_heads=2
    ).to(device)

    test_input = torch.randn(4, 42, 9, 9).to(device)
    output, attr_scores = model(test_input, return_attribution=True)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attribution score shape: {attr_scores.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
