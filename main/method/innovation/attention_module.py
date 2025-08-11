import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def Sequence_2_Image(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Patch_Embedding(nn.Module):

    def __init__(self, in_cdim=3, out_cdim=768):
        super().__init__()
        self.proj = nn.Linear(in_cdim, out_cdim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_cdim_H, in_cdim_i, heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads

        self.scale = in_cdim_H**-0.5

        self.to_q = nn.Linear(in_cdim_H, inner_dim, bias=False)
        self.to_kv = nn.Linear(in_cdim_i, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, in_cdim_H), nn.Dropout(dropout))

    def forward(self, features_H, features_i):
        H = self.heads
        SCALE = self.scale

        q = self.to_q(features_H)
        kv = self.to_kv(features_i).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=H)
        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=H), kv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * SCALE
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, in_cdim_H, in_cdim_i, heads, head_dim, dropout, mlp_dim, depth):
        super().__init__()

        self.attention_norm_layer_H = nn.LayerNorm(in_cdim_H)
        self.attention_norm_layer_i = nn.LayerNorm(in_cdim_i)
        self.attention_layer = Attention(in_cdim_H, in_cdim_i, heads=heads, head_dim=head_dim, dropout=dropout)

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_cdim_H),
            FeedForward(in_cdim_H, mlp_dim, dropout=dropout),
        )

    def forward(self, features_H, features_i):
        residual_features_H = features_H
        # residual_features_i = features_i

        features_H = self.attention_norm_layer_H(features_H)
        features_i = self.attention_norm_layer_i(features_i)

        attention_features = self.attention_layer(features_H, features_i) + residual_features_H

        outs = self.mlp_layer(attention_features) + attention_features
        return outs


class Feature_Pyramid_Network(nn.Module):
    def __init__(self, in_cdim_list, out_cdim=512):
        super(Feature_Pyramid_Network, self).__init__()
        # self.max_pool1 = nn.AdaptiveMaxPool2d((16, 8))
        cdim_1, cdim_2, cdim_3, cdim_4 = in_cdim_list

        self.patch_embedding_1 = Patch_Embedding(in_cdim=cdim_1, out_cdim=cdim_1)
        self.attention_1 = Transformer(in_cdim_H=cdim_4, in_cdim_i=cdim_1, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.patch_embedding_2 = Patch_Embedding(in_cdim=cdim_2, out_cdim=cdim_2)
        self.attention_2 = Transformer(in_cdim_H=cdim_4, in_cdim_i=cdim_2, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.patch_embedding_3 = Patch_Embedding(in_cdim=cdim_3, out_cdim=cdim_3)
        self.attention_3 = Transformer(in_cdim_H=cdim_4, in_cdim_i=cdim_3, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.patch_embedding_4 = Patch_Embedding(in_cdim=cdim_4, out_cdim=cdim_4)
        self.attention_4 = Transformer(in_cdim_H=cdim_4, in_cdim_i=cdim_4, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.to_outs = nn.Sequential(
            nn.Conv2d(cdim_4 * 4, cdim_4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cdim_4),
            nn.GELU(),
            nn.Conv2d(cdim_4, cdim_4, 1, 1, 0, bias=False),
        )

    def forward(self, input_list):
        # feature_maps_1: (B, C, H, W) -> (B, 256, 64, 32)
        # feature_maps_2: (B, 2C, H/2, W/2) -> (B, 512, 32, 16)
        # feature_maps_3: (B, 4C, H/4, W/4) -> (B, 1024, 16, 8)
        # feature_maps_4: (B, 8C, H/4, W/4) -> (B, 2048, 16, 8)
        feature_maps_1, feature_maps_2, feature_maps_3, feature_maps_4 = input_list

        token_4 = self.patch_embedding_4(feature_maps_4)

        feature_maps_1 = F.interpolate(feature_maps_1, size=(16, 8), mode="bilinear")
        token_1 = self.patch_embedding_1(feature_maps_1)
        token_1 = self.attention_1(token_4, token_1)
        feature_maps_1 = Sequence_2_Image(token_1, h=16, w=8)

        feature_maps_2 = F.interpolate(feature_maps_2, size=(16, 8), mode="bilinear")
        token_2 = self.patch_embedding_2(feature_maps_2)
        token_2 = self.attention_2(token_4, token_2)
        feature_maps_2 = Sequence_2_Image(token_2, h=16, w=8)

        token_3 = self.patch_embedding_3(feature_maps_3)
        token_3 = self.attention_3(token_4, token_3)
        feature_maps_3 = Sequence_2_Image(token_3, h=16, w=8)

        outs = self.to_outs(torch.cat([feature_maps_1, feature_maps_2, feature_maps_3, feature_maps_4], dim=1))

        return outs


if __name__ == "__main__":
    B = 64

    feature_maps_1 = torch.randn(B, 256, 64, 32)
    feature_maps_2 = torch.randn(B, 512, 32, 16)
    feature_maps_3 = torch.randn(B, 1024, 16, 8)
    inter_feature_maps = [feature_maps_1, feature_maps_2, feature_maps_3]

    feature_maps_4 = torch.randn(B, 2048, 16, 8)
    attension_feature_maps = inter_feature_maps + [feature_maps_4]

    model = Feature_Pyramid_Network(in_cdim_list=[256, 512, 1024, 2048])
    print(model)

    outs = model(attension_feature_maps)
    print(outs.shape)
