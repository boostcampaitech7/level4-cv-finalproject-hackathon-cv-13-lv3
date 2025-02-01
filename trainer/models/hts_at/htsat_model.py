import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        return x

class HierarchicalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(HierarchicalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class TokenSemanticModule(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TokenSemanticModule, self).__init__()
        self.token_semantic = nn.Conv1d(
            in_channels=embed_dim, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        x = self.token_semantic(x)
        return x.transpose(1, 2)  # (B, C, N) -> (B, N, C)

class HTSATModel(nn.Module):
    def __init__(self, config):
        super(HTSATModel, self).__init__()
        self.patch_embed = PatchEmbedding(
            in_channels=1,  # Mono audio
            embed_dim=config["hidden_size"],
            patch_size=(4, 4)
        )
        self.transformer = HierarchicalTransformer(
            embed_dim=config["hidden_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"]
        )
        self.token_semantic = TokenSemanticModule(
            embed_dim=config["hidden_size"],
            num_classes=config["num_classes"]
        )

    def forward(self, x):
        x = self.patch_embed(x)  # Patch Embedding
        x = self.transformer(x)  # Transformer Encoding
        x = self.token_semantic(x)  # Token-Semantic Module
        return x
