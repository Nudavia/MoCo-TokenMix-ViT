# -*-coding:utf-8-*-

import torch
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


class ViT(torch.nn.Module):
    def __init__(self,
                 input_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=4,
                 num_classes=10
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((input_size // patch_size) ** 2, 1, emb_dim))
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.fc = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.fc(features[0])
        return logits
