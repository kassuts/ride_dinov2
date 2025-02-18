import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from functools import partial
from typing import Union, Sequence, Tuple

from model.dinov2.layers.layers.attention import MemEffAttention
from model.dinov2.layers.layers.block import Block
from model.dinov2.vision_transformer import DinoVisionTransformer

class RIDEDinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        dropout=None,
        num_experts=3,
        returns_feat=False,
        expert_start_layer=10,  # Start expert branches near end
        pretrained_path=None,  # Path to pretrained DINOv2 weights
        **kwargs
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.returns_feat = returns_feat
        self.use_dropout = True if dropout else False
        self.expert_start_layer = expert_start_layer
        
        # Create base transformer (will be loaded with pretrained weights and frozen)
        self.base_transformer = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=expert_start_layer,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            drop_path_rate=0.0,  # Set to 0 since we're freezing these layers
            **kwargs
        )
        
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)
            
        # Create expert branches
        expert_depth = depth - expert_start_layer
        
        # Initialize expert branches with random weights
        self.expert_blocks = nn.ModuleList([
            nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,  # Keep dropout in expert branches
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    attn_class=MemEffAttention
                ) for _ in range(expert_depth)
            ]) for _ in range(num_experts)
        ])
        
        self.expert_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(num_experts)
        ])
        
        # Simple linear heads for experts
        self.expert_heads = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(num_experts)
        ])
            
        # Initialize with pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
            
        # Freeze base transformer
        self.freeze_base_transformer()
            
    def load_pretrained_weights(self, pretrained_path):
        """Load pretrained DINOv2 weights and handle key matching"""
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Filter state dict to only include base transformer keys
        base_state_dict = {}
        for key, value in state_dict.items():
            # Only take keys up to expert_start_layer
            if 'blocks.' in key:
                layer_num = int(key.split('blocks.')[1].split('.')[0])
                if layer_num >= self.expert_start_layer:
                    continue
            base_state_dict[key] = value
            
        # Load weights into base transformer
        self.base_transformer.load_state_dict(base_state_dict, strict=False)
        print(f"Loaded pretrained weights for base transformer (layers 0-{self.expert_start_layer})")
            
    def freeze_base_transformer(self):
        """Freeze all parameters in the base transformer"""
        for param in self.base_transformer.parameters():
            param.requires_grad = False
        self.base_transformer.eval()  # Set to eval mode
        print("Base transformer frozen")
        
    def _separate_part(self, x, ind):
        # Process through expert blocks (with gradients)
        for block in self.expert_blocks[ind]:
            x = block(x)
            
        x = self.expert_norms[ind](x)
        x = x[:, 0]  # CLS token
        
        if self.use_dropout:
            x = self.dropout(x)
            
        self.feat.append(x)
        x = self.expert_heads[ind](x)
        return x
        
    def forward(self, x):
        with autocast():
            with torch.no_grad():  # No gradients for base transformer
                base_features = self.base_transformer.prepare_tokens_with_masks(x)
                for blk in self.base_transformer.blocks:
                    base_features = blk(base_features)
            
            # Process through expert branches (with gradients)
            outs = []
            self.feat = []
            for ind in range(self.num_experts):
                expert_out = self._separate_part(base_features, ind)
                outs.append(expert_out)
                
            final_out = torch.stack(outs, dim=1).mean(dim=1)
            
        if self.returns_feat:
            return {
                "output": final_out,
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out

def ride_dino_small(pretrained_path=None, num_experts=3, num_classes=1000, **kwargs):
    model = RIDEDinoVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_experts=num_experts,
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        **kwargs
    )
    return model
