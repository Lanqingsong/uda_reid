from timm.models.swin_transformer_v2 import SwinTransformerV2,build_model_with_cfg,checkpoint_filter_fn
import torch.nn as nn
from torch.nn import functional as F

class Swim(SwinTransformerV2):

    def __init__(self,num_ids,  **kwargs):
        super().__init__(**kwargs)

        self.classify_head = nn.Linear(self.num_classes, num_ids)

        # split image into non-overlapping patches

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        prob = self.classify_head(x)
        x = F.normalize(x)

        if self.training is False:
            return x
        return prob, x

def _create_swin_transformer_v2(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        Swim,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


def swinv2_tiny_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        img_size=(256, 128),window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_tiny_window16_256', pretrained=pretrained, **model_kwargs)



def swinv2_tiny_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_tiny_window8_256', pretrained=pretrained, **model_kwargs)

def swinv2_small_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_small_window16_256', pretrained=pretrained, **model_kwargs)

def swinv2_small_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer_v2('swinv2_small_window8_256', pretrained=pretrained, **model_kwargs)

def swinv2_base_window16_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window16_256', pretrained=pretrained, **model_kwargs)

def swinv2_base_window8_256(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=8, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window8_256', pretrained=pretrained, **model_kwargs)

def swinv2_base_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer_v2('swinv2_base_window12_192_22k', pretrained=pretrained, **model_kwargs)

def swinv2_base_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_base_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)

def swinv2_base_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_base_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)

def swinv2_large_window12_192_22k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return _create_swin_transformer_v2('swinv2_large_window12_192_22k', pretrained=pretrained, **model_kwargs)

def swinv2_large_window12to16_192to256_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_large_window12to16_192to256_22kft1k', pretrained=pretrained, **model_kwargs)

def swinv2_large_window12to24_192to384_22kft1k(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6), **kwargs)
    return _create_swin_transformer_v2(
        'swinv2_large_window12to24_192to384_22kft1k', pretrained=pretrained, **model_kwargs)
