# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, bootstrap_k=1, mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # bootstrap
        self.bootstrap_k = bootstrap_k

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.down2feature = nn.Linear(int(img_size/patch_size)**2+1, int((img_size/patch_size)**2*(1-self.mask_ratio)+1), bias=True)
        self.decoder_feature = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # print(f"pos1 {x.shape}")
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print(f"pos2 {x.shape}") # [256, 64, 192]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print(f"pos3 {x.shape}") # [256, 16, 192]
        # print(f"mask.shape: {mask.shape}") # [256, 64]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(f"pos4 {x.shape}")

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            # print(f"pos5 {x.shape}") # [256, 17, 192]
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) 
        # print('pos 0 : x.shape', x.shape) # [256, 17, 96]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        # print('pos 1 : x.shape', x.shape) # [256, 65, 96]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            # print('pos 2 : x.shape', x.shape) # [256, 65, 96]
        x = self.decoder_norm(x)
        # print('pos 3 : x.shape', x.shape) # [256, 65, 96]

        # predictor projection
        if self.bootstrap_k <= 1: # 0 or 1
            x = self.decoder_pred(x) # for original image pixel reconstruction
            # print('pos 4 : x.shape', x.shape) # [256, 65, 48]
            # remove cls token
            x = x[:, 1:, :]
        else: # self.bootstrap_k > 1 
            x = x.transpose(1, 2)
            x = self.down2feature(x)
            x = x.transpose(1, 2)
            x = self.decoder_feature(x)
            # print(f"pos2 {x.shape}") # [N, 17, 192]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        # print("loss.shape: ", loss.shape)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def Bforward_loss(self, target, pred): # to be test
        """
        target: [N, L*mask_ratio, embed_dim] # last encoder feature
        pred: [N, L*mask_ratio, embed_dim] # the reconstruction encoder feature
        """
        mse_loss_fn = nn.MSELoss()
        mse_loss = mse_loss_fn(target, pred)
        return mse_loss

    def forward(self, imgs, last_model=None, mask_ratio=0.75):
        # print("imgs.shape: ", imgs.shape) # N, C, H, W : [256, 3, 32, 32]
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # print("latent.shape: ", latent.shape) # N, (patch_num)*mask_ratio+1, embed_dim : [256, 17, 192]
        # print("mask.shape: ", mask.shape) # [256, 64] 64 = (32/4)**2 is the number of tokens
        
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3] or bootstrap_k>1 :[N, L*mask_ratio, embed_dim]
        # print("pred.shape: ", pred.shape) # [256, 64, 48]
        if self.bootstrap_k <= 1:
            loss = self.forward_loss(imgs, pred, mask)
        else: # self.bootstrap_k > 1
            # print("pos 1:", last_model(imgs).shape) # [256, 65, 96]
            
            # 记录执行前的显存使用情况
            # memory_allocated_before = torch.cuda.memory_allocated()  # 已分配的显存
            # memory_reserved_before = torch.cuda.memory_reserved()  # 保留的显存
            
            # # 执行相关操作前的显存使用情况
            # print(f"Before Bforward_loss:")
            # print(f"Memory Allocated: {memory_allocated_before / (1024 ** 2):.2f} MB")
            # print(f"Memory Reserved: {memory_reserved_before / (1024 ** 2):.2f} MB")
            
            feature = last_model(imgs, mask)
            # print("feature.shape ", feature.shape)
            # print("pred.shape ", pred.shape)
            
            loss = self.Bforward_loss(last_model(imgs, mask), pred)
            
            # # 记录执行后的显存使用情况
            # memory_allocated_after = torch.cuda.memory_allocated()  # 已分配的显存
            # memory_reserved_after = torch.cuda.memory_reserved()  # 保留的显存
            
            # # 执行相关操作后的显存使用情况
            # print(f"After Bforward_loss:")
            # print(f"Memory Allocated: {memory_allocated_after / (1024 ** 2):.2f} MB")
            # print(f"Memory Reserved: {memory_reserved_after / (1024 ** 2):.2f} MB")
            
            # # 可以计算显存使用的变化
            # memory_allocated_diff = memory_allocated_after - memory_allocated_before
            # memory_reserved_diff = memory_reserved_after - memory_reserved_before
            
            # print(f"Change in Memory Allocated: {memory_allocated_diff / (1024 ** 2):.2f} MB")
            # print(f"Change in Memory Reserved: {memory_reserved_diff / (1024 ** 2):.2f} MB")
        return loss, pred, mask


class ViTencoder4feature(nn.Module):
    """ VisionTransformer backbone, only need encoder features as the targets.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 bootstrap_k=1, feature_depth=8, mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # bootstrap
        self.bootstrap_k = bootstrap_k
        self.feature_depth = feature_depth

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # self.down2feature = nn.Linear(patch_size**2+1, patch_size**2*self.mask_ratio+1, bias=True)
        # self.decoder_feature = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        # --------------------------------------------------------------------------

        # self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def masking(self, x, mask): # to be test
        '''
        x: N, L, D
        mask: N, L
        return x_masked: N, L*mask_ratio, D
        '''
        # mask the x use the same mask as this epoch
        mask = mask.bool()
        mask = mask.unsqueeze(-1).expand_as(x)
        # print((~mask).shape)
        x_masked = x[~mask].reshape(x.shape[0], -1, x.shape[2])
        # print("x_masked:", x_masked.shape)
        return x_masked
        

    def forward_encoder(self, x, mask):

        # 执行相关操作前的显存使用情况
        # print(f"pos1:")
        # print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print(f"pos0 {x.shape}")

        # masking: length -> length * mask_ratio
        x = self.masking(x, mask)
        # print(f"pos1 {x.shape}")

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 执行相关操作前的显存使用情况
        # print(f"pos2:")
        # print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

        # apply Transformer blocks
        # print(f"pos2 {x.shape}")
        for depth_i, blk in enumerate(self.blocks):
            x = blk(x)
            # print(f"after Transformer blocks {depth_i+1}:")
            # print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            # print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
            if(depth_i+1 == self.feature_depth):
                break
        
        # print(f"pos3:")
        # print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        # print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        
        x = self.norm(x)

        return x

    def forward(self, img, mask):
        encoder_feature = self.forward_encoder(img, mask)
        return encoder_feature

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_Deit_tiny_patch4_dec128d8b_32(**kwargs): # img_size 32
    model = MaskedAutoencoderViT(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=96, decoder_depth=8, decoder_num_heads=3, # how to set the decoder?
        # debug: decoder_embed_dim=128 -> 96 because it % num_head must == 0
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def Bmae_Deit_tiny_patch4_dec128d8b_32(**kwargs): # for bootstrapped
    model = ViTencoder4feature(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=96, decoder_depth=8, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# # The original Deit-tiny is as follow:
# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

mae_Deit_tiny_patch4 = mae_Deit_tiny_patch4_dec128d8b_32 # decoder: 128 dim, 8 blocks img_size:32

Bmae_Deit_tiny_patch4 = Bmae_Deit_tiny_patch4_dec128d8b_32