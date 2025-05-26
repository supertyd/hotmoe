from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from .utils import combine_tokens, token2feature, feature2token, Top2Gating
from lib.models.layers.patch_embed import PatchEmbed
from .vit import VisionTransformer
# from lib.models.vipt.vit_ce_prompt_vis import Prompt_block
import math

class vis_Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=768,
                 bottleneck=384,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, ):
        residual = x
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

# class GumbelSoftmax(nn.Module):
#     def __init__(self, eps=0.66667):
#         super(GumbelSoftmax, self).__init__()
#         self.eps = eps
#         self.sigmoid = nn.Sigmoid()
#
#     def gumbel_sample(self, template_tensor, eps=1e-8):
#         uniform_samples_tensor = template_tensor.clone().uniform_()
#         gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
#             1 - uniform_samples_tensor + eps)
#         return gumble_samples_tensor
#
#     def gumbel_softmax(self, logits):
#         """ Draw a sample from the Gumbel-Softmax distribution"""
#         gsamples = self.gumbel_sample(logits.data)
#         logits = logits + Variable(gsamples)
#         soft_samples = self.sigmoid(logits / self.eps)
#         return soft_samples, logits
#
#     def forward(self, logits):
#         if not self.training:
#             out_hard = (logits >= 0).float()
#             return out_hard, [None]*1
#         out_soft, prob_soft = self.gumbel_softmax(logits)
#         out_hard = ((out_soft == out_soft.max(dim=1, keepdim=True)[0]).float() - out_soft).detach() + out_soft
#         return out_hard, out_soft



class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x



class GlobalAveragePooling(torch.nn.Module):
    def forward(self, x):
        # 对最后一个维度进行平均池化
        return torch.mean(x, dim=-1)










class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output
class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerP(VisionTransformer):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_2 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_3 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_4 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_5 = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)



        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.gap = GlobalAveragePooling()
        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_type = prompt_type
        # self.adapter = MonaOp(in_features=768)
        # self.router = nn.Parameter(torch.randn(1600, 5))
        self.router = nn.Linear(1280, 5, bias=False)
        '''prompt parameters'''
        # if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
        #     adapter = []
        #     block_nums = depth if self.prompt_type == 'vipt_deep' else 1
        #     for i in range(block_nums):
        #         adapter.append(vis_Adapter())
        #     self.adapter = nn.Sequential(*adapter)

            # prompt_norms = []
            # for i in range(block_nums):
            #     prompt_norms.append(embed_dim)
            # self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.blocks_2 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_2 = norm_layer(embed_dim)

        self.blocks_3 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_3 = norm_layer(embed_dim)

        self.blocks_4 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_4 = norm_layer(embed_dim)

        self.blocks_5 = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm_5 = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]


        # rgb_img
        #x_rgb = x[:, :3, :, :]
        #z_rgb = z[:, :3, :, :]
        # depth_img
        x_vis = x[:, 3:, :, :]
        z_vis = z[:, 3:, :, :]
        # overwrite x & z
        #x, z = x_rgb, z_rgb
        x_hsi_1 = self.patch_embed(x_vis[:, [0, 5, 10], :, :])
        x_hsi_2 = self.patch_embed_2(x_vis[:, [1, 6, 11], :, :])
        x_hsi_3 = self.patch_embed_3(x_vis[:, [2, 7, 13], :, :])
        x_hsi_4 = self.patch_embed_4(x_vis[:, [3, 8, 14], :, :])
        x_hsi_5 = self.patch_embed_5(x_vis[:, [4, 9, 15], :, :])

        z_hsi_1 = self.patch_embed(z_vis[:, [0, 5, 10], :, :])
        z_hsi_2 = self.patch_embed_2(z_vis[:, [1, 6, 11], :, :])
        z_hsi_3 = self.patch_embed_3(z_vis[:, [2, 7, 13], :, :])
        z_hsi_4 = self.patch_embed_4(z_vis[:, [3, 8, 14], :, :])
        z_hsi_5 = self.patch_embed_5(z_vis[:, [4, 9, 15], :, :])
        # x_hsi_1 = self.patch_embed(x_vis[:, [0, 1, 2], :, :])
        # x_hsi_2 = self.patch_embed_2(x_vis[:, [3, 4, 5], :, :])
        # x_hsi_3 = self.patch_embed_3(x_vis[:, [6, 7, 8], :, :])
        # x_hsi_4 = self.patch_embed_4(x_vis[:, [9, 10, 11], :, :])
        # x_hsi_5 = self.patch_embed_5(x_vis[:, [13, 14, 15], :, :])
        #
        # z_hsi_1 = self.patch_embed(z_vis[:, [0, 1, 2], :, :])
        # z_hsi_2 = self.patch_embed_2(z_vis[:, [3, 4, 5], :, :])
        # z_hsi_3 = self.patch_embed_3(z_vis[:, [6, 7, 8], :, :])
        # z_hsi_4 = self.patch_embed_4(z_vis[:, [9, 10, 11], :, :])
        # z_hsi_5 = self.patch_embed_5(z_vis[:, [13, 14, 15], :, :])
        # x_hsi_1 = self.patch_embed(x_vis[:, [0, 6, 11], :, :])
        # x_hsi_2 = self.patch_embed_2(x_vis[:, [1, 7, 13], :, :])
        # x_hsi_3 = self.patch_embed_3(x_vis[:, [2, 5, 14], :, :])
        # x_hsi_4 = self.patch_embed_4(x_vis[:, [3, 8, 15], :, :])
        # x_hsi_5 = self.patch_embed_5(x_vis[:, [4, 9, 10], :, :])
        #
        # z_hsi_1 = self.patch_embed(z_vis[:, [0, 6, 15], :, :])
        # z_hsi_2 = self.patch_embed_2(z_vis[:, [1, 7, 13], :, :])
        # z_hsi_3 = self.patch_embed_3(z_vis[:, [2, 5, 14], :, :])
        # z_hsi_4 = self.patch_embed_4(z_vis[:, [3, 8, 15], :, :])
        # z_hsi_5 = self.patch_embed_5(z_vis[:, [4, 9, 10], :, :])


        # x = (x_hsi_1 + x_hsi_2 + x_hsi_3 + x_hsi_4 + x_hsi_5) / 6
        # z = (z_hsi_1 + z_hsi_2 + z_hsi_3 + z_hsi_4 + z_hsi_5) / 6


        # hsi_1_score = self.gap(torch.concat((z_hsi_1, x_hsi_1), dim=1))
        # hsi_2_score = self.gap(torch.concat((z_hsi_2, x_hsi_2), dim=1))
        # hsi_3_score = self.gap(torch.concat((z_hsi_3, x_hsi_3), dim=1))
        # hsi_4_score = self.gap(torch.concat((z_hsi_4, x_hsi_4), dim=1))
        # hsi_5_score = self.gap(torch.concat((z_hsi_5, x_hsi_5), dim=1))

        x = (x_hsi_1+x_hsi_2+x_hsi_3+x_hsi_4+x_hsi_5)/5
        z = (z_hsi_1+z_hsi_2+z_hsi_3+z_hsi_4+z_hsi_5)/5
        z += self.pos_embed_z
        x += self.pos_embed_x
        x_prompted = combine_tokens(z, x, mode=self.cat_mode)

        hsi_1_score = self.gap(x_hsi_1)
        hsi_2_score = self.gap(x_hsi_2)
        hsi_3_score = self.gap(x_hsi_3)
        hsi_4_score = self.gap(x_hsi_4)
        hsi_5_score = self.gap(x_hsi_5)

        hsi_score = torch.concat((hsi_1_score, hsi_2_score, hsi_3_score, hsi_4_score, hsi_5_score), dim=1)

        # hsi_score = self.gap(x)
        hsi_score = F.softmax(hsi_score, dim=-1)



        router = self.router(hsi_score)
        router = router.softmax(dim=-1)
        density_1_proxy = router

        values, indices = torch.topk(router, k=1, dim=-1)
        mask_1 = F.one_hot(indices, 5).float()
        mask_1 = mask_1.squeeze(dim=1)

        # router_without_top1 = router*(1.-mask_1)
        #
        # values_2, indices_2 = torch.topk(router_without_top1, k=1, dim=-1)
        # mask_2 = F.one_hot(indices_2, 5).float()
        # mask_2 = mask_2.squeeze(dim=1)

        loss = (density_1_proxy*mask_1.squeeze(dim=1)).mean()*25

        x_prompted = self.pos_drop(x_prompted)





        for j in range(B):
            if mask_1[j][0] == 1:
                for i, blk in enumerate(self.blocks):
                    x_prompted[[j]] = blk(x_prompted[[j]])
                x_prompted[[j]] = self.norm(x_prompted[[j]])
                # print(1)
            if mask_1[j][1] == 1:
                for i, blk in enumerate(self.blocks_2):
                    x_prompted[[j]] = blk(x_prompted[[j]])
                x_prompted[[j]] = self.norm_2(x_prompted[[j]])
                # print(2)
            if mask_1[j][2] == 1:
                for i, blk in enumerate(self.blocks_3):
                    x_prompted[[j]] = blk(x_prompted[[j]])
                x_prompted[[j]] = self.norm_3(x_prompted[[j]])
                # print(3)
            if mask_1[j][3] == 1:
                for i, blk in enumerate(self.blocks_4):
                    x_prompted[[j]] = blk(x_prompted[[j]])
                x_prompted[[j]] = self.norm_4(x_prompted[[j]])
                # print(4)
            if mask_1[j][4] == 1:
                for i, blk in enumerate(self.blocks_5):
                    x_prompted[[j]] = blk(x_prompted[[j]])
                x_prompted[[j]] = self.norm_5(x_prompted[[j]])
                #print(5)
        #
        # # for i, blk in enumerate(self.blocks_2):
        # #     x_prompted = blk(x_prompted)
        # # x_prompted = self.norm_2(x_prompted)
        #
        # for j in range(B):
        #     if mask_2[j][0] == 1:
        #         for i, blk in enumerate(self.blocks):
        #             x_prompted_2[[j]] = blk(x_prompted_2[[j]])
        #         #x_prompted_2[[j]] = self.norm(x_prompted_2[[j]])
        #         # print(1)
        #     if mask_2[j][1] == 1:
        #         for i, blk in enumerate(self.blocks_2):
        #             x_prompted_2[[j]] = blk(x_prompted_2[[j]])
        #         #x_prompted_2[[j]] = self.norm_2(x_prompted_2[[j]])
        #         # print(2)
        #     if mask_2[j][2] == 1:
        #         for i, blk in enumerate(self.blocks_3):
        #             x_prompted_2[[j]] = blk(x_prompted_2[[j]])
        #         #x_prompted_2[[j]] = self.norm_3(x_prompted_2[[j]])
        #         # print(3)
        #     if mask_2[j][3] == 1:
        #         for i, blk in enumerate(self.blocks_4):
        #             x_prompted_2[[j]] = blk(x_prompted_2[[j]])
        #         #x_prompted_2[[j]] = self.norm_4(x_prompted_2[[j]])
        #         # print(4)
        #     if mask_2[j][4] == 1:
        #         for i, blk in enumerate(self.blocks_5):
        #             x_prompted_2[[j]] = blk(x_prompted_2[[j]])
        #         #x_prompted_2[[j]] = self.norm_5(x_prompted_2[[j]])

        # x_prompted = self.norm(x_prompted)

        '''input prompt: 
        by adding to rgb tokens
        '''
        # if self.prompt_type in ['vipt_shaw', 'vipt_deep']:
        #     z_feat = token2feature(self.prompt_norms[0](z))
        #     x_feat = token2feature(self.prompt_norms[0](x))
        #     z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
        #     x_dte_feat = token2feature(self.prompt_norms[0](x_dte))
        #     z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
        #     x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
        #     z_feat = self.prompt_blocks[0](z_feat)
        #     x_feat = self.prompt_blocks[0](x_feat)
        #     z_dte = feature2token(z_feat)
        #     x_dte = feature2token(x_feat)
        #     z_prompted, x_prompted = z_dte, x_dte
        #
        #     x = x + x_dte
        #     z = z + z_dte
        # else:
        #     x = x + x_dte
        #     z = z + z_dte

        # attention mask handling
        # B, H, W




        aux_dict = {"attn": None}
        return x_prompted, aux_dict, loss

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict, loss = self.forward_features(z, x)

        return x, aux_dict, loss


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerP(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack without CE from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_prompt(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

# if __name__ == "__main__":
#     HGAT = VisionTransformerP()
#     z = torch.randn(3, 128, 128)
#     x = torch.randn(3, 256, 256)
#     x, aux_dict = HGAT