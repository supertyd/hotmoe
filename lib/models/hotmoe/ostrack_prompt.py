"""
Basic HotMoE model.
"""
import math
import os
from typing import List
from typing import Dict
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.hotmoe.vit_prompt import vit_base_patch16_224_prompt
from lib.models.hotmoe.vit_ce_prompt_vis import vit_base_patch16_224_ce_prompt_vis
from lib.models.hotmoe.vit_ce_prompt_nir import vit_base_patch16_224_ce_prompt_nir
from lib.models.hotmoe.vit_ce_prompt_rednir import vit_base_patch16_224_ce_prompt_rednir
from lib.models.hotmoe.vit_ce_prompt_all import vit_base_patch16_224_ce_prompt_all
from lib.models.hotmoe.vit import vit_small_patch16_224,vit_tiny_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class HotMoETrack(nn.Module):
    """ This is the base class for HotMoE """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                train_data_type="",
                ):
        x, aux_dict,loss = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out,loss

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_hotmoetrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and ('HotMoE' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prompt':
        backbone = vit_base_patch16_224_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               prompt_type=cfg.TRAIN.PROMPT.TYPE
                                               )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt_vis':
        backbone = vit_base_patch16_224_ce_prompt_vis(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           prompt_type=cfg.TRAIN.PROMPT.TYPE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt_nir':
        backbone = vit_base_patch16_224_ce_prompt_nir(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           prompt_type=cfg.TRAIN.PROMPT.TYPE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt_rednir':
        backbone = vit_base_patch16_224_ce_prompt_rednir(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           prompt_type=cfg.TRAIN.PROMPT.TYPE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt_all':
        backbone = vit_base_patch16_224_ce_prompt_all(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                         search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                                         template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                                         new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                                         prompt_type=cfg.TRAIN.PROMPT.TYPE
                                                         )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_small_patch16_224':
        backbone = vit_base_patch16_224_ce_prompt_all(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                         search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                                         template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                                         new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                                         prompt_type=cfg.TRAIN.PROMPT.TYPE
                                                         )
    elif cfg.MODEL.BACKBONE.TYPE == "vit_tiny_patch16_224":
        backbone = vit_small_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                         search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                                         template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                                         new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                                         prompt_type=cfg.TRAIN.PROMPT.TYPE
                                                         )


    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = HotMoETrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    # load pretrained weights
    if ('OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'HotMoE' in cfg.MODEL.PRETRAIN_FILE) and training:

        new_ckpt = {}
        # checkpoint_1 = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        if cfg.MODEL.PRETRAIN_FILE:
            ckpt_path = cfg.MODEL.PRETRAIN_FILE

            ckpt_1: Dict[str, torch.Tensor] = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')['net']
            for k, v in ckpt_1.items():
                if "cls_token" in k or "gate" in k:
                    continue
                elif 'patch_embed' in k or 'pos_embed' in k:
                    k = k.replace("patch_embed", "patch_embed")
                    new_ckpt[k] = v
                elif 'blocks.' in k:
                    k = k.replace("blocks.0.", "blocks.0.")
                    k = k.replace("blocks.1.", "blocks.1.")
                    k = k.replace("blocks.2.", "blocks.2.")
                    k = k.replace("blocks.3.", "blocks.3.")
                    k = k.replace("blocks.4.", "blocks.4.")
                    k = k.replace("blocks.5.", "blocks.5.")
                    k = k.replace("blocks.6.", "blocks.6.")
                    k = k.replace("blocks.7.", "blocks.7.")
                    k = k.replace("blocks.8.", "blocks.8.")
                    k = k.replace("blocks.9.", "blocks.9.")
                    k = k.replace("blocks.10.", "blocks.10.")
                    k = k.replace("blocks.11.", "blocks.11.")
                    new_ckpt[k] = v
                elif "backbone.norm" in k:
                    k = k.replace("backbone.norm", "backbone.norm")
                    new_ckpt[k] = v
                else:
                    new_ckpt[k] = v

            ckpt_2: Dict[str, torch.Tensor] = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')['net']
            for k, v in ckpt_2.items():
                if "cls_token" in k or "pos_embed" in k or "gate" in k or "box" in k:
                    continue
                elif 'patch_embed' in k or 'pos_embed' in k:
                    k = k.replace("patch_embed", "patch_embed_2")
                    new_ckpt[k] = v
                elif 'blocks.' in k:
                    k = k.replace("blocks.0.", "blocks_2.0.")
                    k = k.replace("blocks.1.", "blocks_2.1.")
                    k = k.replace("blocks.2.", "blocks_2.2.")
                    k = k.replace("blocks.3.", "blocks_2.3.")
                    k = k.replace("blocks.4.", "blocks_2.4.")
                    k = k.replace("blocks.5.", "blocks_2.5.")
                    k = k.replace("blocks.6.", "blocks_2.6.")
                    k = k.replace("blocks.7.", "blocks_2.7.")
                    k = k.replace("blocks.8.", "blocks_2.8.")
                    k = k.replace("blocks.9.", "blocks_2.9.")
                    k = k.replace("blocks.10.", "blocks_2.10.")
                    k = k.replace("blocks.11.", "blocks_2.11.")
                    new_ckpt[k] = v
                elif "backbone.norm" in k:
                    k = k.replace("backbone.norm", "backbone.norm_2")
                    new_ckpt[k] = v
                else:
                    new_ckpt[k] = v

            ckpt_3: Dict[str, torch.Tensor] = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')['net']
            for k, v in ckpt_3.items():
                if "cls_token" in k or "pos_embed" in k or "gate" in k or "box" in k:
                    continue
                elif 'patch_embed' in k or 'pos_embed' in k:
                    k = k.replace("patch_embed", "patch_embed_3")
                    new_ckpt[k] = v
                elif 'blocks.' in k:
                    k = k.replace("blocks.0.", "blocks_3.0.")
                    k = k.replace("blocks.1.", "blocks_3.1.")
                    k = k.replace("blocks.2.", "blocks_3.2.")
                    k = k.replace("blocks.3.", "blocks_3.3.")
                    k = k.replace("blocks.4.", "blocks_3.4.")
                    k = k.replace("blocks.5.", "blocks_3.5.")
                    k = k.replace("blocks.6.", "blocks_3.6.")
                    k = k.replace("blocks.7.", "blocks_3.7.")
                    k = k.replace("blocks.8.", "blocks_3.8.")
                    k = k.replace("blocks.9.", "blocks_3.9.")
                    k = k.replace("blocks.10.", "blocks_3.10.")
                    k = k.replace("blocks.11.", "blocks_3.11.")
                    new_ckpt[k] = v
                elif "backbone.norm" in k:
                    k = k.replace("backbone.norm", "backbone.norm_3")
                    new_ckpt[k] = v
                else:
                    new_ckpt[k] = v


            ckpt_4: Dict[str, torch.Tensor] = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')['net']
            for k, v in ckpt_4.items():
                if "cls_token" in k or "pos_embed" in k or "gate" in k or "box" in k:
                    continue
                elif 'patch_embed' in k or 'pos_embed' in k:
                    k = k.replace("patch_embed", "patch_embed_4")
                    new_ckpt[k] = v
                elif 'blocks.' in k:
                    k = k.replace("blocks.0.", "blocks_4.0.")
                    k = k.replace("blocks.1.", "blocks_4.1.")
                    k = k.replace("blocks.2.", "blocks_4.2.")
                    k = k.replace("blocks.3.", "blocks_4.3.")
                    k = k.replace("blocks.4.", "blocks_4.4.")
                    k = k.replace("blocks.5.", "blocks_4.5.")
                    k = k.replace("blocks.6.", "blocks_4.6.")
                    k = k.replace("blocks.7.", "blocks_4.7.")
                    k = k.replace("blocks.8.", "blocks_4.8.")
                    k = k.replace("blocks.9.", "blocks_4.9.")
                    k = k.replace("blocks.10.", "blocks_4.10.")
                    k = k.replace("blocks.11.", "blocks_4.11.")
                    new_ckpt[k] = v
                elif "backbone.norm" in k:
                    k = k.replace("backbone.norm", "backbone.norm_4")
                    new_ckpt[k] = v
                else:
                    new_ckpt[k] = v

            ckpt_5: Dict[str, torch.Tensor] = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')['net']
            for k, v in ckpt_5.items():
                if "cls_token" in k or "pos_embed" in k or "gate" in k or "box" in k:
                    continue
                elif 'patch_embed' in k or 'pos_embed' in k:
                    k = k.replace("patch_embed", "patch_embed_5")
                    new_ckpt[k] = v
                elif 'blocks.' in k:
                    k = k.replace("blocks.0.", "blocks_5.0.")
                    k = k.replace("blocks.1.", "blocks_5.1.")
                    k = k.replace("blocks.2.", "blocks_5.2.")
                    k = k.replace("blocks.3.", "blocks_5.3.")
                    k = k.replace("blocks.4.", "blocks_5.4.")
                    k = k.replace("blocks.5.", "blocks_5.5.")
                    k = k.replace("blocks.6.", "blocks_5.6.")
                    k = k.replace("blocks.7.", "blocks_5.7.")
                    k = k.replace("blocks.8.", "blocks_5.8.")
                    k = k.replace("blocks.9.", "blocks_5.9.")
                    k = k.replace("blocks.10.", "blocks_5.10.")
                    k = k.replace("blocks.11.", "blocks_5.11.")
                    new_ckpt[k] = v
                elif "backbone.norm" in k:
                    k = k.replace("backbone.norm", "backbone.norm_5")
                    new_ckpt[k] = v
                else:
                    new_ckpt[k] = v



        missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print(f"Load pretrained model missing_keys: {missing_keys}")
        print(f"Load pretrained model unexpected_keys: {unexpected_keys}")

    return model


if __name__ == '__main__':
    import importlib
    config_module = importlib.import_module("lib.config.%s.config" % 'vipt')
    cfg = config_module.cfg
    config_module.update_config_from_file('/home/lz/PycharmProjects/dev/ViPT/experiments/vipt/deep_all.yaml')
    net = build_hotmoetrack(cfg, training=True)
    # for n, p in net.named_parameters():
    #     print(n)
        # if "prompt" in n and cfg.TRAIN.PROMPT.DATATYPE in n:
        #     # p.requires_grad = True
        #     print(n)
        # else:
        #     # p.requires_grad = False
        #     pass