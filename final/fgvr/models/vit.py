import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange

from pretrained_vit.model import ViT
from pretrained_vit.configs import PRETRAINED_CONFIGS, ViTConfigExtended


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()

        def_config = PRETRAINED_CONFIGS['{}'.format(args.model)]['config']
        cfg = ViTConfigExtended(**def_config)
        cfg.num_classes = args.n_cls
        cfg.image_size = args.image_size

        self.model = ViT(cfg, name=args.model, pretrained=args.pretrained,
                         load_fc_layer=not(args.ifa), ret_interm_repr=args.ifa)

        if args.token_gather == 'gap_pre':
            self.pool_pre = nn.Sequential(
                Rearrange('b s c -> b c s'),
                nn.AdaptiveAvgPool1d(1),
                Rearrange('b c 1 -> b c'),
            )

        if args.ifa:
            if args.token_gather in ['cls', 'gap_pre']:
                self.pooled = True
                self.ifa_head = nn.Sequential(
                    nn.Linear(cfg.num_hidden_layers, 1),
                    Rearrange(' b c 1 -> b c'),
                    nn.ReLU(),
                    nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
                    nn.Linear(cfg.hidden_size, cfg.num_classes)
                )
            else:
                self.ifa_head = nn.Sequential(
                    nn.Linear(cfg.num_hidden_layers, 1),
                    Rearrange(' b s c 1 -> b s c'),
                    nn.ReLU(),
                    Rearrange('b s c -> b c s'),
                    nn.AdaptiveAvgPool1d(1),
                    Rearrange('b c 1 -> b c'),
                    nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
                    nn.Linear(cfg.hidden_size, cfg.num_classes)
                )

    def forward(self, x):
        if hasattr(self, 'ifa_head'):
            _, interm_features = self.model(x)
        else:
            logits = self.model(x)

        if hasattr(self, 'ifa_head'):
            if hasattr(self, 'pool_pre'):
                interm_features = [self.pool_pre(ft) for ft in interm_features]
            elif hasattr(self, 'pooled'):
                interm_features = [ft[:, 0] for ft in interm_features]

            interm_features = torch.stack(interm_features, dim=-1)
            logits = self.ifa_head(interm_features)

        return logits
