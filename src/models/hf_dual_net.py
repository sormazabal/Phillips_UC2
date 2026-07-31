import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, AutoModel, AutoConfig


class SegFormerMLPDecoder(nn.Module):
    """
    MLP Decoder Head for Segmentation inspired by SegFormer architecture.
    Fuses multi-scale feature maps from the backbone.
    """
    def __init__(self, in_channels_list=[64, 128, 320, 512], embedding_dim=256, num_classes=1):
        super().__init__()
        self.linear_c1 = nn.Conv2d(in_channels_list[0], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels_list[1], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels_list[2], embedding_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(in_channels_list[3], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, hidden_states):
        # hidden_states: list of 4 feature maps [B, C_i, H_i, W_i]
        c1, c2, c3, c4 = hidden_states

        target_size = c1.shape[2:]

        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=target_size, mode='bilinear', align_corners=False)
        _c4 = F.interpolate(self.linear_c4(c4), size=target_size, mode='bilinear', align_corners=False)

        fused = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        logits = self.classifier(fused)
        return logits  # [B, num_classes, H/4, W/4]


class DetectionHead(nn.Module):
    """
    Lightweight Anchor-Free Detection Head for Stenosis Bounding Box Regression & Classification.
    """
    def __init__(self, in_channels=512, num_classes=1, num_boxes_per_grid=1):
        super().__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling to fixed grid size, e.g. 7x7 grid
        self.grid_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Predicts per grid cell: 4 bbox coords [x1, y1, x2, y2] + 1 confidence + num_classes
        out_dim = num_boxes_per_grid * (4 + 1 + num_classes)
        self.head = nn.Conv2d(128, out_dim, kernel_size=1)

    def forward(self, feature_map):
        x = self.conv_reduce(feature_map)
        x = self.grid_pool(x)
        out = self.head(x)  # [B, out_dim, 7, 7]
        return out


class HFDualNet(nn.Module):
    """
    Dual-Head Network leveraging a Hugging Face Pre-trained Backbone (SegFormer / ViT).
    - Head 1: Binary Vessel Segmentation.
    - Head 2: Stenosis Bounding Box Localization & Classification.
    """
    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b3",
        freeze_backbone: bool = True,
        num_seg_classes: int = 1,
        num_det_classes: int = 1,
        backbone_kwargs: dict = None
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_det_classes = num_det_classes
        self._is_timm = backbone_name.startswith("timm:")

        # Instantiate pre-trained backbone
        if self._is_timm:
            import timm
            # ponytail: features_only gives the 4 NCHW pyramid maps the decoder wants.
            # out_indices pinned to the last 4 in case a model emits more stages.
            self.backbone = timm.create_model(
                backbone_name[len("timm:"):],
                pretrained=True,
                features_only=True,
                out_indices=(-4, -3, -2, -1),
                **(backbone_kwargs or {}),
            )
            in_channels_list = self.backbone.feature_info.channels()
        else:
            try:
                self.backbone = SegformerModel.from_pretrained(backbone_name)
                in_channels_list = self.backbone.config.hidden_sizes
            except Exception:
                # Fallback for general HF vision backbones
                config = AutoConfig.from_pretrained(backbone_name, output_hidden_states=True)
                self.backbone = AutoModel.from_pretrained(backbone_name, config=config)
                in_channels_list = getattr(config, "hidden_sizes", [64, 128, 320, 512])

        # Segmentation Head
        self.seg_head = SegFormerMLPDecoder(
            in_channels_list=in_channels_list,
            embedding_dim=256,
            num_classes=num_seg_classes
        )

        # Detection Head
        self.det_head = DetectionHead(
            in_channels=in_channels_list[-1],
            num_classes=num_det_classes
        )

        if freeze_backbone:
            self.freeze_backbone(True)
        elif hasattr(self.backbone, "gradient_checkpointing_enable"):
            # ponytail: unfrozen encoder needs backward through every activation;
            # checkpointing trades ~30% more compute for a large activation-memory
            # cut -- the difference between fitting an unfrozen encoder on 8GB or not.
            self.backbone.gradient_checkpointing_enable()

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone parameters for initial fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        # x shape: [B, 3, H, W]
        input_size = x.shape[2:]

        if self._is_timm:
            # timm features_only backbones already return a list of NCHW maps.
            hidden_states = self.backbone(x)
        else:
            # Forward through Hugging Face backbone
            outputs = self.backbone(x, output_hidden_states=True)

            # Extract hidden states (multi-scale feature maps)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states
            elif hasattr(outputs, "feature_maps") and outputs.feature_maps is not None:
                hidden_states = outputs.feature_maps
            else:
                hidden_states = outputs[0]

            # Ensure hidden_states is a tuple/list of feature maps
            if not isinstance(hidden_states, (tuple, list)):
                hidden_states = [hidden_states]

        # Segmentation Head
        seg_logits_low = self.seg_head(hidden_states)
        seg_logits = F.interpolate(seg_logits_low, size=input_size, mode='bilinear', align_corners=False)

        # Detection Head
        det_out = self.det_head(hidden_states[-1])

        return {
            "seg_logits": seg_logits,
            "det_out": det_out
        }


if __name__ == "__main__":
    m = HFDualNet(backbone_name="timm:resnet18", freeze_backbone=False)
    out = m(torch.randn(2, 3, 512, 512))
    assert out["seg_logits"].shape == (2, 1, 512, 512), out["seg_logits"].shape
    assert out["det_out"].shape == (2, 6, 7, 7), out["det_out"].shape

    m3 = HFDualNet(backbone_name="timm:resnet18", freeze_backbone=False, num_det_classes=3)
    out3 = m3(torch.randn(2, 3, 512, 512))
    assert out3["det_out"].shape == (2, 4 + 1 + 3, 7, 7), out3["det_out"].shape
    print("ok")
