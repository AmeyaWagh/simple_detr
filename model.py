from typing import List

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self, shape, eps: float = 1e-5, elementwise_affine=True, device=None, dtype=None
    ):
        super().__init__()
        self.normalized_shape = (shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(
            torch.empty(self.normalized_shape, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(self.normalized_shape, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        pass


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_hidden_dims: int,
        num_heads: int,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.layers: List[TransformerEncoderLayer] = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    num_hidden_dims, num_heads, dim_feedforward=dim_feedforward
                )
            )

        self.layers = nn.ModuleList(self.layers)
        self.norm = LayerNorm(num_hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_hidden_dims: int,
        num_heads: int,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(
                    num_hidden_dims, num_heads, dim_feedforward=dim_feedforward
                )
            )

        self.layers = nn.ModuleList(self.layers)
        self.norm = LayerNorm(num_hidden_dims)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for mod in self.layers:
            x = mod(x, memory)
        x = self.norm(x)
        return x


class CustomTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            num_encoder_layers, hidden_dim, num_heads, dim_feedforward=dim_feedforward
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, hidden_dim, num_heads, dim_feedforward=dim_feedforward
        )

    def forward(self, embeddings: torch.Tensor, queries: torch.Tensor):
        memory = self.encoder(embeddings)
        print(f"memory shape {memory.shape}")
        output = self.decoder(queries, memory)
        return output


class CustomDETR(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = CustomTransformer(
            hidden_dim, num_heads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.num_queries = 100
        self.query_pos = nn.Parameter(torch.rand(self.num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def image_encoder_forward(self, image: torch.Tensor) -> torch.Tensor:
        # propagate inputs through ResNet-50 up to avg-pool layer
        # inputs shape B,3,H,W
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x  # output shape B,C,fH,fW

    def forward(self, inputs):
        print(f"inputs.shape {inputs.shape}")

        x = self.image_encoder_forward(inputs)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        print(f"feature.shape {h.shape}")  # feature map shape B, 256,fh,fw

        # construct positional encodings
        feat_height, feat_width = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:feat_width].unsqueeze(0).repeat(feat_height, 1, 1),
                    self.row_embed[:feat_height].unsqueeze(1).repeat(1, feat_width, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        print(pos.shape)

        h = h.flatten(2).permute(2, 0, 1)
        embeddings = pos + 0.1 * h

        # propagate through the transformer
        outputs = self.transformer(embeddings, self.query_pos.unsqueeze(1)).transpose(
            0, 1
        )

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(outputs),
            "pred_boxes": self.linear_bbox(outputs).sigmoid(),
        }
