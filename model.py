from typing import List, Optional, Tuple
import math
from dataclasses import dataclass
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


logger = logging.getLogger(__name__)


@dataclass
class Predictions:
    """Dataclass for managing predictions."""

    logits: torch.Tensor
    boxes: torch.Tensor


class SimpleDETR(nn.Module):
    """Detection Transformer."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        image_feature_dim: int = 2048,
        max_image_dimensions: Tuple[int, int] = (50, 50),
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(image_feature_dim, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = SimpleTransformer(
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
        self.row_embed = nn.Parameter(
            torch.rand(max_image_dimensions[0], hidden_dim // 2)
        )
        self.col_embed = nn.Parameter(
            torch.rand(max_image_dimensions[1], hidden_dim // 2)
        )

    def _image_encoder_forward(self, image: torch.Tensor) -> torch.Tensor:
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

        x = self.conv(x)
        return x  # output shape B,C,fH,fW

    def _positional_encoding(self, feat_height: int, feat_width: int) -> torch.Tensor:
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
        return pos

    def forward(self, image: torch.Tensor) -> Predictions:
        """Detect objects in the image.

        Args:
            image (torch.Tensor): image tensor of shape (B,C,H,W)

        Returns:
            Predictions: prediction with boxes and logits
        """
        logger.info("image shape:%s", str(image.shape))

        image_features = self._image_encoder_forward(image)
        # convert from 2048 to 256 feature planes for the transformer
        logger.info("Image feature shape %s", str(image_features.shape))

        # construct positional encodings
        feat_height, feat_width = image_features.shape[-2:]
        pos = self._positional_encoding(feat_height=feat_height, feat_width=feat_width)
        logger.info("positional encoding shape: %s", str(pos.shape))

        image_features = image_features.flatten(2).permute(2, 0, 1)
        embeddings = pos + 0.1 * image_features

        logger.info("embedding shape %s", str(embeddings.shape))

        # propagate through the transformer
        outputs = self.transformer(embeddings, self.query_pos.unsqueeze(1))
        outputs = outputs.transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        logits = self.linear_class(outputs)
        boxes = self.linear_bbox(outputs).sigmoid()
        return Predictions(logits=logits, boxes=boxes)

    def infer(self, image: torch.Tensor, threshold: float = 0.7):
        outputs: Predictions = self.forward(image)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold


class SimpleTransformer(nn.Module):
    """Implementation of Simplified Pytorch Transformer."""

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

    def forward(self, embeddings: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Predict output for queries given embeddings.

        Args:
            embeddings (torch.Tensor): embedding representation
            queries (torch.Tensor): queries of objects.

        Returns:
            torch.Tensor: values associated with the queries.
        """

        # Encode the embedding in latent space using Transformer encoder.
        memory = self.encoder(embeddings)
        logger.info("memory shape %s", str(memory.shape))

        # Decode the queries in the latent space
        output = self.decoder(queries, memory)
        return output


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
                TransformerDecoderLayer(
                    num_hidden_dims, num_heads, dim_feedforward=dim_feedforward
                )
            )

        self.layers = nn.ModuleList(self.layers)
        self.norm = LayerNorm(num_hidden_dims)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory)
        x = self.norm(x)
        return x


class LayerNorm(nn.Module):
    """Implementation of Layer Norm operator in pytorch."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform layer norm on input tensor."""
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class MultiheadAttention(nn.Module):
    """Simplified implementation of Pytorch Multiheaded attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.use_torch_impl = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights=True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        attn_output, attn_output_weights = self._multi_head_attention_forward(
            query, key, value, need_weights=need_weights
        )

        return attn_output, attn_output_weights

    def _multi_head_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights=True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logger.debug("----")
        logger.debug("Query shape %s", str(query.shape))
        logger.debug("Key shape %s", str(key.shape))
        logger.debug("Value shape %s", str(value.shape))
        if self.use_torch_impl:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=None,
                need_weights=need_weights,
                attn_mask=None,
                use_separate_proj_weight=False,
                q_proj_weight=None,
                k_proj_weight=None,
                v_proj_weight=None,
                average_attn_weights=True,
            )
            return attn_output, attn_output_weights
        tgt_len, bsz, embed_dim = query.shape

        # reshape q, k, v for multihead attention and make em batch first
        query = (
            query.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        key = (
            key.contiguous()
            .view(key.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        value = (
            value.contiguous()
            .view(value.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_output, attn_output_weights = self._scaled_dot_product_attention(
            query, key, value
        )
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if need_weights:
            src_len = key.size(1)
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        return attn_output, None

    def _scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn


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
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

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
        x = self.norm1(x + self._self_attention_block(x))
        x = self.norm2(x + self._feedforward_block(x))
        return x

    def _self_attention_block(
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
    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm3 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self._self_attention_block(x))
        x = self.norm2(x + self._multi_head_attention_block(x, memory))
        x = self.norm3(x + self._feedforward_block(x))

        return x

    # self-attention block
    def _self_attention_block(
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

    # multihead attention block
    def _multi_head_attention_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x,
            memory,
            memory,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
