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
    """Dataclass for managing predictions.

    Args:
        logits (torch.Tensor): class logits of the predicted boxes
        boxes (torch.Tensor): vector of box dimensions <x,y,width,height> along column.
    """

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
        num_object_queries: int = 100,
        backbone=resnet50(),
    ):
        """Create a simple detection transformer.

        Args:
            num_classes (int): number of classes predicted
            hidden_dim (int, optional): dimensions of the transformer layers. Defaults to 256.
            num_heads (int, optional): number of transformer heads. Defaults to 8.
            num_encoder_layers (int, optional): number of encoder layers. Defaults to 6.
            num_decoder_layers (int, optional): number of decoder layers. Defaults to 6.
            image_feature_dim (int, optional): size of the image feature channel. Defaults to 2048.
            max_image_dimensions (Tuple[int, int], optional): maximum size of the image feature. Defaults to (50, 50).
            num_object_queries (int, optional): number of object queries. Defaults to 100.
        """

        super().__init__()

        # Resnet backbone
        self.backbone = ImageEncoderBackbone(backbone=backbone)

        # 1x1 convolution to reduce the high-level activation map C to a smaller dimension d
        self.conv = nn.Conv2d(image_feature_dim, hidden_dim, 1)

        # create a Simple transformer
        self.transformer = SimpleEncoderDecoderTransformer(
            hidden_dim, num_heads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.num_object_queries = num_object_queries
        # queries of the share (Q,D)
        self.query_pos = nn.Parameter(torch.rand(self.num_object_queries, hidden_dim))

        # Spatial positional encodings
        # NOTE In baseline DETR sine positional encodings is used.
        self.row_embed = nn.Parameter(
            torch.rand(max_image_dimensions[0], hidden_dim // 2)
        )
        self.col_embed = nn.Parameter(
            torch.rand(max_image_dimensions[1], hidden_dim // 2)
        )

    def _positional_encoding(self, feature_shape: Tuple[int, int]) -> torch.Tensor:
        """Creates positional encoding tensor for the given feature map shape.

        Args:
            feature_shape (Tuple[int, int]): shape of the image feature map

        Returns:
            torch.Tensor: positional encoding for each pixel in the image feature map.
        """
        feat_height, feat_width = feature_shape
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
            image (torch.Tensor): image tensor of shape (B,3,H,W)

        Returns:
            Predictions: prediction with boxes and logits
        """
        logger.info("image shape:%s", str(image.shape))

        feat = self.backbone(image)
        # convert from 2048 to 256 feature planes for the transformer
        image_features = self.conv(feat)

        logger.info("Image feature shape %s", str(image_features.shape))

        # construct positional encodings for each pixel in the image feature map.
        positional_encoding = self._positional_encoding(image_features.shape[-2:])
        logger.info("positional encoding shape: %s", str(positional_encoding.shape))

        # flatten image feature map from B,C,fH,fW -> fH*fW,B,C
        image_features = image_features.flatten(2).permute(2, 0, 1)
        embeddings = positional_encoding + 0.1 * image_features

        logger.info("embedding shape %s", str(embeddings.shape))

        # propagate through the transformer
        outputs = self.transformer(embeddings, self.query_pos.unsqueeze(1))
        outputs = outputs.transpose(0, 1)  # (B,Q,D)

        # finally project transformer outputs to class labels and bounding boxes
        logits = self.linear_class(outputs)
        boxes = self.linear_bbox(outputs).sigmoid()
        return Predictions(logits=logits, boxes=boxes)


class ImageEncoderBackbone(nn.Module):
    """Resnet Image Encoder Backbone."""

    def __init__(self, backbone=resnet50()) -> None:
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Resnet forward pass.

        Args:
            image (torch.Tensor): B,3,H,W

        Returns:
            torch.Tensor: image features output shape B,C,fH,fW where fH = H/32, fW = W/32
        """
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class SimpleEncoderDecoderTransformer(nn.Module):
    """Implementation of Simplified Pytorch Encoder-Decoder Transformer."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
    ):
        """Construct a Simple Encoder-Decoder Transformer.

        Args:
            hidden_dim (int, optional): dimension of the transformers. Defaults to 512.
            num_heads (int, optional): number of heads in the transformers. Defaults to 8.
            num_encoder_layers (int, optional): number of encoder layers in the encoder transformer. Defaults to 6.
            num_decoder_layers (int, optional): number of encoder layers in the decoder transformer. Defaults to 6.
            dim_feedforward (int, optional): feed forward network dimensions. Defaults to 2048.
        """

        super().__init__()

        self.encoder = TransformerEncoder(
            num_encoder_layers, hidden_dim, num_heads, dim_feedforward
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, hidden_dim, num_heads, dim_feedforward
        )

    def forward(self, embeddings: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Predict output for queries given embeddings.

        Args:
            embeddings (torch.Tensor): embedding representation, (fH*fW,B,C)
            queries (torch.Tensor): queries of objects.(Q,1,D)

        Returns:
            torch.Tensor: values associated with the object queries. (Q,B,D)
        """
        # Encode the embedding in latent space using Transformer encoder.
        encoded_embeddings = self.encoder(embeddings)
        logger.info("encoded_embeddings shape %s", str(encoded_embeddings.shape))

        # Decode the queries in the latent space
        output = self.decoder(queries, encoded_embeddings)
        return output


class TransformerEncoder(nn.Module):
    """Input Encoder transformer.

    Encodes the image features to embedding vectors
    """

    def __init__(
        self,
        num_layers: int,
        num_hidden_dims: int,
        num_heads: int,
        dim_feedforward: int = 2048,
    ):
        """Create a transformer encoder.

        Args:
            num_layers (int): number of transformer layers
            num_hidden_dims (int): dimensions of transformer
            num_heads (int): number of heads in each transformer
            dim_feedforward (int, optional): dimensions of the feedforward network. Defaults to 2048.
        """
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
        """encodes the given input using a transformer.

        Args:
            x (torch.Tensor): input features. (fH*fW,B,C)

        Returns:
            torch.Tensor: encoded embeddings. (fH*fW,B,C)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """A Queryable decoder Transformer."""

    def __init__(
        self,
        num_layers: int,
        num_hidden_dims: int,
        num_heads: int,
        dim_feedforward: int = 2048,
    ):
        """Create a transformer decoder.

        Args:
            num_layers (int): number of transformer layers
            num_hidden_dims (int): dimensions of transformer
            num_heads (int): number of heads in each transformer
            dim_feedforward (int, optional): dimensions of the feedforward network. Defaults to 2048.
        """

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

    def forward(
        self, q: torch.Tensor, encoded_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Decode the query values from the encoded embeddings.

        Args:
            q (torch.Tensor): query vectors, (Q,1,D)
            encoded_embeddings (torch.Tensor): encoded embeddings. (fH*fW,B,C)

        Returns:
            torch.Tensor: decoded values for the queries. (Q,B,D)
        """
        x: torch.Tensor = q
        for layer in self.layers:
            x = layer(x, encoded_embeddings)
        return self.norm(x)


class TransformerEncoderLayer(nn.Module):
    """Implementation of a Multiheaded attention encoder transformer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        """Create an instance of Multiheaded attention transformer encoder.

        Args:
            hidden_dim (int): dimensions of the transformer.
            num_heads (int): number of heads
            dim_feedforward (int, optional): dimensions of the feedforward network. Defaults to 2048.
            layer_norm_eps (float, optional): epsilon value for layer norm. Defaults to 1e-5.
            dropout (float, optional): dropout percentage. Defaults to 0.1.
        """
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        self.norm1 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = F.relu
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the features using self attention and FFN.

        Args:
            x (torch.Tensor): features to encode. (fH*fW,B,C)

        Returns:
            torch.Tensor: encoded features. (fH*fW,B,C)
        """
        x = self.norm1(x + self._self_attention_block(x))
        x = self.norm2(x + self.ffn(x))
        return x

    def _self_attention_block(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn(
            query=q,
            key=q,
            value=q,
            need_weights=False,
        )[0]
        return self.dropout1(x)


class TransformerDecoderLayer(nn.Module):
    """Implementation of a Multiheaded attention decoder transformer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        """Create an instance of Multiheaded attention transformer decoder.

        Args:
            hidden_dim (int): dimensions of the transformer.
            num_heads (int): number of heads
            dim_feedforward (int, optional): dimensions of the feedforward network. Defaults to 2048.
            layer_norm_eps (float, optional): epsilon value for layer norm. Defaults to 1e-5.
            dropout (float, optional): dropout percentage. Defaults to 0.1.
        """
        super().__init__()
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.multihead_attn = MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

        self.norm1 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm3 = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Performs foward pass of transformer layer.

        Args:
            x (torch.Tensor): input tensor. (Q,1,D)
            embeddings (torch.Tensor): encoded embeddings. (fH*fW,B,C)

        Returns:
            torch.Tensor: output of the transformer decoder. (Q,B,D)
        """
        x = self.norm1(x + self._self_attention_block(x))
        x = self.norm2(x + self._multi_head_attention_block(x, embeddings))
        x = self.norm3(x + self.ffn(x))

        return x

    def _self_attention_block(
        self,
        q: torch.Tensor,
    ) -> torch.Tensor:
        x = self.self_attn(
            query=q,
            key=q,
            value=q,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _multi_head_attention_block(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            q,
            kv,
            kv,
            need_weights=False,
        )[0]
        return self.dropout2(x)


class MultiheadAttention(nn.Module):
    """Simplified implementation of Pytorch Multiheaded attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Create a Multi Head Attention layer

        Args:
            embed_dim (int): embedding dimension
            num_heads (int): number of transformer heads
            dropout (float, optional): dropout percentage. Defaults to 0.0.
        """
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights=True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a transformer forward pass.

        Args:
            query (torch.Tensor): query tensor_
            key (torch.Tensor): key tensor
            value (torch.Tensor): values associated to keys.
            need_weights (bool, optional): if true returns attention weights. Defaults to True.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: tuple of attention output and attention weights
        """

        attn_output, attn_output_weights = self._multi_head_attention_forward(
            query, key, value, need_weights=need_weights
        )
        logger.debug("Attention output shape %s", str(attn_output.shape))
        if attn_output_weights is not None:
            logger.debug("Attention weights shape %s", str(attn_output_weights.shape))
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

        # Query, Key and Value are batched
        assert query.dim() == 3 and key.dim() == 3 and value.dim() == 3

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        self.head_dim = embed_dim // self.num_heads

        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

        # compute in-projection
        q, k, v = self._in_projection_packed(
            query, key, value, self.in_proj_weight, self.in_proj_bias
        )

        # reshape q, k, v for multihead attention and make embedding dim batch first
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(key.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(value.shape[0], bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v)
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
        embedding_dim = q.size(-1)
        q = q / math.sqrt(embedding_dim)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def _in_projection_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        embedding_dim = q.size(-1)  # E is embedding dim
        if k is v:
            if q is k:
                # self-attention
                q_, k_, v_ = F.linear(q, w, b).chunk(3, dim=-1)
                return [q_, k_, v_]
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([embedding_dim, embedding_dim * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([embedding_dim, embedding_dim * 2])
                q_ = F.linear(q, w_q, b_q)
                k_, v_ = F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
                return [q_, k_, v_]
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            q_ = F.linear(q, w_q, b_q)
            k_ = F.linear(k, w_k, b_k)
            v_ = F.linear(v, w_v, b_v)
            return [q_, k_, v_]


class LayerNorm(nn.Module):
    """Implementation of Layer Norm operator in pytorch."""

    def __init__(
        self,
        shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        """Create a layer norm layer.

        Args:
            shape (int): shape of the layer_
            eps (float, optional): epsilon value for norm computation. Defaults to 1e-5.
            elementwise_affine (bool, optional): if true, initializes elementwise weights. Defaults to True.
            device (_type_, optional): pytorch device type. Defaults to None.
            dtype (_type_, optional): pytorch datatype. Defaults to None.
        """
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
        """Reset internal parameters."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class FeedForwardNetwork(nn.Module):
    """Implementation of Feed Forward Network"""

    def __init__(
        self, hidden_dim: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.activation = F.relu
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform feed forward network forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output of the FFN.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        return x
