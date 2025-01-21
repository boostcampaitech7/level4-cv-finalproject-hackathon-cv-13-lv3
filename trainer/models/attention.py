import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from typing import Optional, Tuple, Dict, Union
import math
import torch.nn.functional as F

class FlashAttentionBase(nn.Module):
    """Flash Attention의 기본 구현체"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        is_causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = is_causal

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _reshape_for_flash_attn(self, x: torch.Tensor, batch_first: bool = False):
        """텐서를 Flash Attention 형식으로 변환"""
        batch_size, seq_len, _ = x.shape if batch_first else (x.shape[1], x.shape[0], x.shape[2])
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Flash Attention forward pass
        Args:
            query: (batch_size, seq_len, embed_dim) or (seq_len, batch_size, embed_dim)
            key, value: 같은 형식의 텐서 또는 None (self-attention의 경우)
            attention_mask: (batch_size, seq_len) 형식의 불리언 마스크
        """
        batch_first = query.shape[0] != min(query.shape)
        if not batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)

        # QKV 프로젝션
        q = self._reshape_for_flash_attn(self.q_proj(query))
        k = self._reshape_for_flash_attn(self.k_proj(key if key is not None else query))
        v = self._reshape_for_flash_attn(self.v_proj(value if value is not None else query))

        # Flash Attention 적용
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.is_causal,
            softmax_scale=self.scaling,
        )

        # 출력 형태 변환
        attn_output = attn_output.reshape(attn_output.shape[0], -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if not batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None  # attention weights는 반환하지 않음

class FlashLlamaAttention(FlashAttentionBase):
    """LLaMA용 Flash Attention"""
    def __init__(self, config):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=False,
            is_causal=True,
        )
        self.rotary_emb = config.rotary_emb if hasattr(config, 'rotary_emb') else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Rotary embeddings 적용
        if self.rotary_emb is not None:
            kv_seq_len = key_states.shape[1]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[1]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Past key values 처리
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        past_key_value = (key_states, value_states) if use_cache else None

        # Flash Attention 적용
        if not output_attentions:
            attn_output = flash_attn_func(
                query_states.view(bsz, q_len, self.num_heads, self.head_dim),
                key_states.view(bsz, -1, self.num_heads, self.head_dim),
                value_states.view(bsz, -1, self.num_heads, self.head_dim),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                softmax_scale=self.scaling,
            )
            attn_weights = None
        else:
            # output_attentions=True인 경우 기존 attention 사용
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value

class FlashWhisperAttention(FlashAttentionBase):
    """Whisper용 Flash Attention"""
    def __init__(self, config, is_decoder=False):
        super().__init__(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads if not is_decoder else config.decoder_attention_heads,
            bias=True,
            is_causal=is_decoder,
            dropout=config.attention_dropout,
        )
        self.is_decoder = is_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, tgt_len, _ = hidden_states.size()

        # Cross attention 처리
        is_cross_attention = key_value_states is not None

        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self.q_proj(hidden_states)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # Flash Attention 적용
        if not output_attentions and layer_head_mask is None:
            attn_output = flash_attn_func(
                query_states.view(batch_size, -1, self.num_heads, self.head_dim),
                key_states.view(batch_size, -1, self.num_heads, self.head_dim),
                value_states.view(batch_size, -1, self.num_heads, self.head_dim),
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.is_decoder,
                softmax_scale=self.scaling,
            )
            attn_weights = None
        else:
            # 기존 attention 사용
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            if layer_head_mask is not None:
                attn_weights = layer_head_mask * attn_weights
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value

class FlashBEATsAttention(FlashAttentionBase):
    """BEATs용 Flash Attention"""
    def __init__(self, config):
        super().__init__(
            embed_dim=config.encoder_embed_dim,
            num_heads=config.encoder_attention_heads,
            bias=True,
            dropout=config.attention_dropout,
        )

class FlashBertAttention(FlashAttentionBase):
    """Q-Former용 Flash Attention"""
    def __init__(self, config, is_cross_attention=False):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=True,
            dropout=config.attention_probs_dropout_prob,
        )
        self.is_cross_attention = is_cross_attention
        if is_cross_attention:
            self.k_proj = nn.Linear(config.encoder_width, config.hidden_size, bias=True)
            self.v_proj = nn.Linear(config.encoder_width, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        # Cross attention 처리
        if encoder_hidden_states is not None:
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self.q_proj(hidden_states)

        # Flash Attention 적용
        if not output_attentions and head_mask is None:
            attn_output = flash_attn_func(
                query_states.view(batch_size, -1, self.num_heads, self.head_dim),
                key_states.view(batch_size, -1, self.num_heads, self.head_dim),
                value_states.view(batch_size, -1, self.num_heads, self.head_dim),
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                softmax_scale=self.scaling,
            )
            attn_weights = None
        else:
            # 기존 attention 사용
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value

def replace_attention_layers(model: nn.Module) -> nn.Module:
    """모델의 attention layers를 Flash Attention으로 교체"""
    for name, module in model.named_children():
        if hasattr(module, '__class__'):
            classname = module.__class__.__name__
            if classname == 'LlamaAttention':
                new_attn = FlashLlamaAttention(module.config)
            elif classname == 'WhisperAttention':
                new_attn = FlashWhisperAttention(module.config, module.is_decoder)
            elif classname == 'MultiheadAttention' and 'beats' in str(type(model)):
                new_attn = FlashBEATsAttention(module.config)
            elif classname == 'BertSelfAttention':
                new_attn = FlashBertAttention(module.config, module.is_cross_attention)
            else:
                replace_attention_layers(module)
                continue

            # 가중치 복사
            new_attn.q_proj = module.q_proj
            new_attn.k_proj = module.k_proj
            new_attn.v_proj = module.v_proj
            new_attn.out_proj = module.out_proj
            
            # 특별한 속성 복사
            if hasattr(module, 'rotary_emb'):
                new_attn.rotary_emb = module.rotary_emb

            # 부모 모듈에서 attention layer 교체
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_attn)
            else:
                setattr(model, child_name, new_attn)
        else:
            replace_attention_layers(module)

    return model

def enable_flash_attention(model: nn.Module) -> nn.Module:
    """모델의 모든 attention layers를 Flash Attention으로 교체"""
    model = replace_attention_layers(model)
    # Flash Attention은 float16/bfloat16만 지원
    model = model.half()
    return model
