import torch
from torch import Tensor, nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from modeling_whisper import WhisperAttention
from beats.backbone import MultiheadAttention
from modeling_llama import LlamaConfig, LlamaAttention, apply_rotary_pos_emb
from Qformer import BertSelfAttention, BertSelfOutput, BertAttention

def flash_attention_forward(
    query_layer,
    key_layer,
    value_layer,
    attention_mask=None,
    head_mask=None,
    output_attentions=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """공통 Flash Attention forward 함수"""
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        # 기본 attention 계산
        context_layer = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )

        # attention weights가 필요한 경우에만 계산
        if output_attentions or head_mask is not None:
            scale_factor = 1 / math.sqrt(query_layer.size(-1)) if scale is None else scale
            attn_weights = torch.matmul(query_layer, key_layer.transpose(-2, -1)) * scale_factor
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            if head_mask is not None:
                attn_weights = attn_weights * head_mask
        else:
            attn_weights = None

        return context_layer, attn_weights

class FlashWhisperAttention(WhisperAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, is_decoder: bool = False, bias: bool = True):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # [B*H, T, D] -> [B, H, T, D] 변환
        query_states = query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
        key_states = key_states.view(bsz, self.num_heads, -1, self.head_dim)
        value_states = value_states.view(bsz, self.num_heads, -1, self.head_dim)

        context_layer, attention_weights = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_decoder
        )

        # [B, H, T, D] -> [B, T, H*D]
        attn_output = context_layer.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attention_weights, past_key_value

class FlashBEATsAttention(MultiheadAttention):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            has_relative_attention_bias=False,
            num_buckets=32,
            max_distance=128,
            gru_rel_pos=False,
            rescale_init=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
        )
        self.k_proj.bias = None
        self.q_head_dim = embed_dim // num_heads
        self.scaling = self.q_head_dim ** -0.5
        
        # 필요한 속성들 추가
        self.onnx_trace = False
        self.tpu = False
        self.encoder_decoder_attention = encoder_decoder_attention
        self.self_attention = self_attention
        self.head_dim = embed_dim // num_heads

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0) if key is not None else tgt_len

        scaling = self.scaling

        if not self.training:
            dropout = 0.0
        else:
            dropout = self.dropout

        # q, k, v 프로젝션
        q = self.q_proj(query)
        k = self.k_proj(query if key is None else key)
        v = self.v_proj(query if value is None else value)

        # BEATs 방식대로 차원 변경
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q * scaling

        # [B*H, T, D] -> [B, H, T, D] 변환
        q_4d = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k_4d = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v_4d = v.view(bsz, self.num_heads, src_len, self.head_dim)

        context_layer, attention_weights = flash_attention_forward(
            q_4d,
            k_4d,
            v_4d,
            attention_mask=attn_mask,
            head_mask=None,
            output_attentions=need_weights,
            dropout_p=dropout
        )

        # [B, H, T, D] -> [T, B, E]
        attn_output = context_layer.transpose(1, 2).contiguous()
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attention_weights, None
    
    
class FlashLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
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

        # 프로젝션 및 차원 변경
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer, attention_weights = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            dropout_p=0.0,  # LLaMA는 dropout 사용하지 않음
            scale=None,  # scaling은 내부적으로 처리됨
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attention_weights, past_key_value
    

class FlashBertSelfAttention(BertSelfAttention):
    def __init__(self, config, is_cross_attention):
        super().__init__(config, is_cross_attention)

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
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        context_layer, attention_weights = flash_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.attention_head_size)
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, None) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)

        return outputs

class FlashBertAttention(BertAttention):
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config, is_cross_attention)
        self.self = FlashBertSelfAttention(config, is_cross_attention)

def replace_attention_with_flash_attention(model):
    """모든 attention 모듈을 Flash Attention으로 교체"""
    for name, module in model.named_modules():
        # BERT Attention 교체
        if isinstance(module, BertAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
                    
            flash_attention = FlashBertAttention(module.self.config, is_cross_attention=module.self.is_cross_attention)
            flash_attention.self.query = module.self.query
            flash_attention.self.key = module.self.key
            flash_attention.self.value = module.self.value
            flash_attention.output = module.output
            
            last_name = name.split('.')[-1]
            setattr(parent, last_name, flash_attention)
            
        # LLaMA Attention 교체
        elif isinstance(module, LlamaAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
                    
            flash_attention = FlashLlamaAttention(module.config)
            flash_attention.q_proj = module.q_proj
            flash_attention.k_proj = module.k_proj
            flash_attention.v_proj = module.v_proj
            flash_attention.o_proj = module.o_proj
            flash_attention.rotary_emb = module.rotary_emb
            
            last_name = name.split('.')[-1]
            setattr(parent, last_name, flash_attention)
            
        # Whisper Attention 교체
        elif isinstance(module, WhisperAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
                    
            flash_attention = FlashWhisperAttention(
                module.embed_dim,
                module.num_heads,
                module.dropout,
                module.is_decoder,
                module.bias
            )
            flash_attention.q_proj = module.q_proj
            flash_attention.k_proj = module.k_proj
            flash_attention.v_proj = module.v_proj
            flash_attention.out_proj = module.out_proj
            
            last_name = name.split('.')[-1]
            setattr(parent, last_name, flash_attention)
            
        # BEATs Attention 교체
        elif isinstance(module, MultiheadAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
                    
            flash_attention = FlashBEATsAttention(
                module.embed_dim,
                module.num_heads,
                module.kdim,
                module.vdim,
                module.dropout,
                module.bias,
                module.add_bias_kv,
                module.add_zero_attn,
                module.self_attention,
                module.encoder_decoder_attention
            )
            flash_attention.q_proj = module.q_proj
            flash_attention.k_proj = module.k_proj
            flash_attention.v_proj = module.v_proj
            flash_attention.out_proj = module.out_proj
            
            last_name = name.split('.')[-1]
            setattr(parent, last_name, flash_attention)
    
    return model

