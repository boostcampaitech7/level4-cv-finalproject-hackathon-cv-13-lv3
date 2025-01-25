# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#explicit-dispatcher-control

import torch
from torch import Tensor, nn
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import warnings

from .modeling_whisper import WhisperAttention
from .beats.backbone import MultiheadAttention, TransformerEncoder
from .modeling_llama import LlamaConfig, LlamaAttention, apply_rotary_pos_emb
from .Qformer import BertSelfAttention, BertSelfOutput, BertAttention

class FlashWhisperAttention(WhisperAttention):
    """Whisper Attention을 Flash Attention으로 변환"""
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

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # 원본과 동일한 방식으로 텐서 변환
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        # attention mask 처리를 원본과 동일하게
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attention_mask = attention_mask.view(bsz, 1, tgt_len, src_len)

        # Flash Attention 계산 전에 원본과 동일한 형태로 변환
        query_4d = query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)
        key_4d = key_states.view(bsz, self.num_heads, src_len, self.head_dim)
        value_4d = value_states.view(bsz, self.num_heads, src_len, self.head_dim)

        # attention mask를 원본과 동일한 방식으로 처리
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=query_4d.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_4d.dtype).min

        # Flash Attention 계산
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(
                query_4d,
                key_4d,
                value_4d,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scaling  # 원본과 동일한 scaling 적용
            )

        # layer head mask 처리를 원본과 동일하게
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_output = attn_output * layer_head_mask.view(1, -1, 1, 1)

        # 출력 형태 변환을 원본과 동일하게
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # attention weights 계산도 원본과 동일하게
        attn_weights = None
        if output_attentions:
            attn_weights = torch.matmul(query_4d, key_4d.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)

        return attn_output, attn_weights, past_key_value
    
class FlashMultiheadAttention(MultiheadAttention):
    """BEATs의 MultiheadAttention을 Flash Attention으로 변환"""
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
        """원본 MultiheadAttention과 동일한 입력/출력 구조 유지"""
        
        # 원본과 동일한 초기 처리
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0) if key is not None else tgt_len
        
        scaling = self.scaling
        alpha = 32  # BEATs의 alpha scaling 값

        # Relative Attention Bias 계산
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1)

        # Self/Cross Attention 처리
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # BEATs의 스케일링 적용
        q *= scaling
        q *= 1 / alpha

        # 텐서 변환
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # attention mask 처리
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            expected_mask_size = [bsz, 1, tgt_len, src_len]
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(*expected_mask_size)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.expand(*expected_mask_size) + key_padding_mask

        # position bias 적용
        if position_bias is not None:
            if attn_mask is None:
                attn_mask = position_bias
            else:
                attn_mask = attn_mask + position_bias

        # Flash Attention 계산
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_module.p if self.training else 0.0,
                is_causal=False,
                scale=alpha  # BEATs의 alpha scaling
            )

        # 출력 형태 변환
        attn_output = attn_output.transpose(1, 2).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        # attention weights 계산 (필요한 경우)
        attn_weights = None
        if need_weights:
            attn_weights = torch.bmm(q.view(-1, tgt_len, self.head_dim), 
                                   k.view(-1, src_len, self.head_dim).transpose(1, 2))
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=1)

        return attn_output, attn_weights, position_bias

class FlashBertSelfAttention(BertSelfAttention):
    """Qformer의 Self Attention을 Flash Attention으로 변환"""
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

        # Key, Value 프로젝션
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

        # Relative Position Embedding 계산
        relative_position_scores = None
        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            
            # 기존 distance embedding 로직 유지
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                relative_position_scores = relative_position_scores_query + relative_position_scores_key

        # attention mask 처리 수정
        if attention_mask is not None:
            # attention_mask는 [batch_size, 1, 1, seq_length] 형태
            attention_mask = attention_mask.to(dtype=query_layer.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_layer.dtype).min

        # Flash Attention 계산
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,  # 이제 올바른 형태의 마스크 전달
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )

        # Relative position scores 적용 (필요한 경우)
        if relative_position_scores is not None:
            context_layer = context_layer + relative_position_scores

        # Cross Attention 특수 처리
        if is_cross_attention and self.save_attention:
            attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_probs = attention_probs / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_probs = attention_probs + attention_mask
            attention_probs = nn.functional.softmax(attention_probs, dim=-1)
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # 출력 형태 변환
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Attention weights 계산 (필요한 경우)
        attention_weights = None
        if output_attentions:
            attention_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_weights = attention_weights / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_weights = attention_weights + attention_mask
            attention_weights = nn.functional.softmax(attention_weights, dim=-1)

        return context_layer, attention_weights, past_key_value

class FlashBertAttention(BertAttention):
    """Qformer의 Attention을 Flash Attention으로 변환"""
    def __init__(self, config, is_cross_attention=False):
        super().__init__(config)
        self.attention = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

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
        mixed_query_layer = self.attention.query(hidden_states)
        mixed_key_layer = self.attention.key(hidden_states)
        mixed_value_layer = self.attention.value(hidden_states)

        query_layer = self.attention.transpose_for_scores(mixed_query_layer)
        key_layer = self.attention.transpose_for_scores(mixed_key_layer)
        value_layer = self.attention.transpose_for_scores(mixed_value_layer)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=query_layer.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.attention.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=1.0/math.sqrt(self.attention.attention_head_size)
            )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.output(context_layer, hidden_states)

        return attention_output, None, past_key_value

class FlashLlamaAttention(LlamaAttention):
    """LLaMA Attention을 Flash Attention으로 변환"""
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

        # 원본과 동일한 방식으로 query, key, value 계산
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE 적용도 원본과 동일하게
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # attention mask 처리도 원본과 동일하게
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=query_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(query_states.dtype).min

        # Flash Attention 계산
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # LLaMA는 dropout 사용 안함
                is_causal=False,
                scale=1.0/math.sqrt(self.head_dim)  # LLaMA의 scaling 방식
            )

        # Attention weights 계산 (필요한 경우)
        attention_weights = None
        if output_attentions:
            attention_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attention_weights = attention_weights + attention_mask
            attention_weights = torch.softmax(attention_weights, dim=-1)

        # 출력 형태 변환
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attention_weights, past_key_value

def replace_attention_with_flash_attention(model):
    """모델의 모든 attention 모듈을 Flash Attention으로 교체하는 함수"""
    attention_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (BertAttention, LlamaAttention, WhisperAttention, MultiheadAttention)):
            attention_modules.append((name, module))
    
    for name, module in attention_modules:
        # 부모 모듈 찾기
        parent_name = '.'.join(name.split('.')[:-1])
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        
        last_name = name.split('.')[-1]
        
        # 모듈 타입에 따라 적절한 Flash Attention으로 교체
        if isinstance(module, BertAttention):
            flash_module = FlashBertAttention(module.config, module.attention.is_cross_attention)
            # state_dict를 사용하여 가중치 복사
            flash_module.load_state_dict(module.state_dict())
            
            # 추가 속성들 복사
            for attr in ['position_embedding_type', 'max_position_embeddings', 
                        'layer_norm_eps', 'hidden_dropout_prob']:
                if hasattr(module.attention, attr):
                    setattr(flash_module.attention, attr, getattr(module.attention, attr))
            
            # 특수한 설정들 복사
            if hasattr(module.attention, 'distance_embedding'):
                flash_module.attention.distance_embedding = module.attention.distance_embedding
            flash_module.output = module.output

        elif isinstance(module, LlamaAttention):
            flash_module = FlashLlamaAttention(module.config)
            # state_dict를 사용하여 가중치 복사
            flash_module.load_state_dict(module.state_dict())
            
            # RoPE 관련 설정 복사
            flash_module.rotary_emb = module.rotary_emb
            for attr in ['max_position_embeddings', 'num_attention_heads', 'head_dim']:
                if hasattr(module, attr):
                    setattr(flash_module, attr, getattr(module, attr))

        elif isinstance(module, WhisperAttention):
            flash_module = FlashWhisperAttention(
                module.embed_dim,
                module.num_heads,
                module.dropout,
                module.is_decoder,
                module.bias
            )
            # state_dict를 사용하여 가중치 복사
            flash_module.load_state_dict(module.state_dict())
            
            # 추가 속성들 복사
            for attr in ['scaling', 'head_dim', 'embed_dim']:
                if hasattr(module, attr):
                    setattr(flash_module, attr, getattr(module, attr))

        elif isinstance(module, MultiheadAttention):
            flash_module = FlashMultiheadAttention(
                module.embed_dim,
                module.num_heads,
                dropout=module.dropout_module.p,
                self_attention=module.self_attention,
                encoder_decoder_attention=module.encoder_decoder_attention
            )
            # state_dict를 사용하여 가중치 복사
            flash_module.load_state_dict(module.state_dict())
            
            # BEATs 특수 설정 복사
            for attr in ['scaling', 'head_dim', 'has_relative_attention_bias',
                        'num_buckets', 'max_distance', 'gru_rel_pos']:
                if hasattr(module, attr):
                    setattr(flash_module, attr, getattr(module, attr))
            
            if hasattr(module, 'relative_attention_bias'):
                flash_module.relative_attention_bias = module.relative_attention_bias

        # 모듈 교체
        setattr(parent, last_name, flash_module)
        
        # 검증
        verify_attention_replacement(module, flash_module)

def verify_attention_replacement(original_module, flash_module):
    """교체된 attention 모듈이 올바르게 동작하는지 검증"""
    with torch.no_grad():
        # 테스트 입력 생성
        batch_size = 2
        seq_length = 32
        hidden_size = original_module.embed_dim if hasattr(original_module, 'embed_dim') else original_module.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(next(original_module.parameters()).device)
        attention_mask = torch.ones(batch_size, seq_length).to(next(original_module.parameters()).device)
        
        # 원본과 Flash 버전의 출력 비교
        original_output = original_module(hidden_states, attention_mask)[0]
        flash_output = flash_module(hidden_states, attention_mask)[0]
        
        # 출력 차이 계산
        max_diff = (original_output - flash_output).abs().max().item()
        avg_diff = (original_output - flash_output).abs().mean().item()
        
        if max_diff > 1e-3:  # 허용 오차
            warnings.warn(f"Large difference detected in attention replacement: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")

