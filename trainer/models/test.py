import torch
import time
from Qformer import BertAttention, BertConfig
from attention_torch import FlashBertAttention

def test_bert_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # 테스트용 입력 생성 (Batch x Time x Channel)
    batch_size, seq_len, hidden_size = 32, 1024, 512
    num_heads = 8
    
    # BERT config 생성
    config = BertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        position_embedding_type="absolute"  # relative position embedding 사용
    )
    
    # 모듈 초기화
    orig_attn = BertAttention(config).to(device).to(dtype)
    flash_attn = FlashBertAttention(config).to(device).to(dtype)
    
    # 가중치 공유
    flash_attn.self.query = orig_attn.self.query
    flash_attn.self.key = orig_attn.self.key
    flash_attn.self.value = orig_attn.self.value
    flash_attn.output.dense = orig_attn.output.dense
    flash_attn.output.LayerNorm = orig_attn.output.LayerNorm
    if hasattr(orig_attn.self, 'distance_embedding'):
        flash_attn.self.distance_embedding = orig_attn.self.distance_embedding
    
    # 입력 생성
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    # BERT attention mask 형식으로 생성 (1 for tokens to attend to, 0 for tokens to ignore)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    # Convert attention mask to attention bias
    attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_len)
    attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min  # Convert to additive mask
    
    # 워밍업
    for _ in range(10):
        with torch.no_grad():
            orig_attn(hidden_states, attention_mask=attention_mask)
            flash_attn(hidden_states, attention_mask=attention_mask)
    
    # 원본 attention 시간 측정
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.no_grad():
        for _ in range(100):
            out1 = orig_attn(hidden_states, attention_mask=attention_mask)[0]
    
    end_event.record()
    torch.cuda.synchronize()
    orig_time = start_event.elapsed_time(end_event) / 100

    # Flash attention 시간 측정
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.no_grad():
        for _ in range(100):
            out2 = flash_attn(hidden_states, attention_mask=attention_mask)[0]
    
    end_event.record()
    torch.cuda.synchronize()
    flash_time = start_event.elapsed_time(end_event) / 100

    # 결과 출력
    print(f"\nPerformance Comparison (batch_size={batch_size}, seq_len={seq_len}):")
    print(f"Original Attention: {orig_time:.2f} ms")
    print(f"Flash Attention: {flash_time:.2f} ms")
    print(f"Speedup: {orig_time/flash_time:.2f}x\n")
    
    # 정확도 비교
    diff = (out1 - out2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    relative_diff = (diff / (out1.abs() + 1e-6)).mean().item()
    
    print("Accuracy Comparison:")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Mean relative difference: {relative_diff}")
    print(f"Output ranges: [{out1.min():.3f}, {out1.max():.3f}], [{out2.min():.3f}, {out2.max():.3f}]")

if __name__ == "__main__":
    test_bert_attention()
