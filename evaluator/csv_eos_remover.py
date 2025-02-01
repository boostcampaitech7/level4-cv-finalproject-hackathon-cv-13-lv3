#CSV 파일에서 '</s>' 토큰을 제거하는 후처리 함수

import pandas as pd
import re

def clean_eos_tokens(text):
    """
    다양한 형태의 '</s>' 변형을 제거하면서, 단어 끝을 올바르게 정리
    """
    if not isinstance(text, str):
        return text
    
    # 연속된 '</s>' 전부 제거
    text = re.sub(r"(</s>)+", "", text)
    
    # '</s>'가 단어 중간에 끼어 있는 경우 정리 (예: 'silly</s>y</s>y' -> 'silly')
    text = re.sub(r"(\w+)(</s>)+[a-zA-Z]+(</s>)+", r"\1", text)
    
    # 일반적인 '</s>' 변형 제거
    text = re.sub(r"\s*</?s/?\s*>", "", text)
    text = re.sub(r"\s*</\s*", "", text)  # 문장 끝의 '</' 제거
    text = re.sub(r"\s*<s/?\s*", "", text)  # '<s/' 또는 '<s' 제거
    
    # 문장 내 존재할 수 있는 비정상적인 제어 문자 제거
    text = re.sub(r"[\x00-\x1F]+", "", text)  
    
    return text.strip()

def remove_eos_from_csv(file_path: str, output_path: str):
    """
    CSV 파일에서 '</s>' 및 그 변형을 제거하는 후처리 함수
    """
    df = pd.read_csv(file_path)
    
    # '</s>' 변형 및 이상한 문자 제거
    df["text"] = df["text"].apply(clean_eos_tokens)
    
    # 수정된 데이터 저장
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")


input_csv = "./submission_results/0201_llama_whsperv3turbo_beats_stage2_submission_asr.csv"
output_csv = "./processed_stage2_submission_asr.csv"
remove_eos_from_csv(input_csv, output_csv)
