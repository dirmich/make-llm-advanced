# Make LLM-advanced

**ChatGPT 수준의 LLM을 운영하는 고급 기법**

저자: dirmich | 출판사: Highmaru Press

## 소개

이 패키지는 **Make LLM-advanced** 책의 전체 소스 코드와 LaTeX 원고를 포함합니다. 분산 학습, 최신 아키텍처, 양자화, 정렬, 추론 최적화 등 대규모 LLM을 실제로 운영하기 위한 고급 기법을 다룹니다.

**선수 조건**: 1권 Make LLM-basic을 먼저 읽었거나 동등한 지식이 필요합니다.

## 디렉토리 구조

```
make-llm-advanced/
├── src/makellm/
│   ├── distributed/          # DDP, FSDP, ZeRO, Pipeline, Tensor Parallel
│   ├── architectures/        # RoPE, GQA, SwiGLU, RMSNorm, MoE
│   ├── quantization/         # INT8, INT4, AWQ
│   ├── alignment/            # SFT, RLHF, DPO, LoRA, RewardModel
│   ├── inference/            # KV Cache, PagedAttention
│   ├── data/                 # 필터, MinHash, 합성 데이터
│   └── utils/
├── tests/                    # pytest 단위 테스트 (71개)
├── book/                     # LaTeX 원고
│   ├── main.tex
│   ├── shared/preamble.tex
│   ├── chapters/             # 10개 장 + 2개 부록
│   └── make_llm_advanced.pdf
├── pyproject.toml
├── requirements.txt
└── README.md (이 파일)
```

## 설치

```bash
cd make-llm-advanced
python -m venv venv
source venv/bin/activate
pip install -e .
```

## 테스트 실행

```bash
pytest tests/ -v
```

모든 71개 테스트가 통과해야 합니다.

## 주요 모듈 사용 예제

### LoRA 파인튜닝

```python
import torch.nn as nn
from makellm.alignment import apply_lora, LoRAConfig
from makellm.alignment.lora import count_lora_parameters

model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
apply_lora(model, rank=8, alpha=16)
stats = count_lora_parameters(model)
print(f"Trainable: {stats['trainable_params']:,} ({stats['trainable_pct']:.2f}%)")
```

### INT8 양자화

```python
from makellm.quantization import INT8Quantizer

quantizer = INT8Quantizer(group_size=128)
quantizer.quantize_model(model)
savings = quantizer.memory_savings(model)
print(f"FP32: {savings['fp32_mb']:.1f} MB → INT8: {savings['int8_mb']:.1f} MB")
```

### KV Cache 추론

```python
from makellm.inference.kv_cache import CachedAttention

attn = CachedAttention(d_model=512, n_heads=8)
attn.init_cache(max_seq_len=2048)
# 첫 번째 스텝
out = attn(x, use_cache=True)
# 이후 스텝: 이전 K, V는 캐시에서 재사용
```

## 책 PDF 빌드

```bash
cd book
tectonic -o . --outname make_llm_advanced main.tex
```

## 라이선스

- 코드: MIT
- 책 본문: CC BY-NC-SA 4.0

## 관련 자료

- 1권: Make LLM-basic (토크나이저, 트랜스포머 기초)
- PRD/Plan/Task 문서: 별도 PDF
