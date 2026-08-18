---
license: mit
datasets:
- open-r1/OpenR1-Math-220k
language:
- en
- es
- fr
- de
base_model:
- microsoft/Phi-3.5-mini-instruct
tags:
- CoT
- thinking
- reasoning
- rapple-ml
- rappleml
- math
---



![image.png_2K_202608181207](https://cdn-uploads.huggingface.co/production/uploads/6a5f8c87a38cac087c2c6c05/rdliof5clgu3KeFZsqUX8.jpeg)

**A Long-Context Reasoning & Structural Planning Engine by Rapple ML**

---

## 📌 Overview

**Raisin-4B** is a lightweight (~3.8B parameter), long-context language model fine-tuned and merged by **Rapple ML**. Built specifically for **multi-step reasoning, logical planning, and long-doc processing**, Raisin-4B generates explicit `<think> ... </think>` Chain-of-Thought (CoT) traces before delivering final outputs.

By leveraging **YaRN (Yet another RoPE eXtension)** scaling alongside DARE-TIES parameter merging, Raisin-4B provides a massive **131,072-token context window** while retaining tight instruction adherence and a low memory footprint.

---

## 📊 Model Architecture & Metadata

| Attribute | Specification |
| --- | --- |
| **Organization** | **Rapple ML** |
| **Architecture** | Dense Decoder-Only Transformer (`Phi3ForCausalLM`) |
| **Base Model** | `microsoft/Phi-3.5-mini-instruct` |
| **Active Parameters** | ~3.8 Billion |
| **Context Length** | **131,072 Tokens** (128k) |
| **Position Embedding** | Su-RoPE / YaRN (Factor 4.0) |
| **Precision** | `bfloat16` / `float16` |
| **Fine-Tuning Stack** | Unsloth + TRL (`SFTTrainer`) |
| **Merge Method** | DARE-TIES (`mergekit`) |

---

## 💻 Quickstart Guide

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "RappleML/Raisin-4B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

prompt = "<|user|>\nAnalyze this problem and outline a plan before answering: How many r's are in strawberry?<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

---

## ⚙️ Prompt Template

Raisin-4B uses the standard **Phi-3 Chat Template**. To activate reasoning mode, prompt the assistant to output a `<think>` block:

```text
<|user|>
[YOUR PROMPT / INSTRUCTION HERE]<|end|>
<|assistant|>
<think>
[RAISIN WILL GENERATE STEP-BY-STEP REASONING HERE]
</think>
[FINAL ANSWER]

```

---

## 🛠️ Training & Merge Details

1. **Supervised Fine-Tuning (SFT):** Trained with `unsloth` and `unsloth_zoo` on a curated reasoning slice from `open-r1/OpenR1-Math-220k`.
2. **Merging Strategy:** Merged via `mergekit` using DARE-TIES ($0.6$ density, $0.5$ weight) with `microsoft/Phi-3.5-mini-instruct` to eliminate redundant parameter shifts and preserve base capabilities.
3. **Context Extension:** Configured with a 4.0x **YaRN RoPE** scaling setup to extend positional context to **131,072 tokens**.

---

*Developed with ❤️ by **Rapple ML***