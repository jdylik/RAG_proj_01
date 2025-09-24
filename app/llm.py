import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
_device = "cpu"

def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    model_id =  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tok = AutoTokenizer.from_pretrained(model_id,  trust_remote_code=True)
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = dtype, device_map = None, trust_remote_code = True).to(_device)
    model.eval()
    _model, _tokenizer = model, tok
    return _model, _tokenizer

def llm_answer(system: str, user: str) -> str | None:
    model, tok = _load_model()
    messages = [{"role": "system", "content": system}, 
                {"role": "user", "content": user}]
    print(messages)
    prompt = tok.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    print(prompt)

    inputs = tok(prompt, return_tensors = "pt").to(_device)
    print("inputs", inputs)
    max_new = 256
    temperature = 0.1
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new,
            do_sample = (temperature > 0.0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id = tok.eos_token_id,
            eos_token_id = tok.eos_token_id
        )
    print(tok.decode(outputs[0]))
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text.strip()