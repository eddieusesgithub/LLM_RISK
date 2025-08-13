import csv
import time
import os
import warnings
import re
from typing import Optional, List
from collections import Counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

INPUT_CSV_FILENAME = "prompts.csv"
OUTPUT_CSV_FILENAME = "prompts_generated_sc.csv"  # change per run if you want separate files

TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REQUEST_DELAY_SECONDS = 0.05
SEED = 42

# mode options: "baseline", "pe", "sc"  (sc = self-consistency + PE)
MODE = "sc"

# prompt Engineering instruction (strict, single-sentence, refusal-aware)
PE_INSTRUCTION = (
    "Answer the question concisely and accurately in ONE sentence. "
    "Do NOT include follow-up questions, extra information, or new topics. "
    "If you are not certain, reply exactly: \"insufficient evidence\".\n"
)

# self-Consistency params
SC_NUM_SAMPLES = 5
SC_TEMPERATURE = 0.35
SC_TOP_P = 0.9

# single-shot generation params
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.9
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 96

tokenizer = None
model = None
device = None

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer():
    global tokenizer, model, device

    print(f"Loading model '{TINYLLAMA_MODEL_NAME}' and tokenizer...")
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    print("Model and tokenizer loaded.")


def build_prompt(question_text: str, mode: str = "baseline", context: Optional[str] = None) -> str:
    parts = []
    if mode == "pe":
        parts.append(PE_INSTRUCTION.strip())
    if context:
        parts.append(f"Context:\n{context.strip()}")
    parts.append(f"Question: {question_text.strip()}")
    user_block = "\n\n".join(parts)
    return f"<|user|>\n{user_block}\n<|assistant|>\n"


class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer(s, add_special_tokens=False).input_ids for s in stop_strings]

    def _endswith(self, sequence_ids, suffix_ids):
        L = len(suffix_ids)
        if L == 0 or len(sequence_ids) < L:
            return False
        return sequence_ids[-L:] == suffix_ids

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        for suffix in self.stop_ids:
            if self._endswith(seq, suffix):
                return True
        return False

STOP_STRINGS = [
    "\nQuestion:", "\nQ:", "\nUser:", "\n<|user|>", "\n\nQuestion:", "Question:"
]\

def decode_generated(output_ids, input_len: int) -> str:
    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
        text = text.replace(token, "")
    return text.strip() if text else "[No response generated]"


def truncate_at_stops(text: str) -> str:
    cut_at = None
    for s in ["\nQuestion:", "\nQ:", "\nUser:", "Question:"]:
        idx = text.find(s)
        if idx != -1:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    if cut_at is not None:
        text = text[:cut_at].strip()
    return text


def one_sentence(text: str) -> str:
    if "." in text:
        first = text.split(".")[0].strip()
        return (first + ".") if first else text
    return text


def get_llm_response(
    formatted_prompt: str,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    stops = StoppingCriteriaList([StopOnTokens(tokenizer, STOP_STRINGS)])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stops,
            use_cache=True,
            return_dict_in_generate=False,
        )

    text = decode_generated(output_ids, input_len=input_ids.shape[1])
    text = truncate_at_stops(text)
    text = one_sentence(text)
    return text if text else "[No response generated]"

_ws_punct = re.compile(r"[\W_]+", flags=re.UNICODE)

def normalise_for_vote(text: str) -> str:
    t = text.strip().lower()
    t = _ws_punct.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def generate_self_consistent(
    formatted_prompt: str,
    n_samples: int = SC_NUM_SAMPLES,
    temperature: float = SC_TEMPERATURE,
    top_p: float = SC_TOP_P,
) -> str:
    variants = []
    for _ in range(n_samples):
        ans = get_llm_response(
            formatted_prompt,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        if ans:
            variants.append(ans)

    if not variants:
        return "[No response generated]"

    counts = Counter(normalise_for_vote(v) for v in variants if v.strip())
    if counts:
        winner_norm, _ = counts.most_common(1)[0]
        candidates = [v for v in variants if normalise_for_vote(v) == winner_norm]
        if candidates:
            return sorted(candidates, key=len)[0]

    # Fallback: shortest non-empty response
    non_empty = [v for v in variants if v and v.strip()]
    return sorted(non_empty, key=len)[0] if non_empty else "[No response generated]"


def main():
    set_seed(SEED)
    load_model_and_tokenizer()

    all_data = []
    prompts_to_process = []
    input_headers = []

    print(f"Reading prompts from '{INPUT_CSV_FILENAME}'...")
    with open(INPUT_CSV_FILENAME, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        input_headers = next(reader)
        prompt_col_index = input_headers.index("Prompt")

        ideal_response_col_index = -1
        if "Ideal Response" in input_headers:
            ideal_response_col_index = input_headers.index("Ideal Response")

        for i, row in enumerate(reader):
            if not row:
                continue
            prompt_text = row[prompt_col_index]
            ideal_response_text = row[ideal_response_col_index] if ideal_response_col_index != -1 else ""
            prompts_to_process.append(
                {
                    "prompt": prompt_text,
                    "ideal_response": ideal_response_text,
                    "original_row": row,
                }
            )
            if (i + 1) % 25 == 0:
                print(f"Loaded {i+1} prompts so far...", end="\r")
    print(f"\nSuccessfully loaded {len(prompts_to_process)} prompts.")

    output_headers = input_headers + ["LLM Response"]
    all_data.append(output_headers)

    print(
        f"Generating responses using {TINYLLAMA_MODEL_NAME} "
        f"and saving to '{OUTPUT_CSV_FILENAME}'"
    )

    for i, item in enumerate(prompts_to_process):
        question = item["prompt"]
        original_row = item["original_row"]

        if MODE == "baseline":
            formatted_prompt = build_prompt(question, mode="baseline", context=None)
            generated_response = get_llm_response(formatted_prompt)
        elif MODE == "pe":
            formatted_prompt = build_prompt(question, mode="pe", context=None)
            generated_response = get_llm_response(formatted_prompt)
        elif MODE == "sc":
            # SC uses the PE prompt to bias towards safe behaviour
            formatted_prompt = build_prompt(question, mode="pe", context=None)
            generated_response = generate_self_consistent(formatted_prompt)
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        new_row = original_row + [generated_response]
        all_data.append(new_row)

        # save progress after each response
        with open(OUTPUT_CSV_FILENAME, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(all_data)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(prompts_to_process)} prompts...", end="\r")

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nAll responses generated and saved to '{OUTPUT_CSV_FILENAME}'.")

if __name__ == "__main__":
    main()
