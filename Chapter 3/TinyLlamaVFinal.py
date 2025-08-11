import csv
import time
import os
import warnings

# import from hugging face and pytorch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# suppress warnings and set environment variables before importing transformers
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings("ignore", category=UserWarning)

# loading prompt files and model
INPUT_CSV_FILENAME = 'prompts.csv'
OUTPUT_CSV_FILENAME = 'prompts_generated.csv' #save to this file
TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REQUEST_DELAY_SECONDS = 0.1

# global model and tokeniser
tokenizer = None
model = None
device = None

def load_model_and_tokenizer():
    """Loads the TinyLlama 1.1B model and tokenizer, with fallback to CPU for MPS issues."""
    global tokenizer, model, device

    print(f"Loading model '{TINYLLAMA_MODEL_NAME}' and tokenizer")
    

    if torch.backends.mps.is_available():
        device = torch.device("cpu") #originally tried using mps (gpu) but didn't work so just used cpu although slower
    else:                            #can try use cuda if on windows to use gpu
        device = torch.device("cpu") 

    
    tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL_NAME) #load tokeniser
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
  
    model = AutoModelForCausalLM.from_pretrained(   #load tinyllama model
            TINYLLAMA_MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=True
        )
    model = model.to(device)
    model.eval()
    print("Model and tokenizer loaded.")
        
def get_llm_response(prompt_text: str) -> str:
    """
    Generates a response from the loaded TinyLlama 1.1B model for a given prompt.
    """
    
        # simple prompt formatting - avoid chat template if it causes issues
    formatted_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
        
        # tokenize the prompt
    inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

        # generate response
    with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # disable cache to avoid issues i encountered
                return_dict_in_generate=False
            )
            
        # decode only the generated part
    generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # clean up the response
    generated_text = generated_text.strip()
        
        # remove any remaining special tokens or formatting
    for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
            generated_text = generated_text.replace(token, "")
        
    return generated_text.strip() if generated_text.strip() else "[No response generated]"
        
def main():
    load_model_and_tokenizer()

    all_data = []
    prompts_to_process = []
    input_headers = []

    print(f"Reading prompts from '{INPUT_CSV_FILENAME}'...")
    
    with open(INPUT_CSV_FILENAME, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            input_headers = next(reader)
            prompt_col_index = input_headers.index('Prompt')
            

            ideal_response_col_index = -1
            if 'Ideal Response' in input_headers:
                ideal_response_col_index = input_headers.index('Ideal Response')
            
            for i, row in enumerate(reader):
                if row:
                    prompt_text = row[prompt_col_index]
                    ideal_response_text = row[ideal_response_col_index] if ideal_response_col_index != -1 else ""
                    prompts_to_process.append({
                        'prompt': prompt_text, 
                        'ideal_response': ideal_response_text, 
                        'original_row': row
                    })
                print(f"Loaded {i+1} prompts so far...", end='\r')
    print(f"\nSuccessfully loaded {len(prompts_to_process)} prompts.")

    output_headers = input_headers + ['Generated Response']
    all_data.append(output_headers)

    print(f"Generating responses using {TINYLLAMA_MODEL_NAME} on {device} and saving to '{OUTPUT_CSV_FILENAME}'...")
    
    for i, item in enumerate(prompts_to_process):
        prompt = item['prompt']
        ideal_response = item['ideal_response']
        original_row = item['original_row']

        print(f"Processing prompt {i+1}/{len(prompts_to_process)}: '{prompt[:70]}...'")
        generated_response = get_llm_response(prompt)
        
        new_row = original_row + [generated_response]
        all_data.append(new_row)

        # save progress after each response
        
        with open(OUTPUT_CSV_FILENAME, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(all_data)
        
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nAll responses generated and saved to '{OUTPUT_CSV_FILENAME}'.")
if __name__ == "__main__":
    main()
