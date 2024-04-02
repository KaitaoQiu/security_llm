import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

def load_model():
    base_model = 'NousResearch/Llama-2-7b-chat-hf'
    new_model_path = "./security_miles_model/ppo_model"

    load_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        temperature=1.0,

    )
    
    model = PeftModel.from_pretrained(load_model, new_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    return pipe

os.environ['HF_HOME'] = './cache'
print("huggingface cache is in {}".format(os.getenv('HF_HOME')))


def generate_answer(question, pipeline):
    prompt = f"<s>[INST] {question} [/INST]"

    result = pipeline(prompt)

    return result[0]['generated_text'].replace('<s>', '').replace('</s>', '')

if __name__ == "__main__":
    import sys
    pipeline = load_model()
    question = sys.argv[1] if len(sys.argv) > 1 else "What's cybersecurity?"
    print(generate_answer(question, pipeline))
