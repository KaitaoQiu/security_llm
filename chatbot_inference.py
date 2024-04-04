import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os
import json
def load_model():
    base_model = 'NousResearch/Llama-2-7b-chat-hf'
    new_model_path = "./security_miles_model/ppo_model"
    # new_model_path = "./security_miles_model/ppo_model"

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

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
    return pipe

os.environ['HF_HOME'] = './cache'
print("huggingface cache is in {}".format(os.getenv('HF_HOME')))


def generate_answer(question, pipeline):

    import http.client
    import json

    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": question
    })
    config_path = "./model_config/google_search.json"
    with open(config_path) as f:
        api_config = json.load(f)
    headers = {
    'X-API-KEY': api_config["SERPER_API_KEY"],
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))
    meta_data = json.loads(data.decode("utf-8"))
    background_string  = ""
    if 'knowledgeGraph' in meta_data:
        background_string += str(meta_data['knowledgeGraph'])
    elif "answerBox" in meta_data:
        background_string += str(meta_data["answerBox"])
    else:
        background_string += str(meta_data["organic"][0])
    prompt = f"<s>[INST] {question}, also I will give you some knowledgegraph:{background_string} [/INST]"
    result = pipeline(prompt)
    return result[0]['generated_text'].replace('<s>', '').replace('</s>', '')


if __name__ == "__main__":
    import sys
    pipeline = load_model()
    question = sys.argv[1] if len(sys.argv) > 1 else "What's result of 1+1?"
    print(generate_answer(question, pipeline))
