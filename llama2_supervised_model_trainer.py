import os
import json
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

class SecurityLLMTrainer:
    def __init__(self, config_path):
        self.setup_logger()
        self.load_config(config_path)
        self.set_environment_variables()
        self.load_dataset()
        self.load_model_and_tokenizer()
        self.setup_training()

    def setup_logger(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path):
        with open(config_path) as config_file:
            self.config = json.load(config_file)
        self.logger.info("Configuration loaded from {}".format(config_path))

    def set_environment_variables(self):
        for key, value in self.config['cache_dir'].items():
            os.environ[key] = value
            self.logger.info("{} set to {}".format(key, value))
        # os.environ['HF_HOME'] = self.config['cache_dir']
        # os.environ['HF_DATASETS_CACHE'] = self.config['cache_dir']
        # os.environ['HUGGINGFACE_HUB_CACHE'] = self.config['cache_dir']
        # os.environ['TRANSFORMERS_CACHE'] = self.config['cache_dir']
        
    def load_dataset(self):
        self.dataset = load_dataset("parquet", data_files={'train': self.config['train_data_path']})
        self.logger.info("Dataset loaded")

    def load_model_and_tokenizer(self):
        quant_config = self.config['quant_config']
        compute_dtype = getattr(torch, quant_config['bnb_4bit_compute_dtype'])
        quant_config['bnb_4bit_compute_dtype'] = compute_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            quantization_config=quant_config,
            device_map={"": 0},
            cache_dir=self.config['cache_dir']['HF_HOME']
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.logger.info("Model and tokenizer loaded")
        self.logger.info("Cache directory is {}".format(self.config['cache_dir']['HF_HOME']))
        self.logger.info("CUDA available: {}".format(torch.cuda.is_available()))

    def setup_training(self):
        peft_args = LoraConfig(**self.config['peft_config'])
        training_params = TrainingArguments(**self.config['training_params'], output_dir=self.config['training_log_path'])

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset['train'],
            peft_config=peft_args,
            dataset_text_field="train",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=training_params,
            packing=False,
        )
        self.logger.info("Training setup completed")

    def train(self):
        self.logger.info("Starting training")
        self.trainer.train()
        self.logger.info("Training completed")

    def save_model(self):
        self.trainer.model.save_pretrained(self.config['new_model_path'])
        self.model.config.to_json_file(os.path.join(self.config['new_model_path'], 'config.json'))
        self.logger.info("Model saved to {}".format(self.config['new_model_path']))

if __name__ == "__main__":
    config_path = './model_config/filter_security_training.json'
    print("loading config from {}".format(config_path))
    trainer = SecurityLLMTrainer(config_path=config_path)
    trainer.train()
    trainer.save_model()
