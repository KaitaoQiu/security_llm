import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

from peft import LoraConfig

class PPOModelTrainer:
    def __init__(self, config_path):
        self.setup_logger()
        self.logger.info("torch.cuda.is_available():{}".format(torch.cuda.is_available()))
        self.set_device()
        self.load_config(config_path)
        self.load_model_and_tokenizer()
        self.setup_sentiment_pipeline()
        self.prepare_dataset()
        self.setup_ppo_trainer()

    def setup_logger(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path):
        with open(config_path) as config_file:
            self.config = json.load(config_file)

    def set_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device: {}".format(self.device))
        self.logger.info("Configuration loaded from {}".format(config_path))
    
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        # Create a PeftConfig instance
        lora_config_dict = self.config.get('lora_config', {})
        peft_config = LoraConfig(**lora_config_dict) if lora_config_dict else None
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config['model_name'],
            # peft_config=self.config.get('lora_config', None),
            quantization_config=self.config.get('nf4_config', {}),
            use_safetensors=self.config.get('use_safetensors', False),
            peft_config=peft_config,
        )
        self.model.to("cuda")

        self.logger.info("Model and tokenizer loaded on GPU")

    def setup_sentiment_pipeline(self):
        self.reward_model_name = self.config['reward_model']
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=self.reward_model_name,
            device_map={"": self.device},
            model_kwargs={"load_in_8bit": self.config['rm_load_in_8bit']},
            tokenizer=self.tokenizer,
            return_token_type_ids=False,
        )

        if self.sentiment_pipe.model.config.pad_token_id is None:
            self.sentiment_pipe.model.config.pad_token_id = self.sentiment_pipe.model.config.eos_token_id


    def prepare_dataset(self):
        # dataset = load_dataset(self.config['dataset_name'], split="train[:1%]")

        # input_size = LengthSampler(self.config['input_min_text_length'], self.config['input_max_text_length'])

        # def tokenize(example):
        #     text_size = input_size()
        #     example["input_ids"] = self.tokenizer.encode(example["chosen"])[:text_size]
        #     example["query"] = self.tokenizer.decode(example["input_ids"])
        #     return example

        # self.dataset = dataset.map(tokenize, batched=False)
        # self.dataset.set_format("torch")
        # self.logger.info("Dataset prepared")
        # import pdb; pdb.set_trace()

        # train_dataset = load_dataset(self.config["dataset_name"], data_dir="data/rl", split="train")
        # train_dataset = train_dataset.select(range(10000))
        
        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples["question"]:
                query = "Question: " + question + "\n\nAnswer: "
                tokenized_question = self.tokenizer(query, truncation=True)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

            return new_examples

        ds = load_dataset(self.config["dataset_name"], data_files="*parquet", split='train')
        # ds = ds.select(range(10000))
        original_columns = ds.column_names
        ds = ds.map(
            preprocess_function,
            batched=True,
            remove_columns=original_columns,
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
        ds.set_format(type="torch")
        # return ds
        self.dataset = ds

    def setup_ppo_trainer(self):
        self.ppo_trainer = PPOTrainer(
            PPOConfig(
                **self.config['ppo_config'],
                project_kwargs={"logging_dir": self.config['tsboard_logging_dir']}
            ),
            self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            data_collator=self.collator,
        )
        self.logger.info("PPO Trainer setup completed")

    @staticmethod
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # def train(self):
    #     self.logger.info("Starting PPO training")
    #     generation_kwargs = self.config.get('generation_kwargs', {})

    #     for _epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
    #         question_tensors = batch["input_ids"]
    #         response_tensors = self.ppo_trainer.generate(question_tensors, return_prompt=False, **generation_kwargs)
    #         batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    #         # Compute reward score
    #         texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    #         inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
    #         raw_rewards = self.ppo_trainer.accelerator.unwrap_model(self.ppo_trainer.model).compute_reward_score(**inputs)
    #         rewards = [raw_rewards[i, -1, 1] for i in range(len(raw_rewards))]  # take last token

    #         # Run PPO step
    #         stats = self.ppo_trainer.step(question_tensors, response_tensors, rewards)
    #         self.ppo_trainer.log_stats(stats, batch, rewards)

    #     self.logger.info("PPO Training completed")

    def train(self):
        self.logger.info("Starting PPO training")
        generation_kwargs = self.config.get('generation_kwargs', {})
        output_length_sampler = LengthSampler(self.config['output_min_length'], self.config['output_max_length'])

        for epoch, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
            if epoch >= self.config['total_ppo_epochs']:
                break

            question_tensors = batch["input_ids"]
            response_tensors = self.ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.sentiment_pipe(texts)
            # import pdb; pdb.set_trace()
            rewards = [torch.tensor(output["score"] - self.config['reward_baseline']) if output["label"] == "LABEL_1" else torch.tensor(0.0) for output in pipe_outputs]


            stats = self.ppo_trainer.step(question_tensors, response_tensors, rewards)
            self.ppo_trainer.log_stats(stats, batch, rewards)

            if self.config['save_freq'] and epoch and epoch % self.config['save_freq'] == 0:
                self.ppo_trainer.save_pretrained(self.config['output_dir'] + f"step_{epoch}")

        self.logger.info("PPO Training completed")



    def save_model(self):
        self.model.save_pretrained(self.config['output_dir'])
        self.model.config.to_json_file(os.path.join(self.config['output_dir'], 'config.json'))
        self.logger.info("PPO Model saved to {}".format(self.config['output_dir']))

if __name__ == "__main__":
    config_path = './model_config/ppo_model_config.json'
    print("Loading config from {}".format(config_path))
    ppo_trainer = PPOModelTrainer(config_path=config_path)
    ppo_trainer.train()
    ppo_trainer.save_model()
