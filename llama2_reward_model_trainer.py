import os
import gc
import json
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, HfArgumentParser
from trl import RewardTrainer, RewardConfig, get_peft_config
from peft import LoraConfig

import warnings

# warnings.filterwarnings("ignore", message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.")
# warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
# warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")


class RewardModelTrainer:
    def __init__(self, config_path):
        self.setup_logger()
        self.load_config(config_path)
        self.set_environment_variables()
        self.load_model_and_tokenizer()
        self.load_reward_dataset()
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

    def load_reward_dataset(self):
        raw_datasets = load_dataset(self.config['reward_dataset'])
        raw_datasets = raw_datasets.filter(
            lambda x: len(x["chosen"]) <= self.config['reward_config']['max_length']
            and len(x["rejected"]) <= self.config['reward_config']['max_length']
        )
        self.dataset = raw_datasets.map(self.preprocess_function, batched=True)
        self.logger.info("Dataset loaded and preprocessed")
        del raw_datasets
        gc.collect()

    def preprocess_function(self, examples):
        # Tokenize chosen/rejected pairs of inputs
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # tokenized_chosen = self.tokenizer(chosen)
            # tokenized_rejected = self.tokenizer(rejected)
            # tokenized_chosen = self.tokenizer(chosen, padding='max_length', truncation=True, max_length=self.config['reward_config']['max_length'])
            tokenized_chosen = self.tokenizer(chosen, truncation=True, max_length=self.config['reward_config']['max_length'])
            # tokenized_rejected = self.tokenizer(rejected, padding='max_length', truncation=True, max_length=self.config['reward_config']['max_length'])
            tokenized_rejected = self.tokenizer(rejected, truncation=True, max_length=self.config['reward_config']['max_length'])

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['base_model'],
            num_labels=1,
            cache_dir=self.config['cache_dir']['HF_HOME']
        )
        self.logger.info("Model and tokenizer loaded")
        # import pdb; pdb.set_trace()
        # gc.collect()

    def setup_training(self):
        peft_args = LoraConfig(**self.config['peft_config'])
        reward_config = RewardConfig(**self.config['reward_config'])
        # training_args = TrainingArguments(
        #     output_dir=self.config['reward_config']['output_dir'],
        #     per_device_train_batch_size=self.config['reward_config']['per_device_train_batch_size'],
        #     num_train_epochs=self.config['reward_config']['num_train_epochs'],
        #     gradient_accumulation_steps=self.config['reward_config']['gradient_accumulation_steps'],
        #     learning_rate=self.config['reward_config']['learning_rate'],
        #     logging_dir=self.config['reward_config']['logging_dir'],
        #     logging_steps=self.config['reward_config']['logging_steps'],
        #     evaluation_strategy=self.config['reward_config']['evaluation_strategy'],
        #     max_length=self.config['reward_config']['max_length'],
        #     remove_unused_columns=False
        # )

        self.trainer = RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            # args=training_args,
            args=reward_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            peft_config=peft_args,
        )
        self.logger.info("Training setup completed")

    def train(self):
        self.logger.info("Starting training")
        self.trainer.train()
        self.logger.info("Training completed")

    def save_model(self):
        self.trainer.model.save_pretrained(self.config['reward_config']['output_dir'])
        self.model.config.to_json_file(os.path.join(self.config['reward_config']['output_dir'], 'config.json'))
        self.logger.info("Model saved to {}".format(self.config['reward_config']['output_dir']))

if __name__ == "__main__":
    config_path = './model_config/reward_model_config.json'
    print("Loading config from {}".format(config_path))
    trainer = RewardModelTrainer(config_path=config_path)
    trainer.train()
    trainer.save_model()
