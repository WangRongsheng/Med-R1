import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch

# 配置参数
MAX_LENGTH = 8192
MODEL_PATH = '/root/autodl-tmp/Baichuan-M1-14B-Instruct'
DATA_PATH = '/root/autodl-tmp/test.json'

def process_func(example):
    instruction_text = "\n".join([
        "你是一个有用的助手",
        example["instruction"] + example["input"],
        "\n\nAssistant: "
    ]).strip()
    
    # Tokenize
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    response = tokenizer(example["output"] + tokenizer.eos_token, add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据集
df = pd.read_json(DATA_PATH)
ds = Dataset.from_pandas(df)

# 初始化tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
model.enable_input_require_grads()

# 训练参数
training_args = TrainingArguments(
    output_dir="./output/baichuan-m1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=1000,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="wandb"
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()