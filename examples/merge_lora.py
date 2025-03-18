import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig

def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype = torch.bfloat16).cuda()

    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    lora_path = "/root/autodl-tmp/output/Baichuan-m1/checkpoint-2000" # https://modelscope.cn/models/wangrongsheng/Med-R1
    model_path = "/root/autodl-tmp/Baichuan-M1-14B-Instruct" # https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct
    output = "/root/autodl-tmp/Med-R1"

    apply_lora(model_path,output,lora_path)
