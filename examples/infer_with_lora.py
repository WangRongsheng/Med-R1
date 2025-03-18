from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "/root/autodl-tmp/output/Baichuan-m1/checkpoint-2000" # https://modelscope.cn/models/wangrongsheng/Med-R1
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Baichuan-M1-14B-Instruct", trust_remote_code=True, torch_dtype = torch.bfloat16).cuda() # https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Baichuan-M1-14B-Instruct', trust_remote_code=True) # https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct

model.eval()

# prompt = "9.11和9.9哪个大？"
prompt = """
    患者：被沾有病人血液的实心针头扎了，针头是一天前沾的病人血液，还有传染疾病的可能吗（女, 年龄26岁）
    医生：请问您是否知道那位病人的具体健康状况或是否患有任何传染性疾病？例如，乙肝、丙肝或艾滋病等？
    患者：不知道。
    医生：请问您是否接种过乙型肝炎疫苗？
    患者：接种过。
    医生：请问您被扎到的具体部位是哪里？伤口的深度如何？是否有出血？
    患者：大拇指根部，深度大约0.5厘米，出血了。
    
    根据以上信息，若该病人同时患有乙肝、丙肝或艾滋病，患者最可能感染哪种疾病，并说明理由。
"""
input = tokenizer("<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format(prompt, "").strip() + "\nassistant\n ", return_tensors="pt").to(model.device)

# outputs = model.generate(
#     **input,
#     max_length=8192,
#     eos_token_id=2,
#     do_sample=True,
#     repetition_penalty=1.3,
#     no_repeat_ngram_size=5,
#     temperature=0.1,
#     top_k=40,
#     top_p=0.8,
# )

outputs = model.generate(
    **input,
    max_length=8192,
    temperature=0.6
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).replace("system\n你是一个有用的助手\nuser\n{}\n".format(prompt, "").strip() + "\nassistant\n ", ""))
