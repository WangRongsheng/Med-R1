import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

peft_model_id = "/root/autodl-tmp/output/Baichuan-m1/checkpoint-2000"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Baichuan-M1-14B-Instruct", trust_remote_code=True, torch_dtype = torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Baichuan-M1-14B-Instruct', trust_remote_code=True)
model.eval()

def model_resp(prompt):
    input = tokenizer("<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format(prompt, "").strip() + "\nassistant\n ", return_tensors="pt").to(model.device)
    outputs = model.generate(
        **input,
        max_length=8192,
        max_new_tokens=8192,
        temperature=0.1
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("system\n你是一个有用的助手\nuser\n{}\n".format(prompt, "").strip() + "\nassistant\n ", "")
    
    return response
    
with open('eval_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

save_data = []
print(f"总共包含{len(data)}类别数据")
for i in data:
    print(f"类别：{i}包含{len(data[i])}条数据")
    for j in tqdm(range(len(data[i]))):
        bench_name = i
        question = data[str(i)][j]["question"]
        try:
            choice = 'A. {} \nB. {} \nC. {} \nD. {}'.format(
                    data[str(i)][j]["options"]["A"],
                    data[str(i)][j]["options"]["B"],
                    data[str(i)][j]["options"]["C"],
                    data[str(i)][j]["options"]["D"]
                )
        except:
            choice = 'A. {} \nB. {} \nC. {}'.format(
                    data[str(i)][j]["options"]["A"],
                    data[str(i)][j]["options"]["B"],
                    data[str(i)][j]["options"]["C"]
                )

        merge_ques = "Please answer the following multiple-choice questions. Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\nThe question is: "+question+"\nThe options are: \n"+choice
        print(merge_ques)
        
        answer_idx = data[str(i)][j]["answer_idx"]
        answer = data[str(i)][j]["answer"]

        model_response = model_resp(merge_ques)
        print(model_resp)

        item_data = {
            "benchmark": bench_name,
            "question": merge_ques,
            "answer_idx": answer_idx,
            "answer": answer,
            "model_response": model_response
        }
        save_data.append(item_data)

with open("./model_eval_2000.json", 'w', encoding='utf-8') as f2:
    json.dump(save_data, f2, ensure_ascii=False, indent=4)