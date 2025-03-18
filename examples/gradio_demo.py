from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import gradio as gr

# Load model and tokenizer
peft_model_id = "/root/autodl-tmp/output/Baichuan-m1/checkpoint-2000"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Baichuan-M1-14B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Baichuan-M1-14B-Instruct', trust_remote_code=True)

model.eval()

# Define the inference function
def generate_response(prompt, temperature, max_length):
    input = tokenizer("<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n".format(prompt).strip() + "\nassistant\n ", 
                      return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **input,
        max_length=max_length,
        temperature=temperature
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = response.replace("system\n你是一个有用的助手\nuser\n{}\n".format(prompt).strip() + "\nassistant\n ", "")
    
    return cleaned_response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<center><h1>Med-R1 Chat Interface</h1></center>")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input", placeholder="输入您的问题...", lines=5)
            temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.1, label="Temperature")
            max_length = gr.Slider(minimum=512, maximum=8192, value=8192, step=512, label="Max Lenghth")
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_text = gr.Textbox(label="Output", lines=20, show_copy_button=True)
            # output_md = gr.Markdown(label="Output")
    
    submit_btn.click(
        fn=generate_response,
        inputs=[input_text, temperature, max_length],
        outputs=output_text
    )

demo.launch(server_port=6006, pwa=True)

