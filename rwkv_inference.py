import torch
from transformers import AutoTokenizer
import argparse
from rwkv import load_quant

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='HuggingFace model: https://huggingface.co/RWKV')
parser.add_argument('--load', type=str, default='',help='Load quantized model.')
parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
args = parser.parse_args()

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = load_quant(args.model, args.load, args.wbits, args.groupsize)
model = model.to(device)

prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(inputs["input_ids"].to(device), max_new_tokens=20)
print(tokenizer.decode(output[0].tolist()))