# GPTQ-for-RWKV

# Setup

```
pip install -r requirements.txt
python setup_cuda.py install
```


# Results

- Here is a summary of RWKV:
> Tested on a V100 16GB using the commands below

| Wiki2 PPL | FP16 | 4bit-GPTQ | 4g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|
| RWKV-430M | 17.59375  | 19.28125  | 18.328125 |

- All models can be found in the [HF hub](https://huggingface.co/RWKV)

```
# Fp16
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 16 --benchmark 32
# Quantize to 4bit
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --save rwkv430M_4bit.pt
# Bench 4bit
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --load rwkv430M_4bit.pt --benchmark 32

# Quantize to 4bit groupsize 128
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --groupsize 128 --save rwkv430M_4g128.pt
# Bench 4bit groupsize 128
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --groupsize 128 --load rwkv430M_4g128.pt --benchmark 32
```

- For text generation:
```
python rwkv_inference.py  --model RWKV/rwkv-4-430m-pile --load rwkv430M_4bit.pt --wbits 4
```

# Todo

- [ ] Fix [HuggingFace bug](https://github.com/huggingface/transformers/issues/23368) when dispatching model to CPU & GPU

# Acknowledgements

- [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- [qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)