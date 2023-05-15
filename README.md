# GPTQ-for-RWKV

# Setup

```
pip install -r requirements.txt
python setup_cuda.py install
```


# Results

- Here is a summary of RWKV:

| Wiki2 PPL | FP16 | 4bit-GPTQ | 4g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|
| RWKV-430M | 17.59375  | ???  | ???  |

- All models can be found in the [HF hub](https://huggingface.co/RWKV)

```
# Fp16
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 16 --benchmark 32
# Quantize to 4bit
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --save rwkv430M_4bit.pt
# Bench 4bit
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --load rwkv430M_4bit.pt --benchmark 32

# TO CHECK
# Quantize to 4bit groupsize 128
python rwkv.py --model RWKV/rwkv-4-430m-pile --dataset wikitext2 --wbits 4 --groupsize 128 --save rwkv430M_4g128.pt
# Bench 4bit groupsize 128
python rwkv.py --dataset wikitext2 --wbits 4 --groupsize 128 --load rwkv430M_4g128.pt --benchmark 32
```

- For text generation:
```
python rwkv_inference.py  --model RWKV/rwkv-4-430m-pile --load rwkv430M_4bit.pt --wbits 4
```

# Todo

- [ ] Fix [HuggingFace bug](https://github.com/huggingface/transformers/issues/23368) when dispatching model to CPU & GPU
- Packing ...
rwkv.blocks.0.attention.key
rwkv.blocks.0.attention.output
rwkv.blocks.0.attention.receptance
rwkv.blocks.0.attention.value
rwkv.blocks.0.feed_forward.key
rwkv.blocks.0.feed_forward.receptance
rwkv.blocks.0.feed_forward.value
rwkv.blocks.1.attention.key
rwkv.blocks.1.attention.output
rwkv.blocks.1.attention.receptance
rwkv.blocks.1.attention.value
rwkv.blocks.1.feed_forward.key
rwkv.blocks.1.feed_forward.receptance
rwkv.blocks.1.feed_forward.value
rwkv.blocks.2.attention.key
rwkv.blocks.2.attention.output
rwkv.blocks.2.attention.receptance
rwkv.blocks.2.attention.value
rwkv.blocks.2.feed_forward.key
rwkv.blocks.2.feed_forward.receptance
rwkv.blocks.2.feed_forward.value
rwkv.blocks.3.attention.key
rwkv.blocks.3.attention.output
rwkv.blocks.3.attention.receptance
rwkv.blocks.3.attention.value
rwkv.blocks.3.feed_forward.key
rwkv.blocks.3.feed_forward.receptance
rwkv.blocks.3.feed_forward.value
rwkv.blocks.4.attention.key
rwkv.blocks.4.attention.output
rwkv.blocks.4.attention.receptance
rwkv.blocks.4.attention.value
rwkv.blocks.4.feed_forward.key
rwkv.blocks.4.feed_forward.receptance
rwkv.blocks.4.feed_forward.value
rwkv.blocks.5.attention.key
rwkv.blocks.5.attention.output
rwkv.blocks.5.attention.receptance
rwkv.blocks.5.attention.value
rwkv.blocks.5.feed_forward.key
rwkv.blocks.5.feed_forward.receptance
rwkv.blocks.5.feed_forward.value
rwkv.blocks.6.attention.key
rwkv.blocks.6.attention.output
rwkv.blocks.6.attention.receptance
rwkv.blocks.6.attention.value
rwkv.blocks.6.feed_forward.key
rwkv.blocks.6.feed_forward.receptance
rwkv.blocks.6.feed_forward.value
rwkv.blocks.7.attention.key
rwkv.blocks.7.attention.output
rwkv.blocks.7.attention.receptance
rwkv.blocks.7.attention.value
rwkv.blocks.7.feed_forward.key
rwkv.blocks.7.feed_forward.receptance
rwkv.blocks.7.feed_forward.value
rwkv.blocks.8.attention.key
rwkv.blocks.8.attention.output
rwkv.blocks.8.attention.receptance
rwkv.blocks.8.attention.value
rwkv.blocks.8.feed_forward.key
rwkv.blocks.8.feed_forward.receptance
rwkv.blocks.8.feed_forward.value
rwkv.blocks.9.attention.key
rwkv.blocks.9.attention.output
rwkv.blocks.9.attention.receptance
rwkv.blocks.9.attention.value
rwkv.blocks.9.feed_forward.key
rwkv.blocks.9.feed_forward.receptance
rwkv.blocks.9.feed_forward.value
rwkv.blocks.10.attention.key
rwkv.blocks.10.attention.output
rwkv.blocks.10.attention.receptance
rwkv.blocks.10.attention.value
rwkv.blocks.10.feed_forward.key
rwkv.blocks.10.feed_forward.receptance
rwkv.blocks.10.feed_forward.value
rwkv.blocks.11.attention.key
rwkv.blocks.11.attention.output
rwkv.blocks.11.attention.receptance
rwkv.blocks.11.attention.value
rwkv.blocks.11.feed_forward.key
rwkv.blocks.11.feed_forward.receptance
rwkv.blocks.11.feed_forward.value
rwkv.blocks.12.attention.key
rwkv.blocks.12.attention.output
rwkv.blocks.12.attention.receptance
rwkv.blocks.12.attention.value
rwkv.blocks.12.feed_forward.key
rwkv.blocks.12.feed_forward.receptance
rwkv.blocks.12.feed_forward.value
rwkv.blocks.13.attention.key
rwkv.blocks.13.attention.output
rwkv.blocks.13.attention.receptance
rwkv.blocks.13.attention.value
rwkv.blocks.13.feed_forward.key
rwkv.blocks.13.feed_forward.receptance
rwkv.blocks.13.feed_forward.value
rwkv.blocks.14.attention.key
rwkv.blocks.14.attention.output
rwkv.blocks.14.attention.receptance
rwkv.blocks.14.attention.value
rwkv.blocks.14.feed_forward.key
rwkv.blocks.14.feed_forward.receptance
rwkv.blocks.14.feed_forward.value
rwkv.blocks.15.attention.key
rwkv.blocks.15.attention.output
rwkv.blocks.15.attention.receptance
rwkv.blocks.15.attention.value
rwkv.blocks.15.feed_forward.key
rwkv.blocks.15.feed_forward.receptance
rwkv.blocks.15.feed_forward.value
rwkv.blocks.16.attention.key
rwkv.blocks.16.attention.output
rwkv.blocks.16.attention.receptance
rwkv.blocks.16.attention.value
rwkv.blocks.16.feed_forward.key
rwkv.blocks.16.feed_forward.receptance
rwkv.blocks.16.feed_forward.value
rwkv.blocks.17.attention.key
rwkv.blocks.17.attention.output
rwkv.blocks.17.attention.receptance
rwkv.blocks.17.attention.value
rwkv.blocks.17.feed_forward.key
rwkv.blocks.17.feed_forward.receptance
rwkv.blocks.17.feed_forward.value
rwkv.blocks.18.attention.key
rwkv.blocks.18.attention.output
rwkv.blocks.18.attention.receptance
rwkv.blocks.18.attention.value
rwkv.blocks.18.feed_forward.key
rwkv.blocks.18.feed_forward.receptance
rwkv.blocks.18.feed_forward.value
rwkv.blocks.19.attention.key
rwkv.blocks.19.attention.output
rwkv.blocks.19.attention.receptance
rwkv.blocks.19.attention.value
rwkv.blocks.19.feed_forward.key
rwkv.blocks.19.feed_forward.receptance
rwkv.blocks.19.feed_forward.value
rwkv.blocks.20.attention.key
rwkv.blocks.20.attention.output
rwkv.blocks.20.attention.receptance
rwkv.blocks.20.attention.value
rwkv.blocks.20.feed_forward.key
rwkv.blocks.20.feed_forward.receptance
rwkv.blocks.20.feed_forward.value
rwkv.blocks.21.attention.key
rwkv.blocks.21.attention.output
rwkv.blocks.21.attention.receptance
rwkv.blocks.21.attention.value
rwkv.blocks.21.feed_forward.key
rwkv.blocks.21.feed_forward.receptance
rwkv.blocks.21.feed_forward.value
rwkv.blocks.22.attention.key
rwkv.blocks.22.attention.output
rwkv.blocks.22.attention.receptance
rwkv.blocks.22.attention.value
rwkv.blocks.22.feed_forward.key
rwkv.blocks.22.feed_forward.receptance
rwkv.blocks.22.feed_forward.value
rwkv.blocks.23.attention.key
rwkv.blocks.23.attention.output
rwkv.blocks.23.attention.receptance
rwkv.blocks.23.attention.value
rwkv.blocks.23.feed_forward.key
rwkv.blocks.23.feed_forward.receptance
rwkv.blocks.23.feed_forward.value

# Acknowledgements

- [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- [qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)


