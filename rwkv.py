import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from gptq import *
from modelutils import *
from quant import *
import random
from tqdm import tqdm

DEV = torch.device('cuda:0')

def get_wikitext2_rwkv(model, nsamples, seed, seqlen):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

@torch.no_grad()
def rwkv_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.rwkv.blocks

    # model.rwkv.embeddings = model.rwkv.embeddings.to(dev)
    # Set all LayerNorm layer to device
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.to(dev)
    # model.rwkv.ln_out = model.rwkv.ln_out.to(dev)
    layers[0] = layers[0].to(dev)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0}
    
    # Should infer once to turn off self.layers_are_rescaled and not make Catcher crash
    for batch in dataloader:
        model(batch[0].to(dev))
        break

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            #TODO: add back attention_mask and inputs_ids
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for block in model.rwkv.blocks:
        all_layernorm = find_layers(block, layers=[nn.LayerNorm])
        for layernorm in all_layernorm.values():
            layernorm = layernorm.cpu()
    # model.rwkv.ln_out = model.rwkv.ln_out.cpu()
    # layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            raise Exception('true sequential Not implemented')
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
        
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
                scale,zero,g_idx = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['rwkv.blocks.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
                gptq[name].free()
                
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

# TODO: perform packing on GPU
def rwkv_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name],scale,zero,g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1):
    from transformers import RwkvConfig, RwkvForCausalLM

    config = RwkvConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = RwkvForCausalLM(config)
    model.seqlen = 1024
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)

    qlayers, state_dict = torch.load(checkpoint)

    new_layers = {}
    # keep only 1st decoder block
    for key, value in layers.items():
        if key in qlayers.keys():
            new_layers[key] = value

    # Should infer once to turn off self.layers_are_rescaled
    model = model.to(DEV)
    fake_batch = torch.zeros(1, model.seqlen, dtype=torch.long, device=DEV)
    model(fake_batch)
    model = model.cpu()
    
    make_quant(model, new_layers, wbits, groupsize)
    del layers
    del new_layers

    print('Loading model ...')
    model.load_state_dict(state_dict, strict = False)
    
    print('Done.')

    return model

def benchmark(model, testloader, benchmark_samples):
    
    ctx_len = 1024
    stride = ctx_len // 2
    seq_len = testloader.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, stride * benchmark_samples, stride)):
        end_loc = min(begin_loc + ctx_len, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = testloader.input_ids[:, begin_loc:end_loc].to(DEV)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print(f"nlls: {torch.stack(nlls)}")
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity: {ppl}")

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--model', type=str,
        help='HuggingFace model'
    )

    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='evaluate quantized model.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--save_safetensors', type=str, default='',
        help='Save quantized `.safetensors` checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    
    if args.load:
        print("Loading model ...")
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        print("Create model ...")
        if args.model == "sgugger/rwkv-7b-pile":
            raise NotImplementedError("Wait for https://github.com/huggingface/transformers/issues/23368 to be fixed first")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
        print(model.hf_device_map)
        model.seqlen = 1024 # https://huggingface.co/BlinkDL/rwkv-4-pile-430m
        model.eval()
    
    print("Loading dataset ...")
    dataloader, testloader = get_wikitext2_rwkv(model=args.model, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen)

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        print("Begin quantization...")
        quantizers = rwkv_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.benchmark:
        print("Begin benchmarking...")
        model = model.to(DEV)
        benchmark(model, testloader, args.benchmark)
        
    if args.save:
        print("Saving model ...")
        rwkv_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save([{key: None for key in quantizers.keys()}, model.state_dict()], args.save) 
