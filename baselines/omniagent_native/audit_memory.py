"""Read-only weight/runtime audit; short resource probes, no personality grading."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from amp.common import write_json,read_json


def weight_inventory(model):
    groups=Counter();dtypes=Counter()
    for path in Path(model).glob('*.safetensors'):
        with path.open('rb') as source:
            n=struct.unpack('<Q',source.read(8))[0];header=json.loads(source.read(n))
        for name,entry in header.items():
            if name=='__metadata__':continue
            size=entry['data_offsets'][1]-entry['data_offsets'][0]
            groups[name.split('.')[0]]+=size;dtypes[entry['dtype']]+=size
    return dict(total_tensor_bytes=sum(groups.values()),module_bytes=dict(groups),dtype_bytes=dict(dtypes))


def kernel_probe(output):
    import torch
    from torch.profiler import profile,ProfilerActivity
    results=[]
    for length in (512,1024):
        query=torch.randn(20,length,64,device='cuda',dtype=torch.bfloat16)
        key=torch.randn_like(query);value=torch.randn_like(query)
        mask=torch.zeros(1,length,length,device='cuda',dtype=torch.bool)
        for start in range(0,length,128):mask[:,start:start+128,start:start+128]=True
        for variant in ('3d_bool','4d_bool','4d_float'):
            q,k,v=(query,key,value) if variant=='3d_bool' else tuple(x.unsqueeze(0) for x in (query,key,value))
            m=mask if variant=='3d_bool' else mask.unsqueeze(0)
            if variant=='4d_float':m=torch.zeros_like(m,dtype=query.dtype).masked_fill(~m,torch.finfo(query.dtype).min)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
            with torch.inference_mode(),profile(activities=[ProfilerActivity.CPU]) as prof:
                result=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=m,dropout_p=0.)
                torch.cuda.synchronize()
            results.append(dict(length=length,variant=variant,extra_peak_bytes=torch.cuda.max_memory_allocated()-baseline,
                operators=[e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key],finite=bool(torch.isfinite(result).all())))
            del result
    write_json(output,dict(torch=torch.__version__,device=torch.cuda.get_device_name(),results=results,scope='Small safe kernels; not a full native model replay'))
    print(json.dumps(results,indent=2),flush=True)


def model_probe(model,output):
    import torch
    import soundfile as sf
    from amp.backend import QwenBackend,message,SYSTEM
    backend=QwenBackend(model,max_tokens=8,temperature=0)
    parameters=list(backend.model.parameters())
    devices=Counter();dtypes=Counter()
    for p in parameters:devices[str(p.device)]+=p.numel()*p.element_size();dtypes[str(p.dtype)]+=p.numel()*p.element_size()
    audio=backend.policy.audio_tower;visual=backend.policy.visual
    report=dict(model=model,checkpoint=weight_inventory(model),hf_device_map=backend.model.hf_device_map,
        parameter_bytes_by_device=dict(devices),parameter_bytes_by_dtype=dict(dtypes),
        has_talker=backend.model.has_talker,talker_loaded=hasattr(backend.model,'talker'),token2wav_loaded=hasattr(backend.model,'token2wav'),
        quantized=bool(getattr(backend.model,'is_quantized',False)),
        audio_parameters=sum(p.numel() for p in audio.parameters()),visual_parameters=sum(p.numel() for p in visual.parameters()),
        audio_attention=str(audio.config._attn_implementation),audio_attention_class=type(audio.layers[0].self_attn).__name__,
        cuda_after_load=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()),
        probes=[],training=False,external_api_calls=0)
    if any(not str(p.device).startswith('cuda') for p in parameters):
        report['all_parameters_on_gpu']=False;write_json(output/'report.json',report)
        raise RuntimeError('CPU or meta model parameters detected; this is not a pure GPU load')
    report['all_parameters_on_gpu']=True
    original=read_json(ROOT/'runs/omniagent_expanded_20261117/omniagent_native/cases/IELTS:185.json')
    full=Path(original['trace']['steps'][1]['audio_path'])
    with sf.SoundFile(full) as stream:
        rate=stream.samplerate;wave=stream.read(20*rate,dtype='float32')
    short=output/'audio_20s.wav';short.parent.mkdir(parents=True,exist_ok=True);sf.write(short,wave,rate)
    write_json(output/'report.json',report)
    for label,path in (('short_20s',short),('same_failed_native_audio',full)):
        duration=sf.info(path).duration
        messages=[dict(role='system',content=SYSTEM),message(dict(task='Describe the audible content briefly.'),[dict(type='audio',audio=str(path))])]
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            inputs=backend.prepare(messages)
            mask=inputs['feature_attention_mask']
            observed=[float(n)*backend.processor.feature_extractor.hop_length/backend.processor.feature_extractor.sampling_rate for n in mask.sum(-1).tolist()]
            with torch.inference_mode():
                tokens=backend.policy.generate(**inputs,max_new_tokens=8,do_sample=False,use_audio_in_video=False)
                torch.cuda.synchronize()
            report['probes'].append(dict(label=label,duration=duration,processor_audio_seconds=observed,
                input_tokens=int(inputs['input_ids'].shape[1]),generated_tokens=int(tokens.shape[1]-inputs['input_ids'].shape[1]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),passed=True))
            del inputs,tokens
        except (RuntimeError,ValueError) as error:
            report['probes'].append(dict(label=label,duration=duration,passed=False,error=str(error)))
        write_json(output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=('kernel','model'))
    parser.add_argument('--model');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.mode=='kernel':kernel_probe(args.output)
    else:model_probe(args.model,args.output)


if __name__=='__main__':main()
