"""Opt-in native runtime: preserve the audio encoder's block-diagonal attention.

The official eager kernel allocates an all-recording matrix before masking its
independent windows. Batch those same windows, including their short tail,
without changing audio samples, parameters, positions or observation actions.
"""
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.utils.rnn import pad_sequence
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioAttention
from vllm.worker.worker import Worker


class WindowedAudioAttention(Qwen2_5OmniAudioAttention):
    def forward(self,hidden_states,cu_seqlens=None):
        bounds=cu_seqlens.tolist()
        lengths=[right-left for left,right in zip(bounds,bounds[1:])]
        if bounds[0]!=0 or bounds[-1]!=len(hidden_states) or not lengths or min(lengths)<=0:
            raise ValueError('Invalid original audio window boundaries')
        def pack(projection):
            values=projection(hidden_states).reshape(-1,self.num_heads,self.head_dim)
            return pad_sequence(values.split(lengths),batch_first=True).transpose(1,2)
        query,key,value=pack(self.q_proj),pack(self.k_proj),pack(self.v_proj)
        valid=torch.arange(query.shape[2],device=query.device)[None,:]<torch.tensor(lengths,device=query.device)[:,None]
        attended=scaled_dot_product_attention(query,key,value,attn_mask=valid[:,None,None,:],dropout_p=0.,is_causal=False)
        joined=torch.cat([attended[i,:,:length,:].transpose(0,1).reshape(length,self.embed_dim)
                          for i,length in enumerate(lengths)])
        return self.out_proj(joined)


class WindowedAudioWorker(Worker):
    def load_model(self):
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling
        modeling.QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES['eager']=WindowedAudioAttention
        super().load_model()
        layers=self.model_runner.model.audio_tower.layers
        if not all(isinstance(layer.self_attn,WindowedAudioAttention) for layer in layers):
            raise RuntimeError('Requested windowed audio backend was not selected')
        print(f'Native audio runtime: windowed_sdpa, {len(layers)} verified layers',flush=True)
