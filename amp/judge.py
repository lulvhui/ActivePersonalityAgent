"""Independently check recorded findings against their actual Observe sources."""
import math
from .backend import SYSTEM,message
from .common import digest
from .state import memory_records,record_source
from .perception import decode_observation,observation_view,memory_view,READ_SUPPORT_RULE


class MultimodalJudge:
    def __init__(self,backend):
        self.backend=backend

    def assess(self,trajectory):
        checks={}
        for key,record in memory_records(trajectory['memory']).items():
            source=record_source(record,trajectory['observations'])
            decoded=decode_observation(record['content'])
            if decoded['errors']:
                result=dict(supported=False,reason='Invalid observation protocol')
            elif not decoded['observations']:
                result=dict(supported=None,reason='No recorded findings to assess')
            else:
                schema=dict(type='object',properties=dict(supported=dict(type='boolean'),reason=dict(type='string')),required=['supported','reason'],additionalProperties=False)
                payload=dict(task='Check whether the recorded findings are supported by this actual interview source. Distinguish self-report from directly observed behavior. Judge source fidelity, not agreement with personality labels.',
                    record=dict(id=key,findings=decoded['observations']),source=observation_view(source))
                result,_=self.backend.generate([dict(role='system',content=SYSTEM,output_schema=schema),message(payload,source.get('media',[]))],'judge',0)
                if not isinstance(result.get('supported'),bool) or not isinstance(result.get('reason'),str):
                    raise ValueError('Judge must return supported:boolean and reason:string')
            checks[key]=dict(result,record_fingerprint=digest(record),source_fingerprint=digest(source))
        schema=dict(type='object',properties={k:dict(type='number',minimum=0,maximum=1) for k in ('faithfulness','citation_correctness')},required=['faithfulness','citation_correctness'],additionalProperties=False)
        views=memory_view(trajectory['memory'],observations=trajectory['observations'])['observations']
        payload=dict(task='Assess the final explanation and contextual expressions against the saved findings and their support checks. Return faithfulness and citation_correctness in [0,1]. '+READ_SUPPORT_RULE,
            final=trajectory['final'],reasoning=trajectory.get('final_explanation'),contextual=trajectory.get('contextual',[]),
            records={view['id']:dict({k:v for k,v in view.items() if k not in ('content','content_status')},findings=view['content'],check=checks[view['id']]) for view in views})
        final,_=self.backend.generate([dict(role='system',content=SYSTEM,output_schema=schema),message(payload)],'judge',0)
        for key in ('faithfulness','citation_correctness'):
            if isinstance(final.get(key),bool) or not isinstance(final.get(key),(int,float)) or not math.isfinite(final[key]) or not 0<=final[key]<=1:
                raise ValueError('Invalid final Judge score')
        return dict(records=checks,final=final,automatic_evaluation=True)
