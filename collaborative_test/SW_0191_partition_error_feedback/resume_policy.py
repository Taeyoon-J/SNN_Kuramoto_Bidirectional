"""Compare logical-device compatibility, record physical placement separately."""
POLICY='same_software_hardware_math_flags_and_logical_cuda0_physical_index_may_change'


def compatible_runtime(signature):
    result=dict(signature)
    physical=result.pop('cuda_visible_devices',None)
    if physical not in (None,'','0','1','2','3'):
        raise ValueError('one physical GPU mapped to logical cuda0 required')
    result['sw0191_resume_policy']=POLICY
    return result
