import inspect

def get_agent_params(agent_class):
    '''

    input: agent_class (type:class)
    return agent_param (type:dict)
    
    '''
    param_dict = {}
    signature = inspect.signature(agent_class.__init__)
    for key,value in signature.parameters.items():
        if key not in ['self','env']:
            value = str(value).split('=')[1]
            try:
                param_dict[key] = float(value)
            except:
                param_dict[key] = bool(value)
    return param_dict