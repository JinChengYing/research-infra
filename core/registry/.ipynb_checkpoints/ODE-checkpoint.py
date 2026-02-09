ODE_REGISTRY = {}

def register_ode(name):
    def decorator(cls):
        ODE_REGISTRY[name] = cls
        return cls
    return decorator

def build_ode(cfg):
    """
    cfg: dict from config.yaml
        name: 'neural_ode'
        params: {dim, hidden_dim, t0, t1, solver}
    """
    name = cfg['name']
    params = cfg.get('params', {})
    return ODE_REGISTRY[name](**params)
