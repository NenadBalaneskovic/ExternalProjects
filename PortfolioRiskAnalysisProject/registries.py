RISK_MODEL_REGISTRY = {}
PHYSICS_MODEL_REGISTRY = {}
VIEW_REGISTRY = {}
METRIC_REGISTRY = {}


def register_risk_model(name):
    def wrapper(func):
        RISK_MODEL_REGISTRY[name] = func
        return func
    return wrapper


def register_physics_model(name):
    def wrapper(func):
        PHYSICS_MODEL_REGISTRY[name] = func
        return func
    return wrapper


def register_view(name):
    def wrapper(cls):
        VIEW_REGISTRY[name] = cls
        return cls
    return wrapper


def register_metric(name):
    def wrapper(func):
        METRIC_REGISTRY[name] = func
        return func
    return wrapper
