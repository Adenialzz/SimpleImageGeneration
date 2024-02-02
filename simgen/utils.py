import yaml
import argparse
import datetime

def load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def time_cost(latency):
    time_delta = datetime.timedelta(seconds=int(latency))
    return str(time_delta)
