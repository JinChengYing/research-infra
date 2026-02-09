# infra/registry.py

"""
Central registry for research components.

This file should NEVER import torch, numpy, or any method/problem code.
Its only job is to map string names to classes.
"""

# name -> class
METHODS = {}
PROBLEMS = {}


def register_method(name: str):
    """
    Register a Method implementation.

    Usage:
        @register_method("cnf")
        class CNF(Method):
            ...
    """
    def wrapper(cls):
        if name in METHODS:
            raise KeyError(f"Method '{name}' already registered")
        METHODS[name] = cls
        return cls
    return wrapper


def register_problem(name: str):
    """
    Register a Problem definition.

    Usage:
        @register_problem("gmm_2d")
        class GMM2D(Problem):
            ...
    """
    def wrapper(cls):
        if name in PROBLEMS:
            raise KeyError(f"Problem '{name}' already registered")
        PROBLEMS[name] = cls
        return cls
    return wrapper
