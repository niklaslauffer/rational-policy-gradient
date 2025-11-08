import jaxmarl

from rational_policy_gradient.extra_environments import (
    ExtensiveForm,
    InTheGrid_2p_independent,
    InTheGrid_2p_sabotage,
    InTheGrid_2p_small,
    NormalForm,
)


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        return jaxmarl.make(env_id, **env_kwargs)
    # Normal form game
    elif env_id == "normal_form":
        env = NormalForm(**env_kwargs)
    elif env_id == "extensive_form":
        env = ExtensiveForm(**env_kwargs)
    elif env_id == "storm_2p_small":
        env = InTheGrid_2p_small(**env_kwargs)
    elif env_id == "storm_2p_sabotage":
        env = InTheGrid_2p_sabotage(**env_kwargs)
    elif env_id == "storm_independent":
        env = InTheGrid_2p_independent(**env_kwargs)
    return env


registered_envs = [
    "normal_form",
    "extensive_form",
    "storm_2p_small",
    "storm_2p_sabotage",
    "storm_independent",
    "hanabi_modded",
]
