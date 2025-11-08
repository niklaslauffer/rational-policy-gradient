from math import prod

import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


def make_world_state_wrapper(env_name: str, env: MultiAgentEnv):
    if env_name == "normal_form":
        return NormalWorldStateWrapper(env)
    if env_name == "extensive_form":
        return ExtensiveWorldStateWrapper(env)
    if env_name[:5] == "storm":
        return ConcObsWorldStateWrapper(env)
    if env_name == "hanabi":
        return HanabiWorldStateWrapper(env)
    if env_name == "hanabi_modded":
        return HanabiModdedWorldStateWrapper(env)
    if env_name in ["overcooked", "overcooked_v2"]:
        return StateWorldStateWrapper(env)
    if env_name[:3] == "MPE":
        return MPEWorldStateWrapper(env)
    else:
        raise ValueError(f"Unknown world state wrapper: {env_name}")


class NormalWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        return state.env_state

    def state_feature_size(self):
        return 1


class ExtensiveWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        return state.env_state

    def state_feature_size(self):
        return self.env.init_state.shape


class ConcObsWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        return jnp.concatenate([obs[a] for a in self.env.agents], axis=-1)

    def state_feature_size(self):
        return (sum([self.env.observation_space(a).shape for a in self.env.agents]),)


class HanabiWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        obs = jnp.concatenate([obs[a] for a in self.env.agents], axis=-1)
        return obs
        # jax.debug.print("{x}", x=[obs[a].flatten() for a in self.env.agents])
        # return jnp.concatenate([obs[a].flatten() for a in self.env.agents], axis=-1)

    def state_feature_size(self):
        return (sum([self.env.observation_space(a).shape for a in self.env.agents]),)


class MPEWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        return jnp.concatenate([obs[a] for a in self.env.agents], axis=-1)

    def state_feature_size(self):
        return (sum([self.env.observation_space(a).shape[0] for a in self.env.agents]),)


class StateWorldStateWrapper:
    def __init__(self, env: MultiAgentEnv):
        self.env = env
        self.feature_size = jnp.prod(jnp.array(self.env.observation_space().shape))

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def get_state_features(self, state, obs):
        return obs[self.env.agents[0]].reshape((-1, self.feature_size))

    def state_feature_size(self):
        return self.feature_size
