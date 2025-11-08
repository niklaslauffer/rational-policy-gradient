from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.hanabi import Hanabi
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete


class ConcatenatePlayerSpaces(MultiAgentEnv):
    def __init__(self, baseEnv):
        assert baseEnv.num_agents == 2, "Concatenation currently only supported for two agent games"
        self.baseEnv = baseEnv
        self.agents = baseEnv.agents
        self.num_agents = baseEnv.num_agents
        agent_0 = self.baseEnv.agents[0]
        conc_n = sum([x.n for x in baseEnv.action_spaces.values()])
        self.action_spaces = {a: Discrete(conc_n) for a in baseEnv.agents}
        base_obs = baseEnv.observation_spaces[agent_0]
        conc_shape = sum([x.shape[0] for x in baseEnv.observation_spaces.values()])
        self.observation_spaces = {
            a: Box(base_obs.low, base_obs.high, (conc_shape,), base_obs.dtype) for a in baseEnv.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def concatenate_obs(self, obs):
        agent_0 = self.baseEnv.agents[0]
        agent_1 = self.baseEnv.agents[1]
        return {
            agent_0: jnp.concatenate((obs[agent_0], jnp.zeros_like(obs[agent_1]))),
            agent_1: jnp.concatenate((jnp.zeros_like(obs[agent_0]), obs[agent_1])),
        }
        # return {agent_0: jnp.concatenate((obs[agent_0], obs[agent_1])), agent_1: jnp.concatenate((obs[agent_0], obs[agent_1]))}

    @partial(jax.jit, static_argnums=(0,))
    def truncate_actions(self, actions):
        agent_0 = self.baseEnv.agents[0]
        agent_1 = self.baseEnv.agents[1]
        cutoff = self.baseEnv.action_space(agent_0).n
        agent_0_act = jnp.where(actions[agent_0] >= cutoff, 0, actions[agent_0])
        agent_1_act = jnp.where(actions[agent_1] < cutoff, 0, actions[agent_1] - cutoff)
        # jax.debug.print("0: trunc {x} to {y}", x=actions[agent_0], y=agent_0_act)
        # jax.debug.print("1: trunc {x} to {y}", x=actions[agent_1], y=agent_1_act)

        return {agent_0: agent_0_act.squeeze(), agent_1: agent_1_act.squeeze()}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = self.baseEnv.reset(key)
        conc_obs = self.concatenate_obs(obs)
        return conc_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        truncated_actions = self.truncate_actions(actions)
        obs, states, rewards, dones, infos = self.baseEnv.step(key, state, truncated_actions)
        conc_obs = self.concatenate_obs(obs)
        return conc_obs, states, rewards, dones, infos


class FixSpaceAPI(MultiAgentEnv):
    def __init__(self, baseEnv):
        self.__class__ = type(baseEnv.__class__.__name__, (self.__class__, baseEnv.__class__), {})
        self.__dict__ = baseEnv.__dict__
        self.baseEnv = baseEnv

    def action_space(self, agent_id="") -> Discrete:
        return self.baseEnv.action_space()

    def observation_space(self, agent_id="") -> Discrete:
        return self.baseEnv.observation_space()


class HanabiMod(MultiAgentEnv):

    def __init__(self, base_hanabi_env):
        self.baseEnv = base_hanabi_env
        self.agents = self.baseEnv.agents
        self.num_agents = self.baseEnv.num_agents
        agent_0 = self.baseEnv.agents[0]
        self.action_spaces = self.baseEnv.action_spaces
        base_obs = self.baseEnv.observation_spaces[agent_0]
        self.observation_spaces = {
            a: Box(low=0, high=1, shape=base_obs.shape + self.action_spaces[a].n, dtype=base_obs.dtype)
            for a in self.baseEnv.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def concatenate_actions_to_obs(self, obs, state):
        avail_actions = self.baseEnv.get_legal_moves(state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        obs = {a: jnp.concatenate((obs[a], avail_actions[a]), axis=-1) for a in self.baseEnv.agents}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = self.baseEnv.reset(key)
        obs = self.concatenate_actions_to_obs(obs, state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        obs, new_states, rewards, dones, infos = self.baseEnv.step(key, state, actions)
        obs = self.concatenate_actions_to_obs(obs, new_states)
        return obs, new_states, rewards, dones, infos
