from functools import partial
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete


class NormalForm(MultiAgentEnv):
    def __init__(
        self,
        payoffs,
    ):
        self.num_agents = len(payoffs.shape) - 1
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.num_actions = payoffs.shape[: self.num_agents]

        self.payoffs = payoffs

        # self.action_spaces = {agent: Discrete(payoffs.shape[i]) for i, agent in enumerate(self.agents)}
        self.action_spaces = {
            agent: Discrete(max(self.num_actions)) for agent in self.agents
        }

        fixed_obs_array = jnp.array([[0, 1], [1, 0]], dtype=jnp.float32)
        self.fixed_obs = {a: o for a, o in zip(self.agents, fixed_obs_array)}

        self.observation_spaces = {
            i: Box(low=0, high=1, shape=(self.num_agents,)) for i in self.agents
        }

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Any]:
        obs = self.fixed_obs
        state = jnp.array([0])
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: Any, actions: dict):
        # get the actions as array

        actions = tuple(
            actions[a] % num_a for a, num_a in zip(self.agents, self.num_actions)
        )
        # noise = jax.random.normal(key) * 0.00001
        noise = 0
        reward = self.payoffs[actions] + noise
        obs = self.fixed_obs
        state = jnp.array([0])
        rewards = {a: reward[i] for i, a in enumerate(self.agents)}
        dones = {a: True for a in self.agents + ["__all__"]}
        info = {}

        return obs, state, rewards, dones, info

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def action_space(self, agent_id="") -> Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""

        return self.action_spaces[agent_id]

    def observation_space(self, agent_id="") -> Discrete:
        """Observation space of the environment."""
        return self.observation_spaces[agent_id]


class ExtensiveForm(MultiAgentEnv):
    def __init__(
        self,
        payoffs,
        transitions,
        dones,
        num_actions,
        num_states,
    ):
        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        # self.init_state = jnp.array([0], dtype=jnp.int32)
        self.init_state = jax.nn.one_hot(0, num_states)
        self.num_states = num_states

        self.payoffs = payoffs
        self.transitions = transitions
        self.dones = dones

        self.action_spaces = {
            agent: Discrete(num_actions[i]) for i, agent in enumerate(self.agents)
        }

        # self.observation_spaces = {i: Discrete(num_states) for i in self.agents}
        self.observation_spaces = {
            i: Box(low=0, high=1, shape=(num_states,)) for i in self.agents
        }

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Any]:
        state = self.init_state
        obs = {a: state for a in self.agents}
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: Any, actions: dict):
        # get the actions as array

        actions = tuple(actions[a] for a in self.agents)
        # noise = jax.random.normal(key) * 0.001
        noise = 0
        state = jnp.argwhere(state, size=1).squeeze()
        reward = self.payoffs[state, actions[0], actions[1]] + noise
        done = self.dones[state, actions[0], actions[1]]
        state_idx = self.transitions[state, actions[0], actions[1]]
        state = jax.nn.one_hot(state_idx, self.num_states)
        obs = {a: state for a in self.agents}
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents + ["__all__"]}
        info = {}

        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            jax.lax.stop_gradient(rewards),
            dones,
            info,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def action_space(self, agent_id="") -> Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""

        return self.action_spaces[agent_id]

    def observation_space(self, agent_id="") -> Discrete:
        """Observation space of the environment."""
        return self.observation_spaces[agent_id]
