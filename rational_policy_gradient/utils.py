import functools
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, NamedTuple, Sequence, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.linen.initializers import constant, orthogonal
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.wrappers.baselines import LogWrapper, MPELogWrapper
from PIL import Image

from rational_policy_gradient.wrappers import registration_wrapper
from rational_policy_gradient.wrappers.env_wrappers import (
    ConcatenatePlayerSpaces,
    FixSpaceAPI,
    HanabiMod,
)
from rational_policy_gradient.wrappers.world_state import make_world_state_wrapper


class HanabiActor(nn.Module):
    action_dim: int
    activation: str = ""
    hidden_sizes: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, x):
        obs_size = x.shape[-1] - self.action_dim
        obs, avail_actions = x[..., :obs_size], x[..., obs_size:]
        assert avail_actions.shape[-1] == self.action_dim

        # Pass through hidden layers
        embedding = obs
        for i, hidden_size in enumerate(self.hidden_sizes):
            embedding = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
            if i < len(self.hidden_sizes) - 1:  # Apply relu to all but the last hidden layer
                embedding = nn.relu(embedding)

        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)
        return pi


class HanabiCritic(nn.Module):
    action_dim: int
    num_agents: int
    activation: str = ""
    hidden_sizes: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] % 2 == 0
        obs_size = x.shape[-1] // 2 - self.action_dim
        obs0, obs1 = (
            x[..., : obs_size + self.action_dim],
            x[..., obs_size + self.action_dim :],
        )
        assert obs1.shape[-1] == obs_size + self.action_dim
        obs0 = obs0[..., :obs_size]
        obs1 = obs1[..., :obs_size]
        obs = jnp.concatenate([obs0, obs1], axis=-1)

        # Pass through hidden layers
        embedding = obs
        for i, hidden_size in enumerate(self.hidden_sizes):
            embedding = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
            if i < len(self.hidden_sizes) - 1:  # Apply relu to all but the last hidden layer
                embedding = nn.relu(embedding)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding)
        out = jnp.squeeze(jnp.repeat(critic[:, jnp.newaxis], self.num_agents, axis=1))
        return out


class Actor(nn.Module):
    action_dim: int
    activation: str = "tanh"
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Pass through hidden layers
        actor_mean = x
        for hidden_size in self.hidden_sizes:
            actor_mean = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
            actor_mean = activation(actor_mean)

        # Output layer
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    action_dim: int
    num_agents: int
    activation: str = "tanh"
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Pass through hidden layers
        critic = x
        for hidden_size in self.hidden_sizes:
            critic = nn.Dense(hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
            critic = activation(critic)

        # Output layer
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        out = jnp.squeeze(jnp.repeat(critic[:, jnp.newaxis], self.num_agents, axis=1))
        return out


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # add a zero batch dim if it doesn't exist
        if ins.ndim == 1:
            rnn_state = jnp.expand_dims(rnn_state, 0)
            ins = jnp.expand_dims(ins, 0)
            resets = jnp.expand_dims(resets, 0)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return jnp.squeeze(new_rnn_state), jnp.squeeze(y)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, input):
        x, dones = input
        obs_size = x.shape[-1] - self.action_dim
        obs, avail_actions = x[..., :obs_size], x[..., obs_size:]
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    state_features: jnp.ndarray
    info: jnp.ndarray


@dataclass
class RPO_Edge:
    id: str  # node that does optimization, id == source for base edges
    source: str  # player's payoff to be optimized
    target: str  # coplayer to source
    weight: float  # coefficient on payoff, make weight negative to minimize
    source_player_idx: Union[int, Tuple[int]]  # index of source agent in the game, multiple to optimize over both sides
    shaped_base: str = None  # only for manipulator edges, the base that is being shaped


@dataclass
class RPO_Graph:
    bases: Tuple[str]
    manipulators: Tuple[str]
    base_init_paths: Tuple[str]
    manipulator_init_paths: Tuple[str]
    base_edges: Tuple[RPO_Edge]
    manipulator_edges: Tuple[RPO_Edge]
    base_rollouts: Tuple[Tuple[str]]
    manipulator_rollouts: Tuple[Tuple[str]]

    def __init__(
        self,
        bases,
        manipulators,
        base_init_paths,
        manipulator_init_paths,
        base_edges,
        manipulator_edges,
        base_rollouts,
        manipulator_rollouts,
    ):
        self.bases = bases
        self.optimizing_bases = list(set([edge.id for edge in base_edges]))
        self.manipulators = manipulators
        self.base_init_paths = base_init_paths
        self.manipulator_init_paths = manipulator_init_paths
        self.base_edges = base_edges
        self.manipulator_edges = manipulator_edges
        self.base_rollouts = base_rollouts
        self.manipulator_rollouts = manipulator_rollouts
        self.all_rollouts = list(set(base_rollouts + manipulator_rollouts))

    def is_manipulator(self, node):
        return node in self.manipulators

    def make_manipulator_data_tree(self, data, base_params):
        # data is a tuple matching the order of manipulator_rollouts
        # output a pytree with key for each manipulator and values the data and base params from each rollout
        data_map = {}
        for i, (s, t) in enumerate(self.manipulator_rollouts):
            d = tree_slice(data, i)
            data_map[(s, t)] = d
        traj_batch_array = []
        advantages_array = []
        base_params_array = []
        # loop through all edges and gather corresponding data
        for edge in self.manipulator_edges:
            if edge.source_player_idx == 0:
                # add data corresponding to (edge.source, edge.target) and reweight with edge.weight
                d = data_map[(edge.source, edge.target)]
                reward_traj_batch, reward_advantage = tree_slice(d, 0)
                if edge.shaped_base == edge.source:
                    shaping_traj_batch, _ = tree_slice(d, 0)
                else:
                    shaping_traj_batch, _ = tree_slice(d, 1)
            elif edge.source_player_idx == 1:
                # add data corresponding to (edge.target, edge.source) and reweight with edge.weight
                d = data_map[(edge.target, edge.source)]
                reward_traj_batch, reward_advantage = tree_slice(d, 1)
                if edge.shaped_base == edge.source:
                    shaping_traj_batch, _ = tree_slice(d, 1)
                else:
                    shaping_traj_batch, _ = tree_slice(d, 0)
            else:
                raise ValueError("source_player_idx must be 0 or 1")

            # reward and value from reward_data and everything else from shaping_data
            merged_traj_batch = Transition(
                reward=reward_traj_batch.reward,
                value=reward_traj_batch.value,
                log_prob=shaping_traj_batch.log_prob,
                done=shaping_traj_batch.done,
                action=shaping_traj_batch.action,
                obs=shaping_traj_batch.obs,
                state_features=shaping_traj_batch.state_features,
                info=shaping_traj_batch.info,
            )

            base_actor_params = base_params[edge.shaped_base]

            traj_batch_array.append(merged_traj_batch)
            advantages_array.append(edge.weight * reward_advantage)
            base_params_array.append(base_actor_params)

        traj_batch_array = tree_stack(traj_batch_array)
        advantages_array = jnp.stack(advantages_array)
        base_params_array = tree_stack(base_params_array)

        advantages_array = (advantages_array - jnp.mean(advantages_array)) / (jnp.std(advantages_array) + 1e-8)

        return traj_batch_array, advantages_array, base_params_array

    def make_manipulator_rollouts_data_tree(self, data):
        # data is a tuple matching the order of base_rollouts or manipulator_rollouts
        # out put a pytree with keys for each base and values the data that base should train on
        data_map = {}
        for i, (s, t) in enumerate(self.base_rollouts):
            d = tree_slice(data, i)
            data_map[(s, t)] = d
        traj_batch_map = defaultdict(list)
        # loop through all edges and gather corresponding data
        for edge in self.base_edges:
            if not self.is_manipulator(edge.target):
                continue
            manipulator = edge.target
            if edge.source_player_idx == 0:
                # add data corresponding to (edge.source, edge.target) and reweight with edge.weight
                rollout = data_map[(edge.source, edge.target)]
                d = tree_slice(rollout, 1)
            elif edge.source_player_idx == 1:
                # add data corresponding to (edge.target, edge.source) and reweight with edge.weight
                rollout = data_map[(edge.target, edge.source)]
                # swap so source is first
                d = tree_slice(rollout, 0)
            else:
                raise ValueError("source_player_idx must be 0 or 1")

            traj_batch = d
            traj_batch_map[manipulator].append(traj_batch)

        traj_batch_array = []
        pad_mask = []
        max_length = max([len(traj_batch_map[manipulator]) for manipulator in self.manipulators])

        def stack_and_pad(list_, pad_length):
            stack_ = tree_stack(list_) if len(list_) else jax.tree.map(lambda x: jnp.array(x), list_)
            return jax.tree.map(
                lambda x: jnp.pad(x, [(0, pad_length)] + [(0, 0)] * (x.ndim - 1)),
                stack_,
            )

        for manipulator in self.manipulators:
            content_length = len(traj_batch_map[manipulator])
            pad_length = max_length - content_length
            traj_batch_array.append(stack_and_pad(traj_batch_map[manipulator], pad_length))
            pad_mask.append(jnp.concatenate([jnp.ones(content_length), jnp.zeros(pad_length)]))

        traj_batch_array = tree_stack(traj_batch_array)
        pad_mask = jnp.stack(pad_mask)

        return traj_batch_array, pad_mask

    def make_data_tree(self, data, manipulator_params):
        # data is a tuple matching the order of base_rollouts or manipulator_rollouts
        # out put a pytree with keys for each base and values the data that base should train on
        data_map = {}
        for i, (s, t) in enumerate(self.base_rollouts):
            d = tree_slice(data, i)
            data_map[(s, t)] = d
        traj_batch_map = defaultdict(list)
        advantages_map = defaultdict(list)
        shaping_map = defaultdict(list)
        # loop through all edges and gather corresponding data
        for edge in self.base_edges:
            base = edge.id
            if edge.source_player_idx == 0:
                # add data corresponding to (edge.source, edge.target) and reweight with edge.weight
                d = data_map[(edge.source, edge.target)]
            elif edge.source_player_idx == 1:
                # add data corresponding to (edge.target, edge.source) and reweight with edge.weight
                d = data_map[(edge.target, edge.source)]
                # swap so source is first
                d = jax.tree.map(lambda x: jnp.stack([x[1], x[0]]), d)
            else:
                raise ValueError("source_player_idx must be 0 or 1")

            if self.is_manipulator(edge.target):
                shaping_manipulator_params = manipulator_params[edge.target]
            else:
                arbitrary_manipulator = list(self.manipulators)[0]
                shaping_manipulator_params = jax.lax.stop_gradient(
                    jax.tree.map(lambda x: jnp.zeros_like(x), manipulator_params[arbitrary_manipulator])
                )
                # shaping_manipulator_params = jax.lax.stop_gradient(base_params[edge.target])

            traj_batch, advantages = d

            traj_batch_map[base].append(traj_batch)
            advantages_map[base].append(edge.weight * advantages)
            shaping_map[base].append(shaping_manipulator_params)

        for base in self.optimizing_bases:
            # normalize
            advantages_map[base] = normalize_gae(jnp.array(advantages_map[base]))

        traj_batch_array = []
        advantages_array = []
        shaping_array = []
        pad_mask = []
        max_length = max([len(traj_batch_map[base]) for base in self.optimizing_bases])

        def stack_and_pad(list_, pad_length):
            stack_ = tree_stack(list_) if len(list_) else jax.tree.map(lambda x: jnp.array(x), list_)
            return jax.tree.map(
                lambda x: jnp.pad(x, [(0, pad_length)] + [(0, 0)] * (x.ndim - 1)),
                stack_,
            )

        for base in self.optimizing_bases:
            content_length = len(traj_batch_map[base])
            pad_length = max_length - content_length
            traj_batch_array.append(stack_and_pad(traj_batch_map[base], pad_length))
            advantages_array.append(stack_and_pad(advantages_map[base], pad_length))
            shaping_array.append(stack_and_pad(shaping_map[base], pad_length))
            pad_mask.append(jnp.concatenate([jnp.ones(content_length), jnp.zeros(pad_length)]))

        traj_batch_array = tree_stack(traj_batch_array)
        advantages_array = jnp.stack(advantages_array)
        shaping_array = tree_stack(shaping_array)
        pad_mask = jnp.stack(pad_mask)

        return traj_batch_array, advantages_array, shaping_array, pad_mask


def create_rpg_graph(base_inits, manipulator_inits, base_edges, manipulator_edges):
    bases = list(base_inits.keys())
    manipulators = list(manipulator_inits.keys())
    base_init_paths = base_inits.values()
    manipulator_init_paths = manipulator_inits.values()
    flattened_base_edges = []
    base_rollouts = set()
    for edge in base_edges:
        assert edge.id == edge.source, "id and source must match for base edges"
        assert edge.source in bases, f"{edge.source} not in {bases}"
        iter = (edge.source_player_idx,) if not isinstance(edge.source_player_idx, tuple) else edge.source_player_idx
        for idx in iter:
            if idx == 0:
                base_rollouts.add((edge.source, edge.target))
            else:
                base_rollouts.add((edge.target, edge.source))
            flattened_edge = RPO_Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                weight=edge.weight,
                source_player_idx=idx,
                shaped_base=edge.shaped_base,
            )
            flattened_base_edges.append(flattened_edge)

    flattened_manipulator_edges = []
    lookahead_rollouts = set()
    for edge in manipulator_edges:
        assert edge.id in manipulators, f"{edge.id} not in {manipulators}"
        assert edge.source in bases, f"{edge.source} not in {bases}"
        assert edge.target in bases, f"{edge.target} not in {bases}"
        iter = (edge.source_player_idx,) if not isinstance(edge.source_player_idx, tuple) else edge.source_player_idx
        for idx in iter:
            if idx == 0:
                lookahead_rollouts.add((edge.source, edge.target))
            else:
                lookahead_rollouts.add((edge.target, edge.source))
            flattened_edge = RPO_Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                weight=edge.weight,
                source_player_idx=idx,
                shaped_base=edge.shaped_base,
            )
            flattened_manipulator_edges.append(flattened_edge)

    lookahead_rollouts = list(lookahead_rollouts)
    base_rollouts = list(base_rollouts)

    return RPO_Graph(
        bases,
        manipulators,
        base_init_paths,
        manipulator_init_paths,
        flattened_base_edges,
        flattened_manipulator_edges,
        base_rollouts,
        lookahead_rollouts,
    )


# DiCE operator
@jax.jit
def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


@jax.jit
def dice_ratio(
    log_p,
    manipulator_log_p,
    starts,
    lam_past,
):
    stochastic_nodes = manipulator_log_p + log_p

    def weighted_cumsum_step(carry, inputs):
        weighted_cumsum_t_minus_1 = carry
        start_t, stochastic_node_t = inputs

        reset = stochastic_node_t
        next_value = lam_past * weighted_cumsum_t_minus_1 + stochastic_node_t

        weighted_cumsum_t = jnp.where(start_t, reset, next_value)

        return weighted_cumsum_t, weighted_cumsum_t

    starts = starts.at[:, 0].set(True)

    initial_carry = stochastic_nodes[:, 0]
    inputs = (
        jnp.swapaxes(starts[:, 1:], axis1=0, axis2=1),
        jnp.swapaxes(stochastic_nodes[:, 1:], axis1=0, axis2=1),
    )
    _, weighted_cumsum_body = jax.lax.scan(weighted_cumsum_step, initial_carry, inputs)
    weighted_cumsum_body = jnp.swapaxes(weighted_cumsum_body, axis1=0, axis2=1)

    weighted_cumsum = jnp.concatenate([initial_carry[:, None], weighted_cumsum_body], axis=1)

    deps_exclusive = weighted_cumsum - stochastic_nodes

    # CALCULATE ACTOR LOSS
    return magic_box(weighted_cumsum) - magic_box(deps_exclusive)


def tree_slice(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def normalize_gae(gae, mean=1.0):
    gae_mean = jnp.mean(gae)
    gae_std = jnp.std(gae) + 1e-8
    return (gae - gae_mean) / gae_std + mean


def explained_variance(ypred, y, axis=0):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    vary = jnp.var(y, axis=axis)
    result = jnp.where(vary == 0, jnp.nan, 1 - jnp.var(y - ypred, axis=axis) / vary)
    return result


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_smoothed(data, **kwargs):
    window_size = len(data) // 1000
    if window_size > 1:
        smoothed_data = moving_average(data, window_size)
    else:
        smoothed_data = data
    plt.plot(smoothed_data, **kwargs)


def make_rollout_pair(config, num_pairs):
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if "hanabi" in config["ENV_NAME"]:
        actor_class = HanabiActor
        env = HanabiMod(env)
    else:
        actor_class = Actor

    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env)
    else:
        env = LogWrapper(env)

    def rollout_pair(rng, params):
        actor_network = actor_class(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_pairs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        runner_state = (env_state, obsv, rng)

        def _env_step(runner_state, unused):
            env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs0 = last_obs["agent_0"].reshape((num_pairs, -1))
            # jax.tree.map(lambda x : print(x.shape), params)
            # print(params['agent_0']['params']['Dense_0']['kernel'].shape)
            # print(obs0.shape)
            pi0 = jax.vmap(actor_network.apply, in_axes=(0, 0))(params["agent_0"], obs0)
            action0 = pi0.sample(seed=_rng)

            rng, _rng = jax.random.split(rng)

            obs1 = last_obs["agent_1"].reshape((num_pairs, -1))
            pi1 = jax.vmap(actor_network.apply, in_axes=(0, 0))(params["agent_1"], obs1)
            action1 = pi1.sample(seed=_rng)

            env_act = {"agent_0": action0, "agent_1": action1}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_pairs)

            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
            info = jax.tree.map(lambda x: x.reshape((num_pairs, -1)), info)
            transition = {
                "done": done,
                "info": info,
                "env_state": env_state.env_state,
                "reward": reward,
                "action": env_act,
            }
            runner_state = (env_state, obsv, rng)

            return runner_state, transition

        runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config["MAX_ROLLOUT_STEPS"])

        return transitions

    return rollout_pair


def make_rollout_pair_rnn(config, num_pairs):

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if "hanabi" in config["ENV_NAME"]:
        actor_class = ActorRNN
        env = HanabiMod(env)
    else:
        actor_class = ActorRNN

    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env)
    else:
        env = LogWrapper(env)

    def rollout_pair(rng, params):

        actor_network = actor_class(env.action_space(env.agents[0]).n, config=config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_pairs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = {
            "agent_0": ScannedRNN.initialize_carry(num_pairs, config["GRU_HIDDEN_DIM"]),
            "agent_1": ScannedRNN.initialize_carry(num_pairs, config["GRU_HIDDEN_DIM"]),
        }

        last_dones = {
            "agent_0": jnp.zeros((num_pairs), dtype=jnp.bool_),
            "agent_1": jnp.zeros((num_pairs), dtype=jnp.bool_),
        }

        runner_state = (env_state, obsv, init_hstate, last_dones, rng)

        def _env_step(runner_state, unused):
            env_state, last_obs, hstate, last_dones, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs0 = last_obs["agent_0"].reshape((num_pairs, -1))
            # jax.tree.map(lambda x : print(x.shape), params)
            # print(params['agent_0']['params']['Dense_0']['kernel'].shape)
            ac_in0 = (obs0[:, np.newaxis], last_dones["agent_0"][:, np.newaxis])
            hstate0, pi0 = jax.vmap(actor_network.apply, in_axes=(0, 0, 0))(
                params["agent_0"], hstate["agent_0"], ac_in0
            )
            action0 = pi0.sample(seed=_rng).squeeze(axis=1)

            rng, _rng = jax.random.split(rng)

            obs1 = last_obs["agent_1"].reshape((num_pairs, -1))
            ac_in1 = (obs1[:, np.newaxis], last_dones["agent_1"][:, np.newaxis])
            hstate1, pi1 = jax.vmap(actor_network.apply, in_axes=(0, 0, 0))(
                params["agent_1"], hstate["agent_1"], ac_in1
            )
            action1 = pi1.sample(seed=_rng).squeeze(axis=1)

            env_act = {"agent_0": action0, "agent_1": action1}
            new_hstate = {
                "agent_0": hstate0,
                "agent_1": hstate1,
            }

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_pairs)

            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
            info = jax.tree.map(lambda x: x.reshape((num_pairs, -1)), info)
            new_done = {
                "agent_0": done["agent_0"],
                "agent_1": done["agent_1"],
            }
            transition = {
                "done": done,
                "info": info,
                "env_state": env_state.env_state,
                "reward": reward,
                "action": env_act,
            }
            runner_state = (env_state, obsv, new_hstate, new_done, rng)

            return runner_state, transition

        runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config["MAX_ROLLOUT_STEPS"])

        return transitions

    return rollout_pair


def rollout_agents(config, get_action0, get_action1, rng):
    """
    Rollout of agents (not jitted) using arbitrary action functions.
    Action functions can be any model (i.e., a pytorch model, a JAX function, etc.).

    Args:
        rng: Random number generator
        get_action0: Function that takes (obs, rng) and returns action for agent 0
        get_action1: Function that takes (obs, rng) and returns action for agent 1
        env: Environment instance
    Returns:
        List of transitions from the rollout
    """
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env)
    else:
        env = LogWrapper(env)

    # Convert JAX RNG to Python random seed for empirical implementation
    max_steps = config.get("MAX_ROLLOUT_STEPS", 100)

    # INIT ENV - single episode, no batching
    obs, env_state = env.reset(rng)

    transitions = []

    for step in range(max_steps):
        # SELECT ACTIONS using provided functions
        action0 = get_action0(obs["agent_0"], rng)
        action1 = get_action1(obs["agent_1"], rng)

        env_act = {"agent_0": action0, "agent_1": action1}

        # STEP ENV - single step, no batching
        next_obs, next_env_state, reward, done, info = env.step(rng, env_state, env_act)

        transition = {
            "obs": obs,
            "action": env_act,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "info": info,
            "env_state": env_state,
        }

        transitions.append(transition)

        # Update for next iteration
        obs = next_obs
        env_state = next_env_state

        # Break if episode is done
        if done.get("__all__", True):
            break

    return transitions


def wrap_env(env, config):
    if config["ENV_NAME"] in ["overcooked", "storm_2p", "overcooked_v2"]:
        env = FixSpaceAPI(env)
    if config["ENV_NAME"] in ["hanabi"]:
        env = HanabiMod(env)

    if config.get("CONC_SPACES"):
        env = ConcatenatePlayerSpaces(env)

    env = make_world_state_wrapper(config["ENV_NAME"], env)

    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env)
    else:
        env = LogWrapper(env)

    return env


def get_matrix_policy(config, params, agent_idx=0, obs=None):
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    key = jax.random.PRNGKey(0)

    action_space_size = env.action_space(env.agents[agent_idx]).n

    network = Actor(action_space_size, activation=config["ACTIVATION"], hidden_sizes=config["HIDDEN_SIZES"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)

    key, _rng = jax.random.split(key)
    # reset_rng = jax.random.split(_rng, batch_size)
    if obs is None:
        obs, _ = env.reset(_rng)
    pis = network.apply(params, [obs[env.agents[agent_idx]]])

    return pis


def wandb_visualize(config, rpo_graph, params, rng):
    if config["ENV_NAME"] == "normal_form":
        all_params = params["actor"]["base"].copy()
        all_params.update(params["actor"]["manipulator"])
        distrs = {}
        for s, t in rpo_graph.base_rollouts + rpo_graph.manipulator_rollouts:
            if f"{s}_{0}" not in distrs:
                s_pi = get_matrix_policy(config, all_params[s], agent_idx=0)
                distrs[f"{s}_{0}"] = s_pi.probs[0]
            if f"{t}_{1}" not in distrs:
                t_pi = get_matrix_policy(config, all_params[t], agent_idx=1)
                distrs[f"{t}_{1}"] = t_pi.probs[0]

        columns = list(distrs.keys())
        rows = list(zip(*distrs.values()))

        # Create a wandb.Table
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"policies": table})
        return

    if config["ENV_NAME"] == "overcooked":
        config["MAX_ROLLOUT_STEPS"] = 400
        viz = OvercookedVisualizer()

        def log_video(seq, filename):
            viz.animate(seq, agent_view_size=5, filename=f"{wandb.run.dir}/{filename}")

    if config["ENV_NAME"][:5] == "storm" or config["ENV_NAME"] == "storm":
        config["MAX_ROLLOUT_STEPS"] = 5 * config["ENV_KWARGS"]["num_inner_steps"]

        env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        def log_video(seq, filename):
            pics = [Image.fromarray(env.render(s)) for s in seq]
            pics[0].save(
                f"{wandb.run.dir}/{filename}",
                format="GIF",
                save_all=True,
                optimize=True,
                append_images=pics[1:],
                duration=200,
                loop=0,
            )

    if config["ENV_NAME"][:3] == "MPE":
        config["MAX_ROLLOUT_STEPS"] = 200

        env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        def log_video(state_seq, filename):
            viz = MPEVisualizer(env, state_seq)
            viz.animate(save_fname=f"{wandb.run.dir}/{filename}")

    all_actor_params = params["actor"]["base"].copy()
    all_actor_params.update(params["actor"]["manipulator"])

    rollout_params = {
        "agent_0": tree_stack([all_actor_params[a0] for (a0, _) in rpo_graph.all_rollouts]),
        "agent_1": tree_stack([all_actor_params[a1] for (_, a1) in rpo_graph.all_rollouts]),
    }

    rollout_pair = jax.jit(make_rollout_pair(config, len(rpo_graph.all_rollouts)))

    transitions = rollout_pair(rng, rollout_params)
    state_seq = transitions["env_state"]
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    log_dict = {}
    for i, (a0, a1) in enumerate(rpo_graph.all_rollouts):
        filename = f"{a0}-vs-{a1}.gif"
        seq = [tree_slice(state_seq, jnp.s_[s, i]) for s in range(config["MAX_ROLLOUT_STEPS"])]
        log_video(seq, filename)
        log_dict[filename] = wandb.Video(f"{wandb.run.dir}/{filename}", fps=4, format="gif")
    wandb.log(log_dict)


def visualize(config, rpo_graph, params, rng, save_path):
    if config["ENV_NAME"] == "normal_form":
        all_params = params["actor"]["base"].copy()
        all_params.update(params["actor"]["manipulator"])
        distrs = {}
        for s, t in rpo_graph.base_rollouts + rpo_graph.manipulator_rollouts:
            if f"{s}_{0}" not in distrs:
                s_pi = get_matrix_policy(config, all_params[s], agent_idx=0)
                distrs[f"{s}_{0}"] = s_pi.probs[0]
            if f"{t}_{1}" not in distrs:
                t_pi = get_matrix_policy(config, all_params[t], agent_idx=1)
                distrs[f"{t}_{1}"] = t_pi.probs[0]

        columns = list(distrs.keys())
        rows = list(zip(*distrs.values()))

        # Create a wandb.Table
        table = wandb.Table(columns=columns, data=rows)
        wandb.log({"policies": table})
        return

    if config["ENV_NAME"] == "overcooked":
        config["MAX_ROLLOUT_STEPS"] = 400
        viz = OvercookedVisualizer()

        def log_video(seq, filename):
            viz.animate(seq, agent_view_size=5, filename=f"{wandb.run.dir}/{filename}")

    if config["ENV_NAME"][:5] == "storm" or config["ENV_NAME"] == "storm":
        config["MAX_ROLLOUT_STEPS"] = 5 * config["ENV_KWARGS"]["num_inner_steps"]

        env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        def log_video(seq, filename):
            pics = [Image.fromarray(env.render(s)) for s in seq]
            pics[0].save(
                f"{save_path}/{filename}",
                format="GIF",
                save_all=True,
                optimize=True,
                append_images=pics[1:],
                duration=400,
                loop=0,
            )

    if config["ENV_NAME"][:3] == "MPE":
        config["MAX_ROLLOUT_STEPS"] = 200

        env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        def log_video(state_seq, filename):
            viz = MPEVisualizer(env, state_seq)
            viz.animate(save_fname=f"{save_path}/{filename}")

    all_actor_params = params["actor"]["base"].copy()
    all_actor_params.update(params["actor"]["manipulator"])

    rollout_params = {
        "agent_0": tree_stack([all_actor_params[a0] for (a0, _) in rpo_graph.all_rollouts]),
        "agent_1": tree_stack([all_actor_params[a1] for (_, a1) in rpo_graph.all_rollouts]),
    }

    rollout_pair = jax.jit(make_rollout_pair(config, len(rpo_graph.all_rollouts)))

    transitions = rollout_pair(rng, rollout_params)
    state_seq = transitions["env_state"]
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    for i, (a0, a1) in enumerate(rpo_graph.all_rollouts):
        filename = f"{a0}-vs-{a1}.gif"
        seq = [tree_slice(state_seq, jnp.s_[s, i]) for s in range(config["MAX_ROLLOUT_STEPS"])]
        log_video(seq, filename)


def build_wandb_log_rpg_callback(rpo_graph):
    def wandb_log_rpg_callback(metric):
        wandb_metric = {}
        for i, name in enumerate(rpo_graph.manipulator_rollouts):
            wandb_metric[f"Manipulator Returns.{name}"] = metric["manipulator"]["info"]["returned_episode_returns"][i]
            wandb_metric[f"Manipulator Shaped.{name}"] = (
                metric["manipulator"]["info"]["shaped_reward"][i]
                if "shaped_reward" in metric["manipulator"]["info"]
                else 0
            )
            wandb_metric[f"Manipulator Explained Variance.{name}"] = metric["manipulator"]["explained_variance"][i]
        for i, name in enumerate(rpo_graph.base_rollouts):
            wandb_metric[f"Base Agent Returns.{name}"] = metric["base"]["info"]["returned_episode_returns"][i]
            wandb_metric[f"Base Agent Shaped.{name}"] = (
                metric["base"]["info"]["shaped_reward"][i] if "shaped_reward" in metric["base"]["info"] else 0
            )
            wandb_metric[f"Base Agent Explained Variance.{name}"] = metric["base"]["explained_variance"][i]
        wandb_metric["value_loss"] = metric["critic"]["value_loss"]
        wandb.log(wandb_metric)

    return wandb_log_rpg_callback


def build_wandb_checkpoint_rpg_callback(config):
    def wandb_checkpoint_rpg_callback(params):
        pickle.dump(params, open(f"{wandb.run.dir}/model_params.pkl", "wb"))

        param_artifact = wandb.Artifact("model-parameters", type="model")
        metadata = config.copy()
        param_artifact.metadata = metadata
        param_artifact.add_file(f"{wandb.run.dir}/model_params.pkl")
        wandb.log_artifact(param_artifact)

    return wandb_checkpoint_rpg_callback
