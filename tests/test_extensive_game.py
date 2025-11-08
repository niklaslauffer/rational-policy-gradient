"""
Testing for normal form games. The particles play columns and the copolicies play rows.
"""

import itertools
import os
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from rational_policy_optimization.adv_div_rational_param_share import *
from rational_policy_optimization.utils import *


def tree_slice(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def probs2params(key, probs, config, agent_idx, simple=False):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n
    if simple:
        network = MatrixActorCritic(action_space_size, activation=config["ACTIVATION"])
    else:
        network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    init_params = network.init(key_a, init_x)

    params = jax.tree.map(lambda x: jnp.zeros(probs.shape[0:1] + x.shape), init_params)

    if simple:
        params["params"]["policy"] = jnp.log(probs)
    else:
        params["params"]["Dense_2"]["bias"] = jnp.log(probs)

    return params


def extract_normal_policy(key, config, params, agent_idx, simple=False, obs=None):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n
    if simple:
        network = MatrixActor(action_space_size, activation=config["ACTIVATION"])
    else:
        network = Actor(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)

    key, _rng = jax.random.split(key)
    # reset_rng = jax.random.split(_rng, batch_size)
    if obs is None:
        obs, _ = env.reset(_rng)
    pis = jax.vmap(network.apply, in_axes=(0, None))(params, [obs[env.agents[agent_idx]]])

    return pis


def build_game1(config):
    def get_transition(state, actions):
        accept_actions = [(0, 1), (0, 0), (1, 0), (1, 1)]
        if state < accept_state:
            if actions == accept_actions[state % len(accept_actions)]:
                return state + 1, False
            else:
                return accept_state + state + 1, False
        if state == accept_state:
            return state, True
        elif state == reject_state:
            return state, True
        else:
            return state + 1, False

    num_steps = 5
    accept_state = num_steps
    reject_state = 2 * num_steps
    num_states = 2 * num_steps + 1
    num_actions = (2, 2)
    transitions = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.int32)
    dones = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.bool)
    payoffs = jnp.full((num_states, num_actions[0], num_actions[1]), 0)
    payoffs = payoffs.at[accept_state, :, :].set(1.0)
    # payoffs = payoffs.at[0,0,:].set(1.0)

    for s, a0, a1 in itertools.product(range(num_states), range(num_actions[0]), range(num_actions[1])):
        t, d = get_transition(s, (a0, a1))
        transitions = transitions.at[s, a0, a1].set(t)
        dones = dones.at[s, a0, a1].set(d)

    return payoffs, transitions, dones, num_actions, num_states


def build_game2(config):
    def get_transition(state, actions):
        if state == 0:
            if actions[0] == actions[1]:
                return state + 1, False
            else:
                return state + num_steps + 1, False
        if state == accept_state:
            return state, True
        elif state == reject_state:
            return state, True
        else:
            return state + 1, False

    num_steps = 2
    accept_state = num_steps
    reject_state = 2 * num_steps
    num_states = 2 * num_steps + 1
    num_actions = (2, 2)
    transitions = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.int32)
    dones = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.int32)
    payoffs = jnp.zeros((num_states, num_actions[0], num_actions[1]))
    payoffs = payoffs.at[accept_state, :, :].set(1)

    for s, a0, a1 in itertools.product(range(num_states), range(num_actions[0]), range(num_actions[1])):
        t, d = get_transition(s, (a0, a1))
        transitions = transitions.at[s, a0, a1].set(t)
        dones = dones.at[s, a0, a1].set(d)

    return payoffs, transitions, dones, num_actions, num_states


def build_game3(config):
    def get_transition(state, actions):
        if state == accept_state:
            return state, True
        elif state == reject_state:
            return state, True
        elif state < accept_state:
            if actions[0] == (actions[1] + state) % num_actions[0]:
                return state + 1, False
            else:
                return state + num_steps + 1, False
        else:
            return state + 1, False

    num_steps = 2
    accept_state = num_steps
    reject_state = 2 * num_steps
    num_states = 2 * num_steps + 1
    num_actions = (2, 2)
    transitions = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.int32)
    dones = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.bool)
    payoffs = jnp.zeros((num_states, num_actions[0], num_actions[1]), dtype=jnp.int32)
    payoffs = payoffs.at[accept_state, :, :].set(1)

    for s, a0, a1 in itertools.product(range(num_states), range(num_actions[0]), range(num_actions[1])):
        t, d = get_transition(s, (a0, a1))
        transitions = transitions.at[s, a0, a1].set(t)
        dones = dones.at[s, a0, a1].set(d)

    return payoffs, transitions, dones, num_actions, num_states


def test_plot_metrics():

    seed = 51

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_rational_search_extensive.yaml")
    config = OmegaConf.to_container(config)

    payoffs, transitions, dones, num_actions, num_states = build_game2(config)

    config["ENV_KWARGS"]["payoffs"] = payoffs
    config["ENV_KWARGS"]["transitions"] = transitions
    config["ENV_KWARGS"]["dones"] = dones
    config["ENV_KWARGS"]["num_actions"] = num_actions
    config["ENV_KWARGS"]["num_states"] = num_states

    train = jax.jit(make_train(config))

    out = train(rng)

    params = out["runner_state"][0]
    metrics = out["metrics"]

    return_matrix = metrics["carrot"]["info"]["returned_episode_returns"].mean(axis=1)

    exp_dir = "tests/game2"
    filename = f"metrics_{config['NUM_STEPS']}_actions{num_actions[0]}"
    import os

    os.makedirs(f"{exp_dir}/{filename}", exist_ok=True)
    plot_run(metrics, exp_dir, filename)

    print("after horses")
    ego_policies = extract_normal_policy(rng, config, tree_slice(params["actor"]["horse"], 0), 0).probs.squeeze()
    alter_policies = extract_normal_policy(rng, config, tree_slice(params["actor"]["horse"], 1), 1).probs.squeeze()
    print(ego_policies)
    print(alter_policies)

    print("after carrots")
    # carrot_ego_policies = extract_normal_policy(rng, config, tree_slice(params['carrots'], 0), 1).probs.squeeze()
    # carrot_alter_policies = extract_normal_policy(rng, config, tree_slice(params['carrots'], 1), 0).probs.squeeze()
    # print(carrot_ego_policies)
    # print(carrot_alter_policies)
    for i in range(num_states // 2):
        print(f"state {i}")
        s = jax.nn.one_hot(i, num_states)
        print(
            extract_normal_policy(
                rng, config, tree_slice(params["actor"]["carrot"], 0), 1, obs={"agent_0": s, "agent_1": s}
            ).probs.squeeze()
        )
        print(
            extract_normal_policy(
                rng, config, tree_slice(params["actor"]["carrot"], 1), 0, obs={"agent_0": s, "agent_1": s}
            ).probs.squeeze()
        )


def test_multi_particle():

    seed = 52

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_rational_search_extensive.yaml")
    config = OmegaConf.to_container(config)

    # def payoffs(state, actions):
    #     state = state.squeeze()
    #     payoff_vec = jnp.array([0,0,1,0])
    #     return payoff_vec[state]

    # payoffs = jnp.array([0,0,1])
    # transitions = jnp.array([
    #     [[2, 1],
    #      [1, 2]],
    #     [[1, 1],
    #      [1, 1]],
    #     [[2, 2],
    #      [2, 2]],
    # ])

    # dones = jnp.array([
    #     [[False, False],
    #      [False, False]],
    #     [[True, True],
    #      [True, True]],
    #     [[True, True],
    #      [True, True]],
    # ])

    payoffs, transitions, dones, num_actions, num_states = build_game2(config)

    config["ENV_KWARGS"]["payoffs"] = payoffs
    config["ENV_KWARGS"]["transitions"] = transitions
    config["ENV_KWARGS"]["dones"] = dones
    config["ENV_KWARGS"]["num_actions"] = num_actions
    config["ENV_KWARGS"]["num_states"] = num_states

    train = jax.jit(make_train(config))

    out = train(rng)

    params = out["runner_state"][0]
    metrics = out["metrics"]

    exp_dir = "tests/game2/param_share"
    filename = f"metrics_{config['NUM_STEPS']}_actions{num_actions[0]}"
    import os

    os.makedirs(f"{exp_dir}/{filename}", exist_ok=True)
    plot_run(metrics, exp_dir, filename)

    print("after horses")
    ego_policies = extract_normal_policy(rng, config, tree_slice(params["actor"]["horse"], 0), 0).probs.squeeze()
    print(ego_policies)

    print("after carrots")
    # carrot_ego_policies = extract_normal_policy(rng, config, tree_slice(params['carrots'], 0), 1).probs.squeeze()
    # carrot_alter_policies = extract_normal_policy(rng, config, tree_slice(params['carrots'], 1), 0).probs.squeeze()
    # print(carrot_ego_policies)
    # print(carrot_alter_policies)
    for i in range(num_states // 2):
        print(f"state {i}")
        s = jax.nn.one_hot(i, num_states)
        print(
            extract_normal_policy(
                rng, config, tree_slice(params["actor"]["carrot"], 0), 1, obs={"agent_0": s, "agent_1": s}
            ).probs.squeeze()
        )


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update('jax_disable_jit', True)
    # jnp.set_printoptions(suppress=True)
    wandb.init(mode="disabled")
    start = time.time()
    # with jax.profiler.trace("traces/normal"):
    test_multi_particle()
    print("total time", time.time() - start)
