"""
Testing for normal form games. The particles play columns and the copolicies play rows.
"""

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

from rational_policy_optimization.adv_div_rational import *
from rational_policy_optimization.wrappers import registration_wrapper


class MatrixActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        actor_mean = self.param("policy", lambda rng, shape: jax.random.normal(rng, shape), (self.action_dim,))
        actor_mean = jnp.broadcast_to(actor_mean, (len(x), self.action_dim))

        pi = distrax.Categorical(logits=actor_mean)

        critic = self.param("critic", lambda rng, shape: jax.random.normal(rng, shape), (1,))
        critic = jnp.broadcast_to(critic, (len(x), 1))

        zero_critic = jnp.zeros_like(critic)

        return pi, jnp.squeeze(zero_critic, axis=-1)


def tree_slice(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def probs2params(key, probs, config, agent_idx, simple=False):

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

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


def extract_normal_policy(key, config, params, agent_idx, simple=False):
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n
    if simple:
        network = MatrixActorCritic(action_space_size, activation=config["ACTIVATION"])
    else:
        network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)

    key, _rng = jax.random.split(key)
    # reset_rng = jax.random.split(_rng, batch_size)
    obs, _ = env.reset(_rng)
    pis, _ = jax.vmap(network.apply, in_axes=(0, None))(params, [obs[env.agents[agent_idx]]])

    return pis


def setup_test(payoffs, seed=42):

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_rational_search_normal")
    config = OmegaConf.to_container(config)

    config["ENV_KWARGS"]["payoffs"] = payoffs

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # INIT NETWORK
    network = ActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    init_x = init_x.flatten()

    rng, _rng = jax.random.split(rng)
    # the two comes from having one carrot and one horse for each particles
    params_rng = jax.random.split(_rng, 2 * env.num_agents * config["NUM_PARTICLES"])
    # first half is for horses, second half is for carrots
    network_params_flat = jax.vmap(network.init, in_axes=(0, None))(params_rng, init_x)

    network_params_array = jax.tree.map(
        lambda x: x.reshape((2, env.num_agents, config["NUM_PARTICLES"]) + x.shape[1:]), network_params_flat
    )
    network_params = {"horses": tree_slice(network_params_array, 0), "carrots": tree_slice(network_params_array, 1)}

    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    return env, train_state, config, network, rng


def setup_simple_test(payoffs, config_override={}, seed=42):

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_rational_search_normal.yaml")
    config = OmegaConf.to_container(config)

    config["ENV_KWARGS"]["payoffs"] = payoffs

    # override config
    for key, value in config_override.items():
        config[key] = value

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # INIT NETWORK
    network = MatrixActorCritic(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    init_x = init_x.flatten()

    rng, _rng = jax.random.split(rng)
    # the two comes from having one carrot and one horse for each particles
    params_rng = jax.random.split(_rng, 2 * env.num_agents * config["NUM_PARTICLES"])
    # first half is for horses, second half is for carrots
    network_params_flat = jax.vmap(network.init, in_axes=(0, None))(params_rng, init_x)

    network_params_array = jax.tree.map(
        lambda x: x.reshape((2, env.num_agents, config["NUM_PARTICLES"]) + x.shape[1:]), network_params_flat
    )
    network_params = {"horses": tree_slice(network_params_array, 0), "carrots": tree_slice(network_params_array, 1)}

    # tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    # tx = optax.sgd(config["LR"])
    # tx = optax.adam(.001, eps=1e-5)
    # opt_state = tx.init(network_params)

    # config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = {}
    config["MINIBATCH_SIZE"]["CARROT"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]["CARROT"]
    config["MINIBATCH_SIZE"]["HORSE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]["HORSE"]

    return env, network_params, config, network, rng


def test_step():

    env_state_dim = 1
    env_obs_dim = 1

    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    seed = 34

    env, train_state, config, network, rng = setup_simple_test(payoffs, seed=seed)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_PARTICLES"] * config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    obsv = jax.tree.map(lambda x: x.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), obsv)
    env_state = env_state.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    both_env_step = build_env_step(config, env, network)

    horse_params = train_state.params["horses"]
    carrot_params = train_state.params["carrots"]

    rng, step_rng = jax.random.split(rng)
    step_state = (tree_slice(horse_params, 0), tree_slice(carrot_params, 1), env_state, obsv, step_rng)
    step_state, traj_batch = jax.lax.scan(both_env_step, step_state, None, config["NUM_STEPS"])

    ego_params, alter_params, env_state, obsv, rng = step_state

    # check correct dimensions for everything
    assert env_state.shape == (config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)
    for agent in env.agents:
        assert obsv[agent].shape == (config["NUM_PARTICLES"], config["NUM_ENVS"], env_obs_dim)
    assert traj_batch.action.shape == (config["NUM_STEPS"], env.num_agents, config["NUM_PARTICLES"], config["NUM_ENVS"])
    assert traj_batch.obs.shape == (
        config["NUM_STEPS"],
        env.num_agents,
        config["NUM_PARTICLES"],
        config["NUM_ENVS"],
        env_obs_dim,
    )

    # check that params are unchanged
    jax.tree_util.tree_all(jax.tree.map(jnp.allclose, ego_params, horse_params))
    jax.tree_util.tree_all(jax.tree.map(jnp.allclose, alter_params, carrot_params))

    # check that rewards are computed correctly
    for step_num in range(config["NUM_STEPS"]):
        for particle in range(config["NUM_PARTICLES"]):
            for env_num in range(config["NUM_ENVS"]):
                action = traj_batch.action[step_num, :, particle, env_num]
                reward = traj_batch.reward[step_num, :, particle, env_num]

                expected_reward = payoffs[action[0], action[1]]

                assert jnp.allclose(reward, expected_reward)

    # SETUP ONE-SIDED STEP
    ego_env_step = build_env_step(config, env, network, keep_agent_idxs=0)
    alter_env_step = build_env_step(config, env, network, keep_agent_idxs=1)

    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    obsv = jax.tree.map(lambda x: x.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), obsv)
    env_state = env_state.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    step_state = (tree_slice(horse_params, 0), tree_slice(carrot_params, 1), env_state, obsv, step_rng)
    ego_step_state, ego_traj_batch = jax.lax.scan(ego_env_step, step_state, None, config["NUM_STEPS"])
    alter_step_state, alter_traj_batch = jax.lax.scan(alter_env_step, step_state, None, config["NUM_STEPS"])

    ind_traj_batch = jax.tree.map(lambda x, y: jnp.stack([x, y], axis=1), ego_traj_batch, alter_traj_batch)

    # check equivalence of traj_batch and ind_traj_batch to check if both-sided and one-sided steps are equivalent
    assert jnp.allclose(traj_batch.action, ind_traj_batch.action)
    assert jnp.allclose(traj_batch.reward, ind_traj_batch.reward)
    assert jnp.allclose(traj_batch.obs, ind_traj_batch.obs)
    assert jnp.allclose(traj_batch.log_prob, ind_traj_batch.log_prob)
    assert jnp.allclose(traj_batch.done, ind_traj_batch.done)


def test_inner_step():

    config_override = {"NUM_PARTICLES": 3}

    env_state_dim = 1
    env_obs_dim = 1

    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    seed = 33

    env, params, config, network, rng = setup_simple_test(payoffs, config_override=config_override, seed=seed)

    tx = optax.adam(config["LR"]["OUT_HORSE"])
    opt_state = tx.init(params["horses"])

    config["N_LOOKAHEAD"] = 100

    bias_to_action_0 = jnp.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    bias_to_action_1 = jnp.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])

    bias_params_0 = probs2params(rng, bias_to_action_0, config, 0, simple=True)
    bias_params_1 = probs2params(rng, bias_to_action_1, config, 1, simple=True)

    init_carrot_params = tree_stack([bias_params_0, bias_params_1])

    params["carrots"] = init_carrot_params

    in_lookahead = build_in_lookahead(config, network, env, tx)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, 2 * config["NUM_PARTICLES"] * config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    obsv = jax.tree.map(lambda x: x.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), obsv)
    env_state = env_state.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    runner_state = (params, opt_state, env_state, obsv, rng)

    lookahead_traj_batch, (new_state, new_obs) = in_lookahead(runner_state)

    in_lookahead_update = build_in_lookahead_update(config, network, tx, config["N_LOOKAHEAD"])
    (lookahead_params, out_horse_opt_state) = in_lookahead_update(params, opt_state, lookahead_traj_batch)

    ego_policies = extract_normal_policy(
        rng, config, tree_slice(lookahead_params["horses"], 0), 0, simple=True
    ).probs.squeeze()
    alter_policies = extract_normal_policy(
        rng, config, tree_slice(lookahead_params["horses"], 1), 1, simple=True
    ).probs.squeeze()
    print(ego_policies)
    print(alter_policies)

    ego_targets = jnp.array([[1, 0], [0, 1], [1, 0]])
    alter_targets = jnp.array([[0, 1], [1, 0], [0, 1]])

    assert jnp.allclose(ego_policies, ego_targets, atol=1e-2)
    assert jnp.allclose(alter_policies, alter_targets, atol=1e-2)


def test_outer_loss1():
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.5, 0.5]], [[0.1, 0.9]]])
    target_increase = jnp.array([[-1, 0.05], [-1, -1]])

    test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase)


def test_outer_loss2():
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.9, 0.1]], [[0.5, 0.5]]])
    target_increase = jnp.array([[jnp.nan, jnp.nan], [0.05, jnp.nan]])

    test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase)


def test_outer_loss3():
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.6, 0.4]], [[0.6, 0.4]]])
    target_increase = jnp.array([[0.01, jnp.nan], [0.01, jnp.nan]])

    test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase)


def test_outer_loss3():
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.6, 0.4]], [[0.6, 0.4]]])
    target_increase = jnp.array([[0.01, jnp.nan], [0.01, jnp.nan]])

    test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase)


def test_outer_loss4():
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.1, 0.9]], [[0.9, 0.1]]])
    target_increase = jnp.array([[0.01, jnp.nan], [jnp.nan, 0.01]])

    test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase)


def test_outer_loss_template(payoffs, init_horse, init_carrot, target_increase, seed=42):

    config_override = {"NUM_PARTICLES": 1}

    env_state_dim = 1
    env_obs_dim = 1

    env, train_state, config, network, rng = setup_simple_test(payoffs, config_override, seed=seed)

    bias_params_0 = probs2params(rng, init_horse[0], config, 0, simple=True)
    bias_params_1 = probs2params(rng, init_horse[1], config, 1, simple=True)

    init_horse_params = tree_stack([bias_params_0, bias_params_1])
    train_state.params["horses"] = init_horse_params

    bias_params_0 = probs2params(rng, init_carrot[0], config, 0, simple=True)
    bias_params_1 = probs2params(rng, init_carrot[1], config, 1, simple=True)

    init_carrot_params = tree_stack([bias_params_0, bias_params_1])
    train_state.params["carrots"] = init_carrot_params

    in_lookahead = build_in_lookahead(config, network, env)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, 2 * config["NUM_PARTICLES"] * config["NUM_ENVS"])
    in_obsv, in_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    in_obsv = jax.tree.map(
        lambda x: x.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), in_obsv
    )
    in_env_state = in_env_state.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    runner_state = (train_state, in_env_state, in_obsv, rng)

    past_lookahead_train_state = train_state
    lookahead_traj_batch, _ = in_lookahead(runner_state)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_PARTICLES"] * config["NUM_ENVS"])
    out_obsv, out_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    out_obsv = jax.tree.map(lambda x: x.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), out_obsv)
    out_env_state = out_env_state.reshape((config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    before_carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(train_state.params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    before_carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(train_state.params["carrots"], 1), 0, simple=True
    ).probs.squeeze()

    runner_state = (train_state, out_env_state, out_obsv, rng)

    outer_step = build_outer_step(config, network, env, lookahead_traj_batch, past_lookahead_train_state)

    new_runner_state, _ = outer_step(runner_state, None)

    train_state = new_runner_state[0]

    carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(train_state.params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(train_state.params["carrots"], 1), 0, simple=True
    ).probs.squeeze()

    print("before carrots")
    print(before_carrot_ego_policies)
    print(before_carrot_alter_policies)

    print("after carrots")
    print(carrot_ego_policies)
    print(carrot_alter_policies)

    ego_diff = carrot_ego_policies - before_carrot_ego_policies
    alter_diff = carrot_alter_policies - before_carrot_alter_policies

    # check that the change in probability mass is at least target_increase for all non-nan values
    diff = jnp.stack([ego_diff, alter_diff])
    is_greater = jnp.greater_equal(diff, target_increase)
    assert jnp.where(jnp.isnan(target_increase), True, is_greater).all()


def test_outer_ad_div_loss1():
    seed = 45
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    init_horse = jnp.array([[[0.6, 0.4], [0.4, 0.6]], [[0.6, 0.4], [0.4, 0.6]]])
    target_increase = jnp.array([[[0.01, jnp.nan], [jnp.nan, 0.01]], [[0.01, jnp.nan], [jnp.nan, 0.01]]])

    test_outer_adv_div_loss_template(payoffs, init_horse, init_carrot, target_increase, num_particles=2, seed=seed)


def test_outer_ad_div_loss2():
    seed = 46
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    init_horse = jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
    # first particle should also have incentive to change to decrease its payoff with adversarial partner
    target_increase = jnp.array([[[0.05, jnp.nan], [jnp.nan, 0.05]], [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]])

    test_outer_adv_div_loss_template(payoffs, init_horse, init_carrot, target_increase, num_particles=2, seed=seed)


def test_outer_ad_div_loss3():
    seed = 46
    payoffs = jnp.array(
        [
            [1, 0],
            [0, 0],
        ]
    )
    init_carrot = jnp.array([[[0.5, 0.5]], [[0.5, 0.5]]])
    init_horse = jnp.array([[[0.9, 0.1]], [[0.9, 0.1]]])
    # both sides of carrot should try to increase LR by going to first action
    target_increase = jnp.array([[[0.05, jnp.nan], [0.05, jnp.nan]]])

    test_outer_adv_div_loss_template(payoffs, init_horse, init_carrot, target_increase, num_particles=1, seed=seed)


def test_outer_adv_div_loss_template(payoffs, init_horse, init_carrot, target_increase, num_particles=2, seed=42):

    config_override = {"NUM_PARTICLES": num_particles}

    env_state_dim = 1
    env_obs_dim = 1

    env, params, config, network, rng = setup_simple_test(payoffs, config_override, seed=seed)

    carrot_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["CARROT"]), optax.sgd(learning_rate=config["LR"]["CARROT"])
    )
    in_horse_tx = optax.sgd(learning_rate=config["LR"]["IN_HORSE"])
    out_horse_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["OUT_HORSE"]),
        optax.sgd(learning_rate=config["LR"]["OUT_HORSE"]),
    )

    carrot_opt_state = carrot_tx.init(params["carrots"])
    in_horse_opt_state = in_horse_tx.init(params["horses"])
    out_horse_opt_state = out_horse_tx.init(params["horses"])

    bias_params_0 = probs2params(rng, init_horse[0], config, 0, simple=True)
    bias_params_1 = probs2params(rng, init_horse[1], config, 1, simple=True)

    init_horse_params = tree_stack([bias_params_0, bias_params_1])
    params["horses"] = init_horse_params

    bias_params_0 = probs2params(rng, init_carrot[0], config, 0, simple=True)
    bias_params_1 = probs2params(rng, init_carrot[1], config, 1, simple=True)

    init_carrot_params = tree_stack([bias_params_0, bias_params_1])
    params["carrots"] = init_carrot_params

    in_lookahead = build_in_lookahead(config, network, env, in_horse_tx)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, 2 * config["NUM_PARTICLES"] * config["NUM_ENVS"])
    in_obsv, in_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    in_obsv = jax.tree.map(
        lambda x: x.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), in_obsv
    )
    in_env_state = in_env_state.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))

    in_runner_state = (params.copy(), in_horse_opt_state, in_env_state, in_obsv, rng)

    past_in_opt_state = in_horse_opt_state
    lookahead_rollout_info, lookahead_params, _ = in_lookahead(in_runner_state)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_PARTICLES"] * config["NUM_PARTICLES"] * config["NUM_ENVS"])
    out_obsv, out_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    out_obsv = jax.tree.map(
        lambda x: x.reshape((config["NUM_PARTICLES"], config["NUM_PARTICLES"], config["NUM_ENVS"], env_obs_dim)),
        out_obsv,
    )
    out_env_state = out_env_state.reshape(
        (config["NUM_PARTICLES"], config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)
    )

    before_carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    before_carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 1), 0, simple=True
    ).probs.squeeze()

    in_step = build_in_lookahead_update(config, network, in_horse_tx, config["N_LOOKAHEAD"], inner=True)

    out_runner_state = (params.copy(), carrot_opt_state, out_env_state, out_obsv, rng)
    outer_step = build_adv_div_outer_step(
        config,
        network,
        env,
        lookahead_rollout_info,
        lookahead_params,
        past_in_opt_state,
        in_step,
        carrot_tx,
        out_horse_tx,
    )

    new_runner_state, _ = outer_step(out_runner_state, None)

    new_params = new_runner_state[0]

    carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(new_params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(new_params["carrots"], 1), 0, simple=True
    ).probs.squeeze()

    print("before carrots")
    print(before_carrot_ego_policies)
    print(before_carrot_alter_policies)

    print("after carrots")
    print(carrot_ego_policies)
    print(carrot_alter_policies)

    ego_diff = carrot_ego_policies - before_carrot_ego_policies
    alter_diff = carrot_alter_policies - before_carrot_alter_policies

    # check that the change in probability mass is at least target_increase for all non-nan values
    diff = jnp.stack([ego_diff, alter_diff])
    is_greater = jnp.greater_equal(diff, target_increase)
    assert jnp.where(jnp.isnan(target_increase), True, is_greater).all()


def test_full_loop():

    config_override = {"NUM_PARTICLES": 2}

    env_state_dim = 1
    env_obs_dim = 1

    payoffs = jnp.array([[1, 0.7, -0.1, -1], [-0.1, -1, 1, 0.7], [-10, -10, -10, -10], [-10, -10, -10, -10]])

    seed = 44

    env, params, config, network, rng = setup_simple_test(payoffs, config_override=config_override, seed=seed)

    carrot_tx = optax.sgd(config["LR"])
    out_horse_tx = optax.sgd(config["LR"] / 10)
    in_horse_tx = optax.sgd(config["LR"] / 10)

    txs = carrot_tx, out_horse_tx, in_horse_tx

    carrot_opt_state = carrot_tx.init(params["carrots"])
    horse_opt_state = out_horse_tx.init(params["horses"])

    opt_state = (carrot_opt_state, horse_opt_state)

    # rng, _rng = jax.random.split(rng)
    # reset_rng = jax.random.split(_rng, 2*config["NUM_PARTICLES"]*config["NUM_ENVS"])
    # in_obsv, in_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    # in_obsv = jax.tree.map(lambda x : x.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim)), in_obsv)
    # in_env_state = in_env_state.reshape((2, config["NUM_PARTICLES"], config["NUM_ENVS"], env_state_dim))
    vect_reset = jnp.vectorize(env.reset, signature="(2)->(),(1)")

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, (2, config["NUM_PARTICLES"], config["NUM_ENVS"]))
    in_obsv, in_env_state = vect_reset(reset_rng)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, (config["NUM_PARTICLES"], config["NUM_PARTICLES"], config["NUM_ENVS"]))
    out_obsv, out_env_state = vect_reset(reset_rng)

    before_carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    before_carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 1), 0, simple=True
    ).probs.squeeze()

    before_horse_ego_policies = extract_normal_policy(
        rng, config, tree_slice(params["horses"], 0), 0, simple=True
    ).probs.squeeze()
    before_horse_alter_policies = extract_normal_policy(
        rng, config, tree_slice(params["horses"], 1), 1, simple=True
    ).probs.squeeze()

    full_runner_state = (params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, rng)

    _update_step = build_step_all(config, network, env, txs)

    new_runner_state, _ = jax.lax.scan(_update_step, full_runner_state, None, 1000)

    params = new_runner_state[0]

    print("before horses")
    print(before_horse_ego_policies)
    print(before_horse_alter_policies)

    print("after horses")
    ego_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 0), 0, simple=True).probs.squeeze()
    alter_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 1), 1, simple=True).probs.squeeze()
    print(ego_policies)
    print(alter_policies)

    print("before carrots")
    print(before_carrot_ego_policies)
    print(before_carrot_alter_policies)

    print("after carrots")
    carrot_ego_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 0), 1, simple=True
    ).probs.squeeze()
    carrot_alter_policies = extract_normal_policy(
        rng, config, tree_slice(params["carrots"], 1), 0, simple=True
    ).probs.squeeze()
    print(carrot_ego_policies)
    print(carrot_alter_policies)


def test_make_train():

    config_override = {"NUM_PARTICLES": 2}

    payoffs = jnp.array([[1, 0.7, -0.1, -1], [-0.1, -1, 1, 0.7], [-10, -10, -10, -10], [-10, -10, -10, -10]])

    seed = 60

    _, _, config, _, rng = setup_simple_test(payoffs, config_override=config_override, seed=seed)

    train = jax.jit(make_train(config))

    out = train(rng)

    params = out["runner_state"][0]

    print("after horses")
    ego_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 0), 0).probs.squeeze()
    alter_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 1), 1).probs.squeeze()
    print(ego_policies)
    print(alter_policies)

    print("after carrots")
    carrot_ego_policies = extract_normal_policy(rng, config, tree_slice(params["carrots"], 0), 1).probs.squeeze()
    carrot_alter_policies = extract_normal_policy(rng, config, tree_slice(params["carrots"], 1), 0).probs.squeeze()
    print(carrot_ego_policies)
    print(carrot_alter_policies)


def test_plot_metrics():

    config_override = {"NUM_PARTICLES": 1}

    # payoffs = jnp.array(
    #     [[1, .7, 0, -1],
    #     [0, -1, 1, .7],
    #     [-10, -10, -10, -10],
    #     [-10, -10, -10, -10]]
    # )

    payoffs = jnp.array(
        [[1, 0], [0, 0]],
    )

    seed = 47

    _, _, config, _, rng = setup_simple_test(payoffs, config_override=config_override, seed=seed)

    train = jax.jit(make_train(config))

    out = train(rng)

    params = out["runner_state"][0]
    metrics = out["metrics"]

    exp_dir = "tests/test_plots"
    filename = "metrics"

    print("after horses")
    ego_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 0), 0).probs.squeeze()
    alter_policies = extract_normal_policy(rng, config, tree_slice(params["horses"], 1), 1).probs.squeeze()
    print(ego_policies)
    print(alter_policies)

    print("after carrots")
    carrot_ego_policies = extract_normal_policy(rng, config, tree_slice(params["carrots"], 0), 1).probs.squeeze()
    carrot_alter_policies = extract_normal_policy(rng, config, tree_slice(params["carrots"], 1), 0).probs.squeeze()
    print(carrot_ego_policies)
    print(carrot_alter_policies)

    plot_run(metrics, exp_dir, filename)


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update('jax_disable_jit', True)
    # jnp.set_printoptions(precision=3, suppress=True)
    start = time.time()
    # with jax.profiler.trace("traces/normal"):
    test_make_train()
    print("total time", time.time() - start)
