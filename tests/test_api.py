"""
Testing for RPO API.
"""

import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest
import wandb
from omegaconf import OmegaConf

from rational_policy_gradient import utils
from rational_policy_gradient.rpg import *
from rational_policy_gradient.rpg_algs import *
from rational_policy_gradient.wrappers import registration_wrapper


def tree_slice(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def probs2params(key, probs, config, agent_idx):

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n

    network = utils.Actor(action_space_size, activation=config["ACTIVATION"], hidden_sizes=config["HIDDEN_SIZES"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    init_params = network.init(key_a, init_x)

    params = jax.tree.map(lambda x: jnp.zeros(x.shape), init_params)

    params["params"]["Dense_2"]["bias"] = jnp.log(probs)

    return params


def extract_normal_policy(config, params, agent_idx, obs=None):
    key = jax.random.PRNGKey(0)
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n

    network = utils.Actor(action_space_size, activation=config["ACTIVATION"], hidden_sizes=config["HIDDEN_SIZES"])

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    key, _rng = jax.random.split(key)
    if obs is None:
        obs, _ = env.reset(_rng)
    pis = network.apply(params, [obs[env.agents[agent_idx]]])

    return pis


def extract_normal_value(config, params):
    key = jax.random.PRNGKey(0)
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = make_world_state_wrapper("normal_form", env)
    env = LogWrapper(env)

    network = utils.Critic(None, num_agents=2, activation=config["ACTIVATION"])

    init_x = jnp.zeros(env.state_feature_size())
    init_x = init_x.flatten()

    key, _rng = jax.random.split(key)

    obs, state = env.reset(_rng)
    s_features = env.get_state_features(state, obs)
    pis = network.apply(params, [s_features])

    return pis


def test_in_lookahead(seed=42):

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_doublesided_RAP(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, params, actor_network, critic_network, txs, opt_state, rollout_status = initialize_training(
        config, env, rpo_config, _rng
    )

    init_horse_probs = jnp.array([[0.01, 0.99], [0.5, 0.5]])
    init_horse_params = probs2params(rng, init_horse_probs, config, 0)

    init_carrot_probs = jnp.array([[0.99, 0.01]])
    init_carrot_params = probs2params(rng, init_carrot_probs, config, 1)

    params["actor"]["horse"] = {k: tree_slice(init_horse_params, i) for i, k in enumerate(rpo_graph.horses)}
    params["actor"]["carrot"] = {k: tree_slice(init_carrot_params, i) for i, k in enumerate(rpo_graph.carrots)}

    in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv = rollout_status
    out_carrot_tx, out_horse_tx, in_horse_tx, critic_tx = txs
    carrot_opt_state, horse_opt_state, critic_opt_state = opt_state

    in_lookahead = build_in_lookahead(config, rpo_graph, actor_network, critic_network, env, in_horse_tx)

    rng, in_rng = jax.random.split(rng)
    in_runner_state = (params.copy(), horse_opt_state, in_env_state, in_obsv, reg_env_state, reg_obsv, in_rng)

    lookahead_rollout_info, lookahead_params, in_value_update_info, in_env_state, in_obsv = in_lookahead(
        in_runner_state
    )

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(config, lookahead_params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(config, lookahead_params["actor"]["carrot"][c], 1).probs.squeeze())


def test_out_step(seed=2):

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    config["NUM_ENVS"] = 2
    config["NUM_STEPS"] = 2
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_doublesided_RAP(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, params, actor_network, critic_network, txs, opt_state, rollout_status, rew_shaping_schedule = (
        initialize_training(config, env, rpo_config, _rng)
    )

    init_policies = {
        "carrot": {
            "adversary_carrot": jnp.array([0.5, 0.5]),
        },
        "horse": {
            "victim_horse": jnp.array([0.01, 0.99]),
            "adversary_horse": jnp.array([0.5, 0.5]),
        },
    }
    init_agent = {
        "carrot": {
            "adversary_carrot": 1,
        },
        "horse": {
            "victim_horse": 0,
            "adversary_horse": 0,
        },
    }
    # init_horse_probs = jnp.array([[[0.01, 0.99], [0.5, 0.5]]])
    # init_horse_params = probs2params(rng, init_horse_probs, config, 0)

    # init_carrot_probs = jnp.array([[[0.5, 0.5]]])
    # init_carrot_params = probs2params(rng, init_carrot_probs, config, 1)
    update_step = 0

    params["actor"] = jax.tree.map(lambda x, y: probs2params(rng, x, config, y), init_policies, init_agent)

    # params["actor"]["horse"] = {k: tree_slice(init_horse_params, i) for i, k in enumerate(rpo_graph.horses)}
    # params["actor"]["carrot"] = {k: tree_slice(init_carrot_params, i) for i, k in enumerate(rpo_graph.carrots)}

    in_env_state, in_obsv, out_env_state, out_obsv = rollout_status
    out_carrot_tx, out_horse_tx, in_horse_tx, critic_tx = txs
    carrot_opt_state, horse_opt_state, critic_opt_state = opt_state

    # save opt_state for horse lookahead later
    # past_in_opt_state = horse_opt_state[1]
    in_opt_state = horse_opt_state[1]

    in_lookahead = build_in_lookahead(config, rpo_graph, actor_network, critic_network, env, in_horse_tx)
    in_lookahead_update = build_in_lookahead_update(config, rpo_graph, actor_network, in_horse_tx)

    rng, in_rng, out_rng = jax.random.split(rng, 3)
    in_runner_state = (deepcopy(params), in_opt_state, in_env_state, in_obsv, update_step, in_rng)

    lookahead_params, lookahead_rollout_info, in_value_update_info, horse_in_metrics, in_env_state, in_obsv = (
        in_lookahead(in_runner_state)
    )

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(config, lookahead_params["actor"]["horse"][h], 0).probs.squeeze())

    # get rollouts for everything in rpo_graph.carrot_rollouts using new horse actor params
    # and perform update for carrot using rpo_graph.carrot_edges
    out_runner_state = (deepcopy(params), carrot_opt_state, out_env_state, out_obsv, update_step, out_rng)

    # build outer step using lookahead_traj_batch, past_lookahead_train_state
    out_step = build_out_step(
        config,
        rpo_graph,
        actor_network,
        critic_network,
        env,
        lookahead_rollout_info,
        lookahead_params,
        in_opt_state,
        in_lookahead_update,
        out_carrot_tx,
    )
    out_runner_state, out_value_update_info, carrot_metrics = out_step(out_runner_state)

    params, carrot_opt_state, out_env_state, out_obsv, rng = out_runner_state
    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(config, params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(config, params["actor"]["carrot"][c], 1).probs.squeeze())


def test_RAT_loop(seed=53):

    wandb.init(mode="disabled")

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_singlesided_RAP(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, _, _, _, _, _, _, _ = initialize_training(config, env, rpo_config, _rng)

    train = make_train(config, rpo_config)

    rng, train_rng = jax.random.split(rng)
    out = train(train_rng)

    params = out["runner_state"][0]

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(rng, config, params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(rng, config, params["actor"]["carrot"][c], 1).probs.squeeze())


def run_alg(seed, payoff, rpg_alg, init_policies, target_policies, num_updates=1, atol=1e-2):
    """
    Test the final policies of the algorithm.
    :param seed: Random seed
    :param payoff: Payoff matrix
    :param rpg_alg: Algorithm to test
    :param target_policies: Target policies for each agent
    :param init_policies: Initial policies for each agent
    :param atol: Absolute tolerance for policy comparison
    :return: None
    """

    wandb.init(mode="disabled")

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    config["RPPO_ALG"] = rpg_alg
    rpo_config = make_rpg_alg(config)

    rng, _rng = jax.random.split(rng)
    (
        rpo_graph,
        params,
        actor_network,
        critic_network,
        txs,
        opt_state,
        rollout_status,
        rew_shaping_schedule,
    ) = initialize_training(config, env, rpo_config, _rng)

    if init_policies:
        params["actor"] = jax.tree.map(lambda x: probs2params(rng, x, config, 0), init_policies)

    update_step = 0
    rng, _rng = jax.random.split(rng)
    full_runner_state = (params, opt_state, rollout_status, update_step, _rng)

    _update_step = build_full_step(config, rpo_graph, actor_network, critic_network, env, txs, rew_shaping_schedule)

    runner_state, metric = jax.lax.scan(_update_step, full_runner_state, None, num_updates)

    params = runner_state[0]

    return params, config


def test_RAD_convergence1():
    seed = 0
    payoff = jnp.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
    )

    rpg_alg = "doublesided_RAD"
    init_policies = {
        "carrot": {
            "carrot_0": jnp.array([0.5, 0.5]),
            "carrot_1": jnp.array([0.5, 0.5]),
        },
        "horse": {
            "horse_0": jnp.array([0.4, 0.6]),
            "horse_1": jnp.array([0.6, 0.4]),
        },
    }
    target_policies = {
        "carrot": {
            "carrot_0": jnp.array([0.0, 1.0]),
            "carrot_1": jnp.array([1.0, 0.0]),
        },
        "horse": {
            "horse_0": jnp.array([0.0, 1.0]),
            "horse_1": jnp.array([1.0, 0.0]),
        },
    }

    target_vals = {
        ("horse_0", "carrot_0"): jnp.array([1.0, 1.0]),
        ("carrot_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_1", "carrot_1"): jnp.array([1.0, 1.0]),
        ("carrot_1", "horse_1"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_1"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_0"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_1"): jnp.array([1.0, 1.0]),
    }

    params, config = run_alg(seed, payoff, rpg_alg, init_policies, target_policies, num_updates=80)

    # check that final horse policies are correct
    for h in target_policies["horse"]:
        if target_policies["horse"][h] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["horse"][h], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["horse"][h], atol=0.05)

    # check that final carrot policies are correct
    for c in target_policies["carrot"]:
        if target_policies["carrot"][c] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["carrot"][c], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["carrot"][c], atol=0.05)

    # check that final values from critic are correct
    for label, critic_params in params["critic"].items():
        val = extract_normal_value(config, critic_params)
        target_val = target_vals[label]
        assert jnp.allclose(val, target_val, atol=0.02)


def test_RAD_convergence2():
    seed = 0
    payoff = jnp.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
    )

    rpg_alg = "doublesided_RAD"
    init_policies = {
        "carrot": {
            "carrot_0": jnp.array([0.5, 0.5]),
            "carrot_1": jnp.array([0.5, 0.5]),
        },
        "horse": {
            "horse_0": jnp.array([0.6, 0.4]),
            "horse_1": jnp.array([0.4, 0.6]),
        },
    }
    target_policies = {
        "carrot": {
            "carrot_0": jnp.array([1.0, 0.0]),
            "carrot_1": jnp.array([0.0, 1.0]),
        },
        "horse": {
            "horse_0": jnp.array([1.0, 0.0]),
            "horse_1": jnp.array([0.0, 1.0]),
        },
    }

    target_vals = {
        ("horse_0", "carrot_0"): jnp.array([1.0, 1.0]),
        ("carrot_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_1", "carrot_1"): jnp.array([1.0, 1.0]),
        ("carrot_1", "horse_1"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_1"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_0"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_1"): jnp.array([1.0, 1.0]),
    }

    params, config = run_alg(seed, payoff, rpg_alg, init_policies, target_policies, num_updates=80)

    # check that final horse policies are correct
    for h in target_policies["horse"]:
        if target_policies["horse"][h] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["horse"][h], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["horse"][h], atol=0.05)

    # check that final carrot policies are correct
    for c in target_policies["carrot"]:
        if target_policies["carrot"][c] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["carrot"][c], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["carrot"][c], atol=0.05)

    # check that final values from critic are correct
    for label, critic_params in params["critic"].items():
        val = extract_normal_value(config, critic_params)
        target_val = target_vals[label]
        assert jnp.allclose(val, target_val, atol=0.02)


def test_RAP_convergence():
    seed = 0
    payoff = jnp.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
        ]
    )

    rpg_alg = "doublesided_RAD"
    init_policies = {
        "carrot": {
            "carrot_0": jnp.array([0.5, 0.5]),
            "carrot_1": jnp.array([0.5, 0.5]),
        },
        "horse": {
            "horse_0": jnp.array([0.6, 0.4]),
            "horse_1": jnp.array([0.4, 0.6]),
        },
    }
    target_policies = {
        "carrot": {
            "carrot_0": jnp.array([1.0, 0.0]),
            "carrot_1": jnp.array([0.0, 1.0]),
        },
        "horse": {
            "horse_0": jnp.array([1.0, 0.0]),
            "horse_1": jnp.array([0.0, 1.0]),
        },
    }

    target_vals = {
        ("horse_0", "carrot_0"): jnp.array([1.0, 1.0]),
        ("carrot_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_1", "carrot_1"): jnp.array([1.0, 1.0]),
        ("carrot_1", "horse_1"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_0"): jnp.array([1.0, 1.0]),
        ("horse_0", "horse_1"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_0"): jnp.array([0.0, 0.0]),
        ("horse_1", "horse_1"): jnp.array([1.0, 1.0]),
    }

    params, config = run_alg(seed, payoff, rpg_alg, init_policies, target_policies, num_updates=80)

    # check that final horse policies are correct
    for h in target_policies["horse"]:
        if target_policies["horse"][h] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["horse"][h], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["horse"][h], atol=0.05)

    # check that final carrot policies are correct
    for c in target_policies["carrot"]:
        if target_policies["carrot"][c] is not None:
            final_pi = extract_normal_policy(config, params["actor"]["carrot"][c], 0).probs.squeeze()
            assert jnp.allclose(final_pi, target_policies["carrot"][c], atol=0.05)

    # check that final values from critic are correct
    for label, critic_params in params["critic"].items():
        val = extract_normal_value(config, critic_params)
        target_val = target_vals[label]
        assert jnp.allclose(val, target_val, atol=0.02)


def test_RAD_loop(seed=55):

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_doublesided_RAD(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, _, _, _, _, _, _ = initialize_training(config, env, rpo_config, _rng)

    train = make_train(config, rpo_config)

    rng, train_rng = jax.random.split(rng)
    out = train(train_rng)

    params = out["runner_state"][0]

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(rng, config, params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(rng, config, params["actor"]["carrot"][c], 1).probs.squeeze())


def test_RAD_outer(seed=54):

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_doublesided_RAD(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, params, actor_network, critic_network, txs, opt_state, rollout_status = initialize_training(
        config, env, rpo_config, _rng
    )

    init_horse_probs = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    init_horse_params = probs2params(rng, init_horse_probs, config, 0)

    init_carrot_probs = jnp.array([[0.54, 0.46], [0.5, 0.5]])
    init_carrot_params = probs2params(rng, init_carrot_probs, config, 1)

    params["actor"]["horse"] = {k: tree_slice(init_horse_params, i) for i, k in enumerate(rpo_graph.horses)}
    params["actor"]["carrot"] = {k: tree_slice(init_carrot_params, i) for i, k in enumerate(rpo_graph.carrots)}

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(rng, config, params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(rng, config, params["actor"]["carrot"][c], 1).probs.squeeze())

    for _ in range(20):

        in_env_state, in_obsv, out_env_state, out_obsv = rollout_status
        out_carrot_tx, out_horse_tx, in_horse_tx, critic_tx = txs
        carrot_opt_state, horse_opt_state, critic_opt_state = opt_state

        # save opt_state for horse lookahead later
        past_in_opt_state = horse_opt_state

        in_lookahead = build_in_lookahead(config, rpo_graph, actor_network, critic_network, env, in_horse_tx)
        in_lookahead_update = build_in_lookahead_update(config, rpo_graph, actor_network, in_horse_tx)

        rng, in_rng, out_rng = jax.random.split(rng, 3)
        in_runner_state = (deepcopy(params), horse_opt_state, in_env_state, in_obsv, in_rng)

        lookahead_rollout_info, lookahead_params, in_value_update_info, in_env_state, in_obsv = in_lookahead(
            in_runner_state
        )

        # get rollouts for everything in rpo_graph.carrot_rollouts using new horse actor params
        # and perform update for carrot using rpo_graph.carrot_edges
        out_runner_state = (deepcopy(params), carrot_opt_state, out_env_state, out_obsv, out_rng)

        # build outer step using lookahead_traj_batch, past_lookahead_train_state
        out_step = build_out_step(
            config,
            rpo_graph,
            actor_network,
            critic_network,
            env,
            lookahead_rollout_info,
            lookahead_params,
            past_in_opt_state,
            in_lookahead_update,
            out_carrot_tx,
        )
        out_runner_state, out_value_update_info, carrot_metrics = out_step(out_runner_state)

        params, carrot_opt_state, out_env_state, out_obsv, rng = out_runner_state

        for h in rpo_graph.horses:
            print(h)
            print(extract_normal_policy(rng, config, params["actor"]["horse"][h], 0).probs.squeeze())

        for c in rpo_graph.carrots:
            print(c)
            print(extract_normal_policy(rng, config, params["actor"]["carrot"][c], 1).probs.squeeze())


def test_in_step(seed=42):

    rng = jax.random.PRNGKey(seed)

    payoff = jnp.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    config = OmegaConf.load("tests/configs/test_adv_normal.yaml")
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["payoffs"] = payoff
    # config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_doublesided_RAP(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, params, actor_network, critic_network, txs, opt_state, rollout_status = initialize_training(
        config, env, rpo_config, _rng
    )

    init_horse_probs = jnp.array([[0.5, 0.5], [0.1, 0.9]])
    init_horse_params = probs2params(rng, init_horse_probs, config, 0)

    init_carrot_probs = jnp.array([[0.01, 0.99]])
    init_carrot_params = probs2params(rng, init_carrot_probs, config, 1)

    params["actor"]["horse"] = {k: tree_slice(init_horse_params, i) for i, k in enumerate(rpo_graph.horses)}
    params["actor"]["carrot"] = {k: tree_slice(init_carrot_params, i) for i, k in enumerate(rpo_graph.carrots)}

    in_env_state, in_obsv, out_env_state, out_obsv = rollout_status
    out_carrot_tx, out_horse_tx, in_horse_tx, critic_tx = txs
    carrot_opt_state, horse_opt_state, critic_opt_state = opt_state

    # get rollouts for everything in rpo_graph.horse_rollouts
    all_actor_params = params["actor"]["horse"].copy()
    all_actor_params.update(params["actor"]["carrot"])
    actor_params = tree_stack(
        [tree_stack([all_actor_params[a0], all_actor_params[a1]]) for (a0, a1) in rpo_graph.horse_rollouts]
    )
    critic_params = tree_stack([params["critic"][pair] for pair in rpo_graph.horse_rollouts])
    rng, _rng = jax.random.split(rng)
    rollout_rng = jax.random.split(_rng, len(rpo_graph.horse_rollouts))
    _get_rollouts = build_get_rollouts(config, env, actor_network, critic_network)
    traj_batch, advantages, targets, in_env_state, in_obsv = jax.vmap(_get_rollouts, in_axes=(0, 0, 0, 0, 0))(
        actor_params, critic_params, in_env_state, in_obsv, rollout_rng
    )

    # save this rng value to be used later
    update_rng = rng

    # and perform lookahead for horse using rpo_graph.horse_edges
    update_state = (params, horse_opt_state, traj_batch, advantages, update_rng)
    _horse_update = build_horse_update(config, actor_network, rpo_graph, in_horse_tx)
    lookahead_params, _ = _horse_update(update_state)

    for h in rpo_graph.horses:
        print(h)
        print(extract_normal_policy(rng, config, lookahead_params["actor"]["horse"][h], 0).probs.squeeze())

    for c in rpo_graph.carrots:
        print(c)
        print(extract_normal_policy(rng, config, lookahead_params["actor"]["carrot"][c], 1).probs.squeeze())


def test_make_data_trees(seed=42):

    # payoff = jnp.array([
    #     [1,0],
    #     [0,1],
    # ])

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load("tests/configs/test_storm_step.yaml")
    config = OmegaConf.to_container(config)
    # config["ENV_KWARGS"]["payoffs"] = payoff
    config["ENV_KWARGS"]["payoff_matrix"] = jnp.array(config["PAYOFF_MATRIX"])

    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    rpo_config = build_onesided_RAP(config)

    rng, _rng = jax.random.split(rng)
    rpo_graph, params, actor_network, critic_network, txs, opt_state, rollout_status = initialize_training(
        config, env, rpo_config, _rng
    )

    in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv = rollout_status

    # get rollouts for everything in rpo_graph.horse_rollouts
    all_actor_params = params["actor"]["horse"].copy()
    all_actor_params.update(params["actor"]["carrot"])
    actor_params = tree_stack(
        [tree_stack([all_actor_params[a0], all_actor_params[a1]]) for (a0, a1) in rpo_graph.horse_rollouts]
    )
    critic_params = tree_stack([params["critic"][pair] for pair in rpo_graph.horse_rollouts])
    rng, _rng = jax.random.split(rng)
    rollout_rng = jax.random.split(_rng, len(rpo_graph.horse_rollouts))
    _get_rollouts = build_get_rollouts(config, env, actor_network, critic_network)
    traj_batch, advantages, targets, in_env_state, in_obsv = jax.vmap(_get_rollouts, in_axes=(0, 0, 0, 0, 0))(
        actor_params, critic_params, in_env_state, in_obsv, rollout_rng
    )

    traj_batch, advantages, shaping_carrot_params, mask = rpo_graph.make_data_tree(
        (traj_batch, advantages), params["actor"]["carrot"]
    )

    # dim is (rpo_graph.horses, edge, horse and carrot, num_steps, num_envs)
    print(rpo_graph.horses)
    print(traj_batch.reward.shape)
    print(advantages.shape)
    print(shaping_carrot_params.keys())
    print(mask)
    # for k,v in rollout_map.items():
    #     print(k)
    #     print(v.keys())


def test_gae_linearity():
    rng = jax.random.PRNGKey(42)
    config = {
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
    }
    get_gae = build_calculate_gae(config)

    rng, b_rng, r_rng, v_rng, l_rng = jax.random.split(rng, 5)
    traj_batch = Transition(
        done=jax.random.binomial(b_rng, 1, 0.1, (20, 8)),
        reward=jax.random.normal(r_rng, shape=(20, 8)),
        value=jax.random.normal(v_rng, shape=(20, 8)),
        action=None,
        log_prob=None,
        obs=None,
        state_features=None,
        info=None,
    )
    scalar = 0.1
    scaled_traj_batch = Transition(
        done=jax.random.binomial(b_rng, 1, 0.1, (20, 8)),
        reward=scalar * jax.random.normal(r_rng, shape=(20, 8)),
        value=scalar * jax.random.normal(v_rng, shape=(20, 8)),
        action=None,
        log_prob=None,
        obs=None,
        state_features=None,
        info=None,
    )

    last_val = jax.random.normal(l_rng, (8,))

    gae, _ = get_gae(traj_batch, last_val)

    scaled_gae, _ = get_gae(scaled_traj_batch, scalar * last_val)

    jnp.allclose(scaled_traj_batch, scalar * last_val, atol=1e-6)


# test entropy regularization


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update('jax_disable_jit', True)
    # jnp.set_printoptions(suppress=True)
    # wandb.init(mode='disabled')
    start = time.time()

    test_out_step()

    print("total time", time.time() - start)
