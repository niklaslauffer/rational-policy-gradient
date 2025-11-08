"""
Testing for normal form games. The particles play columns and the copolicies play rows.
"""

import itertools
import os

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

import jaxmarl
from adv_rational_param_share import *
from utils import *

from flax.training.train_state import TrainState
import flax.linen as nn
from typing import Sequence, NamedTuple, Any
from flax.linen.initializers import constant, orthogonal

def tree_slice(tree, idx):
    return jax.tree.map(lambda x: x[idx], tree)

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def probs2params(key, probs, config, agent_idx):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n

    network = Actor(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    init_params = network.init(key_a, init_x)

    params = jax.tree.map(lambda x :jnp.zeros(probs.shape[0:1] + x.shape), init_params)

    params['params']['Dense_2']['bias'] = jnp.log(probs)

    return params

def extract_normal_policy(key, config, params, agent_idx, obs=None):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n

    network = Actor(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)

    key, _rng = jax.random.split(key)
    # reset_rng = jax.random.split(_rng, batch_size)
    if obs is None:
        obs, _ = env.reset(_rng)
    pis = jax.vmap(network.apply, in_axes=(0,None))(params, [obs[env.agents[agent_idx]]])

    return pis

def setup_test(payoffs, config_override=dict(), seed=42):

    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load('tests/configs/test_adv_normal.yaml')
    config = OmegaConf.to_container(config)     

    config["ENV_KWARGS"]["payoffs"] = payoffs

    # override config
    config.update(config_override)

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    env = make_world_state_wrapper(config['ENV_NAME'], env)
    
    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env, replace_info=True)
    else:
        env = LogWrapper(env, replace_info=True)

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] 
    )
    config["MINIBATCH_SIZE"] = {}
    config["MINIBATCH_SIZE"]['CARROT'] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]['CARROT']
    )
    config["MINIBATCH_SIZE"]['HORSE'] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]['HORSE']
    )

    # INIT ACTOR NETWORK
    actor_network = Actor(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
    
    init_x = init_x.flatten()

    rng, _rng = jax.random.split(rng)
    params_rng = jax.random.split(_rng, 3)

    actor_network_params_flat = jax.vmap(actor_network.init, in_axes=(0,None))(params_rng, init_x)
    actor_network_params = {
        "horse": tree_slice(actor_network_params_flat, jnp.s_[:2]),
        "carrot": tree_slice(actor_network_params_flat, jnp.s_[2:])
    }

    # INIT CRITIC NETWORK
    critic_network = Critic(num_agents=2, activation=config["ACTIVATION"])
    init_x = jnp.zeros(env.state_feature_size())

    init_x = init_x.flatten()

    rng, _rng = jax.random.split(rng)
    params_rng = jax.random.split(_rng, 2)

    critic_network_params_flat = jax.vmap(critic_network.init, in_axes=(0,None))(params_rng, init_x)
    critic_network_params = {"horse": tree_slice(critic_network_params_flat, 0), "carrot": tree_slice(critic_network_params_flat, 1)}

    network_params = {"actor": actor_network_params, "critic": critic_network_params}


    carrot_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]['CARROT']), 
        optax.adam(config['LR']["CARROT"])
    )
    # out_horse_tx = optax.adam(config['LR']["OUT_HORSE"], eps=1e-5)
    # in_horse_tx = optax.adam(config['LR']["IN_HORSE"], eps=1e-5)   
    out_horse_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]['CARROT']), 
        optax.sgd(config['LR']["OUT_HORSE"])
    )  
    in_horse_tx = optax.sgd(config['LR']["IN_HORSE"])

    txs = carrot_tx, out_horse_tx, in_horse_tx

    carrot_opt_state = carrot_tx.init((actor_network_params['carrot'], critic_network_params['carrot']))
    horse_opt_state = out_horse_tx.init((actor_network_params['horse'], critic_network_params['horse']))

    opt_state = (carrot_opt_state, horse_opt_state)

    # vect_reset = jnp.vectorize(env.reset, signature='(2)->(),()')
    # vmap reset over three dimensions, would like to just use vectorize but it doesn't work on dictionary outputs 
    vect_reset = jax.vmap(env.reset)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    in_obsv, in_env_state = vect_reset(reset_rng)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    out_obsv, out_env_state = vect_reset(reset_rng)
    reg_env_state, reg_obsv = deepcopy(out_env_state), deepcopy(out_obsv)

    rng, _rng = jax.random.split(rng)
    
    return config, network_params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng, actor_network, critic_network, env, txs



def test_inner_step():

    config_override = {}

    payoffs = jnp.array([
        [1,0],
        [0,1],
    ])

    seed = 33

    config, params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng, actor_network, critic_network, env, txs = setup_test(payoffs, config_override=config_override, seed=seed)

    bias_act = jnp.array([[0.5, 0.5],
                          [0.01, 0.99]])


    init_horse_params = probs2params(rng, bias_act, config, 0)

    params['actor']['horse'] = init_horse_params

    out_carrot_tx, out_horse_tx, in_horse_tx = txs
    carrot_opt_state, horse_opt_state = opt_state

    config['N_LOOKAHEAD'] = 10

    in_lookahead = build_adv_in_lookahead(config, actor_network, critic_network, env, in_horse_tx)
    in_lookahead_update = build_adv_in_lookahead_update(config, actor_network, critic_network, in_horse_tx, n_lookahead=config['N_LOOKAHEAD'])
    in_step = build_adv_in_lookahead_update(config, actor_network, critic_network, out_horse_tx, n_lookahead=config['N_LOOKAHEAD'], use_value_loss=True)

    rng, in_rng = jax.random.split(rng)
    in_runner_state = (params.copy(), horse_opt_state, in_env_state, in_obsv, reg_env_state, reg_obsv, in_rng)

    lookahead_rollout_info, lookahead_params, (in_env_state, in_obsv, reg_env_state, reg_obsv) = in_lookahead(in_runner_state)

    # first_lookahead_traj_batch = jax.tree.map(lambda x : x[:1,], lookahead_rollout_info)
    # lookahead_params, horse_opt_state, horse_update = in_step(params.copy(), horse_opt_state, first_lookahead_traj_batch)

    ego_policies = extract_normal_policy(rng, config, lookahead_params['actor']['horse'], 0).probs.squeeze()
    print('adversary horse')
    print(ego_policies[0])
    print('victim horse')
    print(ego_policies[1])

    print('adversary carrot')
    print(extract_normal_policy(rng, config, lookahead_params['actor']['carrot'], 1).probs.squeeze())

def test_step():

    config_override = {}

    payoffs = jnp.array([
        [1,0],
        [0,1],
    ])

    seed = 33

    config, params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng, actor_network, critic_network, env, txs = setup_test(payoffs, config_override=config_override, seed=seed)

    bias_act = jnp.array([[0.6, 0.4],
                          [0.1, 0.9]])
    
    mixed_act = jnp.array([[0.5, 0.5]])


    init_horse_params = probs2params(rng, bias_act, config, 0)
    init_carrot_params = probs2params(rng, mixed_act, config, 0)

    params['actor']['horse'] = init_horse_params
    params['actor']['carrot'] = init_carrot_params

    full_runner_state = (params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng)

    _update_step = build_adv_step(config, actor_network, critic_network, env, txs)

    runner_state, metric = jax.lax.scan(
        _update_step, full_runner_state, None, config["NUM_UPDATES"]
    )

    after_params = runner_state[0]

    ego_policies = extract_normal_policy(rng, config, after_params['actor']['horse'], 0).probs.squeeze()
    print('adversary horse')
    print(ego_policies[0])
    print('victim horse')
    print(ego_policies[1])

    print('adversary carrot')
    print(extract_normal_policy(rng, config, after_params['actor']['carrot'], 1).probs.squeeze())

def test_fixed_victim_step():

    config_override = {'FIXED_VICTIM': True}

    payoffs = jnp.array([
        [1,0],
        [0,1],
    ])

    seed = 33

    config, params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng, actor_network, critic_network, env, txs = setup_test(payoffs, config_override=config_override, seed=seed)

    bias_act = jnp.array([[0.6, 0.4],
                          [0.1, 0.9]])
    
    mixed_act = jnp.array([[0.5, 0.5]])


    init_horse_params = probs2params(rng, bias_act, config, 0)
    init_carrot_params = probs2params(rng, mixed_act, config, 0)

    params['actor']['horse'] = init_horse_params
    params['actor']['carrot'] = init_carrot_params

    full_runner_state = (params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, reg_env_state, reg_obsv, rng)

    _update_step = build_adv_step(config, actor_network, critic_network, env, txs)

    runner_state, metric = jax.lax.scan(
        _update_step, full_runner_state, None, config["NUM_UPDATES"]
    )

    after_params = runner_state[0]

    ego_policies = extract_normal_policy(rng, config, after_params['actor']['horse'], 0).probs.squeeze()
    print('adversary horse')
    print(ego_policies[0])
    print('victim horse')
    print(ego_policies[1])

    print('adversary carrot')
    print(extract_normal_policy(rng, config, after_params['actor']['carrot'], 1).probs.squeeze())


def test_adv_normal():

    seed = 57
    
    rng = jax.random.PRNGKey(seed)

    config = OmegaConf.load('tests/configs/test_adv_normal.yaml')
    config = OmegaConf.to_container(config) 

    payoffs = jnp.array([
        [1,0],
        [0,1],
    ])

    config["ENV_KWARGS"]["payoffs"] = payoffs

    train = jax.jit(make_train(config))

    out = train(rng)

    params = out['runner_state'][0]
    
    ego_policies = extract_normal_policy(rng, config, params['actor']['horse'], 0).probs.squeeze()
    print('adversary horse')
    print(ego_policies[0])
    print('victim horse')
    print(ego_policies[1])

    print('adversary carrot')
    print(extract_normal_policy(rng, config, params['actor']['carrot'], 1).probs.squeeze())



if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update('jax_disable_jit', True)
    # jnp.set_printoptions(suppress=True)
    wandb.init(mode='disabled')
    start = time.time()
    # with jax.profiler.trace("traces/normal"):
    test_fixed_victim_step()
    print('total time', time.time()-start)