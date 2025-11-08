import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

# from baselines.IPPO.mer_ff import ActorCritic, Transition, make_train
# from data.testing import TESTING_DATA_DIR
# from jaxmarl.environments.multi_agent_env import OverridePlayer
from omegaconf import OmegaConf


def probs2params(key, probs, config, agent_idx):

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n
    network = ActorCritic(action_space_size, activation=config["ACTIVATION"])
    key, key_a = jax.random.split(key, 2)

    init_x = jnp.zeros(env.observation_space(env.agents[agent_idx]).shape)
    init_x = init_x.flatten()

    init_params = network.init(key_a, init_x)
    # print(init_params)
    params = jax.tree.map(lambda x: jnp.zeros(probs.shape[0:1] + x.shape), init_params)
    params["params"]["Dense_2"]["bias"] = jnp.log(probs)

    # dummy_obs = np.array([[1],[1],[1],[1]])
    # pis, _ = jax.vmap(network.apply, in_axes=(0,0))(params, dummy_obs)

    return params


def run_fixed_coparam_setup(rng, payoffs, probs, override_config={}, num_trials=10):
    config = OmegaConf.load(TESTING_DATA_DIR + "test_mer_ff_normal.yaml")
    config = OmegaConf.to_container(config)

    config["ENV_KWARGS"]["payoffs"] = payoffs

    rng, _rng = jax.random.split(rng, 2)
    coparams = probs2params(_rng, probs, config, config["COPLAYER"])

    config["COPARAMS_SOURCE"] = "pytree"
    config["COPARAMS_BATCH"] = coparams

    config.update(override_config)

    return run_test_core(rng, config, num_trials)


def run_test_core(rng, config, num_trials=10):
    rng, _rng = jax.random.split(rng, 2)
    rngs = jax.random.split(_rng, num_trials)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)

    train_state = out["runner_state"][0]
    metrics = out["metrics"]

    ### Evaluate
    extract_rng = jax.random.split(rng, num_trials)
    return (
        jax.vmap(extract_normal_policy, in_axes=(0, None, 0, None))(extract_rng, config, train_state.params, 1),
        metrics,
    )


def extract_normal_policy(key, config, params, agent_idx):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    action_space_size = env.action_space(env.agents[agent_idx]).n
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
