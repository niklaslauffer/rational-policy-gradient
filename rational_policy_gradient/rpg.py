import os
import pickle
import time
from copy import deepcopy
from typing import cast

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jaxmarl.environments.overcooked import (
    overcooked_layouts as base_overcooked_layouts,
)
from omegaconf import OmegaConf

from rational_policy_optimization.extra_environments import (
    overcooked_layouts as extra_overcooked_layouts,
)
from rational_policy_optimization.root import CONFIG_DIR, MODEL_DIR
from rational_policy_optimization.rpg_algs import make_rpg_alg
from rational_policy_optimization.utils import Actor as NonHanabiActor
from rational_policy_optimization.utils import Critic as NonHanabiCritic
from rational_policy_optimization.utils import (
    HanabiActor,
    HanabiCritic,
    Transition,
    build_wandb_checkpoint_rpg_callback,
    build_wandb_log_rpg_callback,
    create_rpg_graph,
    dice_ratio,
    explained_variance,
    tree_slice,
    tree_stack,
    wandb_visualize,
    wrap_env,
)
from rational_policy_optimization.wrappers import registration_wrapper

overcooked_layouts = {**base_overcooked_layouts, **extra_overcooked_layouts}


def build_env_step(config, env, actor_network, critic_network, rew_shaping_schedule=lambda x: 0.0):
    """Builds a function that steps the environment once across a batch dimension.

    Args:
        config: Configuration dictionary.
        env: The environment to step.
        actor_network: The actor network.
        critic_network: The critic network.
        rew_shaping_schedule: A function that takes in the update step and returns a scalar for reward shaping.
    Returns:
        A function that steps the environment once with signature:
            (step_state, unused) -> (new_step_state, transition)
        where step_state is a tuple of (params, env state, obs, update_step, rng)"""

    def _env_step(step_state, unused):
        params, env_state, last_obs, update_step, rng = step_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)

        # vectorize obs and flatten observation dimension
        obs_batch = jnp.stack([last_obs[a].reshape((config["NUM_ENVS"], -1)) for a in env.agents])
        env_state_features = jax.vmap(env.get_state_features, in_axes=(0, 0))(env_state, last_obs)

        # stacked_params has shape (num_agents, -1)
        # obs_batch has shape (num_agents, num_envs, -1)

        # vmap across agents and then particles
        pi = jax.vmap(actor_network.apply, in_axes=(0, 0))(params["actor"], obs_batch)
        action = pi.sample(seed=_rng)
        value = critic_network.apply(params["critic"], env_state_features)

        # dictionize actions  and flatten observation dimension
        env_act = {agent: action[i] for i, agent in enumerate(env.agents)}

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        # vmap over particles and envs
        obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)

        # shape reward
        if rew_shaping_schedule:
            reward = jax.tree.map(lambda x, y: x + y * rew_shaping_schedule(update_step), reward, info["shaped_reward"])
            stacked_shaped_reward = jnp.stack([info["shaped_reward"][a] for a in env.agents])
            stacked_shaped_reward = jnp.moveaxis(stacked_shaped_reward, 0, 1)
            info["shaped_reward"] = stacked_shaped_reward * rew_shaping_schedule(update_step)

        info = jax.tree.map(lambda x: x.reshape((config["NUM_ENVS"], env.num_agents)), info)

        # bring the agent dimension to the front
        info = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), info)
        value = jnp.moveaxis(value, 1, 0)

        # duplicate along the agent (first) dimension
        env_state_features_batch = jnp.expand_dims(env_state_features, 0).repeat(env.num_agents, axis=0)

        log_prob = pi.log_prob(action)

        # stack dict to create agent dimension in front
        done = jnp.stack([done[a] for a in env.agents])
        reward = jnp.stack([reward[a] for a in env.agents])

        transition = Transition(
            done,
            action,
            value,
            reward,
            log_prob,
            obs_batch,
            env_state_features_batch,
            info,
        )
        step_state = (params, env_state, obsv, update_step, rng)
        return step_state, transition

    return _env_step


def build_calculate_gae(config):
    """Builds a function to calculate Generalized Advantage Estimation (GAE).

    Args:
        config: Configuration dictionary containing GAMMA (discount factor) and
                GAE_LAMBDA (GAE parameter for bias-variance tradeoff).

    Returns:
        A function that calculates advantages and targets with signature:
            (traj_batch, last_val) -> (advantages, targets)
        where advantages are GAE estimates and targets are value targets.
    """

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    return _calculate_gae


def build_get_rollouts(config, env, actor_network, critic_network, rew_shaping_schedule=None):
    """Builds a function to collect environment rollouts using input actor and critic params,
     performs rollouts and computes advantages.

    Args:
        config: Configuration dictionary containing NUM_STEPS and other parameters.
        env: The environment to collect rollouts from.
        actor_network: The actor network for action selection.
        critic_network: The critic network for value estimation.
        rew_shaping_schedule: Optional function for reward shaping based on update step.

    Returns:
        A function that collects rollouts and advantages with signature:
            (actor_params, critic_params, env_state, obs, update_step, rng) ->
            (traj_batch, advantages, targets, final_env_state, final_obs)
    """
    _env_step = build_env_step(
        config,
        env,
        actor_network,
        critic_network,
        rew_shaping_schedule=rew_shaping_schedule,
    )
    _calculate_gae = build_calculate_gae(config)

    def _get_rollouts(actor_params, critic_params, env_state, obsv, update_step, rng):
        # obsv has shape {agent_name : (num_envs, obs_dim)}
        # env_state has shape (num_envs, obs_dim)

        params = {"actor": actor_params, "critic": critic_params}

        runner_state = (params, env_state, obsv, update_step, rng)
        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
        # update runner_state
        _, env_state, last_obs, _, rng = runner_state

        # traj_batch has shape (env_rollouts, player, envs, -1)
        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        last_state_features = env.get_state_features(env_state, last_obs)

        last_val = critic_network.apply(params["critic"], last_state_features)
        last_val = jnp.moveaxis(last_val, 1, 0)

        # vmap over both players
        advantages, targets = jax.vmap(_calculate_gae, in_axes=(0, 0))(traj_batch, last_val)

        return traj_batch, advantages, targets, env_state, last_obs

    return _get_rollouts


# factor the API (loss, graph) out of here and next function
def build_base_update(config, actor_network, rpo_graph, tx):
    """Builds a function to update base agent parameters against manipulators.

    Args:
        config: Configuration dictionary.
        actor_network: The actor network for policy evaluation.
        rpo_graph: RPG graph specifying the adversarial optimization objective.
        tx: optimizer for parameter updates.

    Returns:
        A function that performs one epoch of base agent updates with signature:
            (params, opt_state, traj_batch, advantages, update_rng) -> (updated_params, updated_opt_state).
    """

    def _update_epoch(update_state):
        def _loss_vs_manipulator_fn(base_actor_params, manipulator_shaping_params, traj_batch, gae):
            base_gae = gae[0]

            # rollouts have shape (manipulator or base, env, rollouts)
            # RERUN NETWORK
            pi = jax.vmap(actor_network.apply, in_axes=(None, 0))(base_actor_params, traj_batch.obs[0])
            log_prob = pi.log_prob(traj_batch.action[0])

            manipulator_pi = jax.vmap(actor_network.apply, in_axes=(None, 0))(
                manipulator_shaping_params, traj_batch.obs[1]
            )
            manipulator_log_prob = manipulator_pi.log_prob(traj_batch.action[1])

            done = traj_batch.done[0]  # adjust indexing if needed
            starts = jnp.roll(done, 1, axis=1)

            ratio = dice_ratio(log_prob, manipulator_log_prob, starts, lam_past=config["DICE_LAMBDA"])

            loss_actor1 = ratio * base_gae

            loss_actor = -loss_actor1
            loss_actor = loss_actor.mean()
            # loss_actor = loss_actor.mean()

            entropy = config["ENT_COEF"]["BASE"] * pi.entropy().mean()

            total_loss = loss_actor - entropy
            return total_loss, (loss_actor, entropy)

        params, opt_state, traj_batch_pool, advantages_pool, _ = update_state

        traj_batch, advantages, shaping_manipulator_params, pad_mask = rpo_graph.make_data_tree(
            (traj_batch_pool, advantages_pool), params["actor"]["manipulator"]
        )
        # dim is now (rpo_graph.bases, edge, base and manipulator, num_steps, num_envs)

        def v_value_and_grads(base_actor_params, traj_batch, gae, shaping_manipulator_params, loss_mask):
            # swap env and rollout dim
            traj_batch, gae = jax.tree.map(lambda x: jnp.moveaxis(x, 2, 3), (traj_batch, gae))

            def loss_fn(base_actor_params):
                loss_, aux = jax.vmap(_loss_vs_manipulator_fn, in_axes=(None, 0, 0, 0))(
                    base_actor_params, shaping_manipulator_params, traj_batch, gae
                )

                return jnp.mean(loss_ * loss_mask)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

            return grad_fn(base_actor_params)

        param_actor_base_v = tree_stack(params["actor"]["base"][h] for h in rpo_graph.optimizing_bases)

        values, grads = jax.vmap(
            v_value_and_grads,
            in_axes=(0, 0, 0, 0, 0),
        )(param_actor_base_v, traj_batch, advantages, shaping_manipulator_params, pad_mask)

        grads = {h: tree_slice(grads, i) for i, h in enumerate(rpo_graph.optimizing_bases)}
        for h in rpo_graph.bases:
            if h not in rpo_graph.optimizing_bases:
                grads[h] = jax.tree.map(jnp.zeros_like, (params["actor"]["base"][h]))

        updates, opt_state = tx.update(grads, opt_state, params["actor"]["base"])

        params["actor"]["base"] = optax.apply_updates(params["actor"]["base"], updates)

        return params, opt_state

    return _update_epoch


def build_in_lookahead(
    config,
    rpo_graph,
    actor_network,
    critic_network,
    env,
    tx,
    n_lookahead=1,
    rew_shaping_schedule=None,
):
    """Builds a function that performs inner updates for base agents.

    This function implements the inner lookahead mechanism in RPG that manipulators will use
    to take higher-order gradients.

    Args:
        config: Configuration dictionary.
        rpo_graph: RPG graph specifying the adversarial optimization objective.
        actor_network: The actor network for policy evaluation.
        critic_network: The critic network for value estimation.
        env: The environment for rollout collection.
        tx: Optimizer for base agent updates.
        n_lookahead: Number of lookahead steps to perform (default: 1).
        rew_shaping_schedule: Optional reward shaping schedule function.

    Returns:
        A function that collects rollouts and performs inner gradient step with signature:
            (params, opt_state, env_state, obsv, update_step, rng) ->
            (updated_params, rollout_info, value_info, metrics, env_state, obs)
        The rollouts are returns as rollout_info for later use in manipulator updates.
    """
    _base_update_epoch = build_base_update(config, actor_network, rpo_graph, tx)
    _get_rollouts = build_get_rollouts(
        config,
        env,
        actor_network,
        critic_network,
        rew_shaping_schedule=rew_shaping_schedule,
    )

    def in_lookahead(runner_state):
        def _step_ahead(runner_state, unused):
            # runner_state : (params, opt_state, env_state, obsv, rng)

            params, opt_state, env_state, obsv, update_step, rng = runner_state

            # get rollouts for everything in rpo_graph.base_rollouts
            all_actor_params = params["actor"]["base"].copy()
            all_actor_params.update(params["actor"]["manipulator"])

            actor_params = tree_stack(
                [tree_stack([all_actor_params[a0], all_actor_params[a1]]) for (a0, a1) in rpo_graph.base_rollouts]
            )
            critic_params = tree_stack([params["critic"][pair] for pair in rpo_graph.base_rollouts])

            rng, _rng = jax.random.split(rng)
            rollout_rng = jax.random.split(_rng, len(rpo_graph.base_rollouts))
            traj_batch, advantages, targets, env_state, last_obs = jax.vmap(
                _get_rollouts, in_axes=(0, 0, 0, 0, None, 0)
            )(actor_params, critic_params, env_state, obsv, update_step, rollout_rng)

            # save this rng value to be used later
            update_rng = rng

            # and perform lookahead for base using rpo_graph.base_edges
            update_state = (params, opt_state, traj_batch, advantages, update_rng)
            new_params, opt_state = _base_update_epoch(update_state)

            runner_state = (
                new_params,
                opt_state,
                env_state,
                last_obs,
                update_step + 1,
                rng,
            )

            rollout_info = (traj_batch, advantages, update_rng)
            value_update_info = (traj_batch, targets)

            # traj_batch has shape (player, rollouts, envs, -1)
            # mean_info = jax.tree.map(lambda x: jnp.mean(x, axis=(1, 2, 3)), traj_batch.info)
            mean_info = jax.tree.map(
                lambda x: jnp.mean(x, axis=(1, 2)), tree_slice(traj_batch.info, jnp.s_[:, 0, :, :])
            )
            exp_var = explained_variance(traj_batch.value, targets, axis=(1, 2, 3))

            metrics = {"info": mean_info, "explained_variance": exp_var}

            return runner_state, (
                rollout_info,
                metrics,
                value_update_info,
                env_state,
                last_obs,
            )

        (
            new_runner_state,
            (rollout_info, metrics, value_update_info, env_state, last_obs),
        ) = jax.lax.scan(_step_ahead, runner_state, None, n_lookahead)

        new_params = new_runner_state[0]

        return (
            new_params,
            rollout_info,
            tree_slice(value_update_info, 0),
            tree_slice(metrics, 0),
            tree_slice(env_state, 0),
            tree_slice(last_obs, 0),
        )

    return in_lookahead


def build_in_lookahead_update(config, rpo_graph, actor_network, tx, n_lookahead=1):
    """Builds a function to apply inner updates to base agent parameters.

    Similar to build_in_lookahead but takes pre-computed inner trajectories and
    applies the corresponding parameter updates to base agents.

    Args:
        config: Configuration dictionary.
        rpo_graph: RPG graph specifying the adversarial optimization objective.
        actor_network: The actor network for policy evaluation.
        tx: Optimizer for parameter updates.
        n_lookahead: Number of lookahead steps to apply (default: 1).

    Returns:
        A function that applies lookahead updates with signature:
            (params, opt_state, lookahead_trajectories) -> (updated_params, updated_opt_state, metrics)
        where lookahead_trajectories are pre-computed trajectories for each lookahead step.
    """
    _base_update_epoch = build_base_update(config, actor_network, rpo_graph, tx)

    def in_lookahead_update(params, opt_state, lookahead_traj_batches):
        def _step_lookahead(runner_state, lookahead_traj_batch):
            params, opt_state = runner_state
            traj_batch, advantages, update_rng = lookahead_traj_batch

            update_state = (params, opt_state, traj_batch, advantages, update_rng)
            params, opt_state = _base_update_epoch(update_state)
            metrics = {"update": 0}

            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(
            _step_lookahead,
            (params, opt_state),
            lookahead_traj_batches,
            length=n_lookahead,
        )

        return params, opt_state, metrics

    return in_lookahead_update


def build_manipulator_update(
    config,
    rpo_graph,
    actor_network,
    lookahead_traj_batches,
    past_in_opt_state,
    _in_lookahead_update,
    tx,
):
    """Builds a function to update manipulator agent parameters.

    This function implements the manipulator update step in RPG, where manipulators
    are trained to shape base agent learning through the adversarial objective.

    Args:
        config: Configuration dictionary.
        rpo_graph: RPG graph specifying the adversarial optimization objective.
        actor_network: The actor network for policy evaluation.
        lookahead_traj_batches: Pre-computed trajectories from lookahead rollouts.
        past_in_opt_state: Previous optimizer state for base agents.
        _in_lookahead_update: Function to apply lookahead updates.
        tx: Optimizer for manipulator parameter updates.

    Returns:
        A function that performs manipulator updates with signature:
            (params, opt_state, traj_batch_pool, advantages_pool, rng) ->
            (updated_params, updated_opt_state, update_info)
    """

    def _update_epoch(update_state):
        def _entropy_loss(params, traj_batch):
            pi = actor_network.apply(params, traj_batch.obs)
            entropy = config["ENT_COEF"]["MANIPULATOR"] * pi.entropy().mean()
            return entropy

        def _loss(params, traj_batch, gae):
            # RERUN NETWORK
            pi = actor_network.apply(params, traj_batch.obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)

            loss_actor = -ratio * gae
            loss_actor = loss_actor.mean()

            return loss_actor

        params, opt_state, traj_batch_pool, advantages_pool, _ = update_state

        def full_loss_fn(params, traj_batch_pool, advantages_pool):
            lookahead_params, _, _ = _in_lookahead_update(
                deepcopy(params), deepcopy(past_in_opt_state), lookahead_traj_batches
            )

            traj_batch, advantages, acting_base_params = rpo_graph.make_manipulator_data_tree(
                (traj_batch_pool, advantages_pool),
                deepcopy(lookahead_params["actor"]["base"]),
            )

            # traj_batch and advantages has shape (rollout pairs, env steps, num envs)
            traj_batch, advantages = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (
                        len(rpo_graph.manipulator_edges),
                        config["NUM_STEPS"],
                        config["NUM_ENVS"],
                    )
                    + x.shape[3:],
                ),
                (traj_batch, advantages),
            )

            lookahead_loss = jax.vmap(_loss, in_axes=(0, 0, 0))(acting_base_params, traj_batch, advantages)

            manipulator_traj_batch_pool, _, _ = tree_slice(lookahead_traj_batches, 0)
            entropy_traj_batch, entropy_pad = rpo_graph.make_manipulator_rollouts_data_tree(manipulator_traj_batch_pool)

            entropy_traj_batch = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (
                        len(rpo_graph.manipulators),
                        x.shape[1],
                        config["NUM_STEPS"] * config["NUM_ENVS"],
                    )
                    + x.shape[4:],
                ),
                entropy_traj_batch,
            )

            manipulator_params_array = tree_stack(params["actor"]["manipulator"][c] for c in rpo_graph.manipulators)

            if len(entropy_traj_batch) > 0:
                entropy_loss = (
                    jax.vmap(
                        jax.vmap(
                            _entropy_loss,
                            in_axes=(None, 0),
                        ),
                        in_axes=(0, 0),
                    )(manipulator_params_array, entropy_traj_batch)
                    * entropy_pad
                )
            else:
                entropy_loss = 0.0

            full_loss = jnp.mean(lookahead_loss) + jnp.mean(entropy_loss) * config["ENT_COEF"]["MANIPULATOR"]

            return full_loss

        value, grads = jax.value_and_grad(full_loss_fn, has_aux=False)(
            deepcopy(params), traj_batch_pool, advantages_pool
        )

        updates, opt_state = tx.update(grads["actor"]["manipulator"], opt_state, params["actor"]["manipulator"])

        update_info = {}

        params["actor"]["manipulator"] = optax.apply_updates(params["actor"]["manipulator"], updates)

        return params, opt_state, update_info

    return _update_epoch


def build_out_step(
    config,
    rpo_graph,
    actor_network,
    critic_network,
    env,
    lookahead_traj_batch,
    lookahead_params,
    past_in_opt_state,
    in_lookahead_update,
    manipulator_tx,
    rew_shaping_schedule=None,
):
    """Builds a function that performs the outer step of RPG algorithm.

    The outer step collects rollouts using updated base agent parameters from lookahead
    and trains manipulator agents based on these rollouts.

    Args:
        config: Configuration dictionary with environment and training parameters.
        rpo_graph: RPG graph specifying the adversarial optimization objective.
        actor_network: The actor network for policy evaluation.
        critic_network: The critic network for value estimation.
        env: The environment for rollout collection.
        lookahead_traj_batch: Trajectory batches from lookahead rollouts.
        lookahead_params: Updated parameters from lookahead step.
        past_in_opt_state: Previous optimizer state for inner updates.
        in_lookahead_update: Function to apply lookahead updates.
        manipulator_tx: Optimizer for manipulator updates.
        rew_shaping_schedule: Optional reward shaping schedule function.

    Returns:
        A function that performs outer step with signature:
            (params, opt_state, env_state, obsv, update_step, rng) ->
            (updated_runner_state, value_update_info, metrics)
    """
    _manipulator_update_epoch = build_manipulator_update(
        config,
        rpo_graph,
        actor_network,
        lookahead_traj_batch,
        past_in_opt_state,
        in_lookahead_update,
        manipulator_tx,
    )
    _get_rollouts = build_get_rollouts(config, env, actor_network, critic_network, rew_shaping_schedule)

    def _out_step(runner_state):
        params, opt_state, env_state, obsv, update_step, rng = runner_state

        # get rollouts for everything in rpo_graph.manipulator_rollouts using new base actor params
        # all rollouts should be using base params
        actor_params = tree_stack(
            [
                tree_stack(
                    [
                        lookahead_params["actor"]["base"][a0],
                        lookahead_params["actor"]["base"][a1],
                    ]
                )
                for (a0, a1) in rpo_graph.manipulator_rollouts
            ]
        )
        critic_params = tree_stack([params["critic"][pair] for pair in rpo_graph.manipulator_rollouts])

        rng, _rng = jax.random.split(rng)
        rollout_rng = jax.random.split(_rng, len(rpo_graph.manipulator_rollouts))
        traj_batch, advantages, targets, env_state, last_obs = jax.vmap(_get_rollouts, in_axes=(0, 0, 0, 0, None, 0))(
            actor_params, critic_params, env_state, obsv, update_step, rollout_rng
        )

        update_state = (deepcopy(params), opt_state, traj_batch, advantages, rng)
        params, opt_state, update_info = _manipulator_update_epoch(update_state)

        # dim (rollout pairs, env.agents, env_rollouts, num_env)
        mean_info = jax.tree.map(lambda x: jnp.mean(x, axis=(1, 2, 3)), traj_batch.info)
        # update_info = jax.tree.map(lambda x : jnp.mean(x), update_info)
        exp_variance = explained_variance(traj_batch.value, targets, axis=(1, 2, 3))

        metrics = {"explained_variance": exp_variance, "info": mean_info}

        runner_state = (params, opt_state, env_state, last_obs, rng)
        value_update_info = (traj_batch, targets)

        return runner_state, value_update_info, metrics

    return _out_step


def build_update_critic(config, rpo_graph, critic_network, tx):
    """Builds a function to update critic network parameters.

    The critic is trained on both inner (base) and outer (manipulator) rollouts
    to learn value functions for all agent pairs in the RPG graph.

    Args:
        config: Configuration dictionary.
        rpo_graph: RPG graph defining rollout pairs for critic training.
        critic_network: The critic network to be updated.
        tx: Optimizer for critic parameter updates.

    Returns:
        A function that updates critic with signature:
            (params, opt_state, in_value_info, out_value_info) -> (updated_params, updated_opt_state, metrics)
        where value_info contains trajectory batches and value targets.
    """

    def _update_critic(params, opt_state, in_value_update_info, out_value_update_info):
        # in_value_update_info, out_value_update_info : (traj_batch, targets)

        def _loss(critic_params, traj_batch, targets):
            values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, traj_batch.state_features[0])
            values = jnp.moveaxis(values, 1, 0)

            value_pred_clipped = traj_batch.value + (values - traj_batch.value).clip(
                -config["CLIP_EPS"]["CRITIC"], config["CLIP_EPS"]["CRITIC"]
            )
            value_losses = jnp.square(values - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return value_loss

        def full_loss_fn(critic_params):
            critic_params_array = tree_stack(
                critic_params[pair] for pair in rpo_graph.base_rollouts + rpo_graph.manipulator_rollouts
            )

            in_traj_batch, in_targets = in_value_update_info
            out_traj_batch, out_targets = out_value_update_info

            traj_batch = jax.tree.map(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                in_traj_batch,
                out_traj_batch,
            )
            targets = jnp.concatenate([in_targets, out_targets], axis=0)

            rollout_len = len(rpo_graph.base_rollouts) + len(rpo_graph.manipulator_rollouts)
            traj_batch = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (rollout_len, 2, config["NUM_STEPS"] * config["NUM_ENVS"]) + x.shape[4:],
                ),
                traj_batch,
            )
            targets = jnp.reshape(
                targets,
                (rollout_len, 2, config["NUM_STEPS"] * config["NUM_ENVS"]) + targets.shape[4:],
            )

            # vmap over rollout pairs
            in_loss = jax.vmap(_loss, in_axes=(0, 0, 0))(critic_params_array, traj_batch, targets)

            return jnp.mean(in_loss)

        loss, grads = jax.value_and_grad(full_loss_fn, has_aux=False)(params["critic"])

        updates, opt_state = tx.update(grads, opt_state, params["critic"])

        params["critic"] = optax.apply_updates(params["critic"], updates)

        metrics = {"value_loss": loss}

        return params, opt_state, metrics

    return _update_critic


def build_full_step(
    config,
    rpo_graph,
    actor_network,
    critic_network,
    env,
    txs,
    rew_shaping_schedule=None,
):
    """Builds a function that performs one complete RPG training step.

    This function orchestrates the full RPG algorithm: inner lookahead for base agents,
    outer step for manipulator training, base agent updates, and critic updates.

    Args:
        config: Configuration dictionary with all training parameters.
        rpo_graph: RPG graph defining all agent relationships and rollout structure.
        actor_network: The actor network for all agents.
        critic_network: The critic network for value estimation.
        env: The environment for rollout collection.
        txs: Tuple of optimizers (manipulator_tx, out_base_tx, in_base_tx, critic_tx).
        rew_shaping_schedule: Optional reward shaping schedule function.

    Returns:
        A function that performs complete training step with signature:
            (params, opt_state, rollout_status, update_step, rng, unused) ->
            (new_params, new_opt_state, new_rollout_status, new_update_step, new_rng, training_metrics)
        that can be called with a JAX scan for multiple training iterations.
    """
    out_manipulator_tx, out_base_tx, in_base_tx, critic_tx = txs

    in_lookahead = build_in_lookahead(
        config,
        rpo_graph,
        actor_network,
        critic_network,
        env,
        in_base_tx,
        n_lookahead=config["N_LOOKAHEAD"],
        rew_shaping_schedule=rew_shaping_schedule,
    )
    in_step = build_in_lookahead_update(config, rpo_graph, actor_network, out_base_tx)
    in_lookahead_update = build_in_lookahead_update(
        config, rpo_graph, actor_network, in_base_tx, n_lookahead=config["N_LOOKAHEAD"]
    )
    update_critic = build_update_critic(config, rpo_graph, critic_network, critic_tx)
    wandb_log_rppo_callback = build_wandb_log_rpg_callback(rpo_graph)
    wandb_checkpoint_rppo_callback = build_wandb_checkpoint_rpg_callback(config)

    def _step_all(full_runner_state, unused):
        params, opt_state, rollout_status, update_step, rng = full_runner_state
        in_env_state, in_obsv, out_env_state, out_obsv = rollout_status
        manipulator_opt_state, base_opt_state, critic_opt_state = opt_state

        # save opt_state for base but don't get the max norm part of it (index 0)
        in_opt_state = base_opt_state[1]

        # get rollouts for everything in rpo_graph.base_rollouts
        # and perform lookahead for base using rpo_graph.base_edges
        rng, in_rng, out_rng = jax.random.split(rng, 3)
        in_runner_state = (
            deepcopy(params),
            deepcopy(in_opt_state),
            in_env_state,
            in_obsv,
            update_step,
            in_rng,
        )

        (
            lookahead_params,
            lookahead_rollout_info,
            in_value_update_info,
            base_in_metrics,
            in_env_state,
            in_obsv,
        ) = in_lookahead(in_runner_state)

        # get rollouts for everything in rpo_graph.manipulator_rollouts using new base actor params
        # and perform update for manipulator using rpo_graph.manipulator_edges
        out_runner_state = (
            deepcopy(params),
            manipulator_opt_state,
            out_env_state,
            out_obsv,
            update_step,
            out_rng,
        )

        # build outer step using lookahead_traj_batch, past_lookahead_train_state
        out_step = build_out_step(
            config,
            rpo_graph,
            actor_network,
            critic_network,
            env,
            lookahead_rollout_info,
            lookahead_params,
            deepcopy(in_opt_state),
            in_lookahead_update,
            out_manipulator_tx,
            rew_shaping_schedule,
        )
        out_runner_state, out_value_update_info, manipulator_metrics = out_step(out_runner_state)

        params, manipulator_opt_state, out_env_state, out_obsv, _ = out_runner_state

        # update base
        first_lookahead_rollout_info = jax.tree.map(lambda x: x[:1], lookahead_rollout_info)
        params, base_opt_state, base_out_metrics = in_step(
            deepcopy(params), base_opt_state, first_lookahead_rollout_info
        )

        # update critic using all base and manipulator rollouts
        params, critic_opt_state, critic_metric = update_critic(
            deepcopy(params),
            critic_opt_state,
            in_value_update_info,
            out_value_update_info,
        )

        rollout_status = (in_env_state, in_obsv, out_env_state, out_obsv)
        opt_state = (manipulator_opt_state, base_opt_state, critic_opt_state)

        full_runner_state = (params, opt_state, rollout_status, update_step + 1, rng)

        base_metrics = {
            "info": base_in_metrics["info"],
            "update": base_out_metrics["update"],
            "explained_variance": base_in_metrics["explained_variance"],
        }
        metric = {
            "manipulator": manipulator_metrics,
            "base": base_metrics,
            "critic": critic_metric,
        }

        jax.debug.callback(wandb_log_rppo_callback, metric)
        jax.lax.cond(
            update_step % config["CHECKPOINT_INTERVAL"] == 0,
            lambda x: jax.debug.callback(wandb_checkpoint_rppo_callback, x),
            lambda _: None,
            params,
        )

        return full_runner_state, metric

    return _step_all


def initialize_training(config, env, rpo_config, rng):
    """Initializes all components needed for RPG training.

    This function sets up optimizers, neural networks, RPG graph, initial parameters,
    and environment states required for the training loop.

    Args:
        config: Configuration dictionary with all training hyperparameters.
        env: The environment to train in.
        rpo_config: RPG configuration defining agent structure and objectives.
        rng: JAX random key for initialization.

    Returns:
        A tuple containing:
            - rpo_graph: The constructed RPG graph
            - network_params: Initial parameters for all networks
            - actor_network: The actor network instance
            - critic_network: The critic network instance
            - txs: Tuple of all optimizers
            - opt_state: Initial optimizer states
            - rollout_status: Initial environment states and observations
            - rew_shaping_schedule: Reward shaping schedule function (if enabled)
    """

    def make_linear_schedule(lr, num_minibatch, update_epochs):
        def linear_schedule(count):
            frac = 1.0 - (count // (num_minibatch * update_epochs)) / config["NUM_UPDATES"]
            return lr * frac

        return linear_schedule

    manipulator_lr_schedule = make_linear_schedule(
        config["LR"]["MANIPULATOR"],
        config["NUM_MINIBATCHES"]["MANIPULATOR"],
        config["UPDATE_EPOCHS"]["MANIPULATOR"],
    )
    out_base_lr_schedule = make_linear_schedule(
        config["LR"]["OUT_BASE"],
        config["NUM_MINIBATCHES"]["BASE"],
        config["UPDATE_EPOCHS"]["BASE"],
    )
    in_base_lr_schedule = make_linear_schedule(
        config["LR"]["IN_BASE"],
        config["NUM_MINIBATCHES"]["BASE"],
        config["UPDATE_EPOCHS"]["BASE"],
    )
    critic_lr_schedule = make_linear_schedule(
        config["LR"]["CRITIC"],
        config["NUM_MINIBATCHES"]["MANIPULATOR"],
        config["UPDATE_EPOCHS"]["MANIPULATOR"],
    )
    if config.get("REW_SHAPING"):
        rew_shaping_schedule = optax.linear_schedule(
            init_value=1.0, end_value=1.0, transition_steps=config["NUM_UPDATES"]
        )
    else:
        rew_shaping_schedule = None

    rpo_graph = create_rpg_graph(*rpo_config)
    num_bases = len(rpo_graph.bases)
    num_manipulators = len(rpo_graph.manipulators)
    num_rollouts = len(rpo_graph.all_rollouts)

    # INIT ACTOR NETWORK
    actor_class = HanabiActor if config["ENV_NAME"] == "hanabi" else NonHanabiActor
    actor_network = actor_class(
        env.action_space(env.agents[0]).n, activation=config["ACTIVATION"], hidden_sizes=config["HIDDEN_SIZES"]
    )
    if config["ENV_NAME"] == "hanabi":
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
    else:
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    init_x = init_x.flatten()

    rng, h_rng, c_rng = jax.random.split(rng, 3)
    base_params_rng = jax.random.split(h_rng, num_bases)
    manipulator_params_rng = jax.random.split(c_rng, num_manipulators)

    actor_network_params = {
        "base": {
            h: pickle.load(open(f"{str(MODEL_DIR)}/{path}", "rb")) if path else actor_network.init(g, init_x)
            for g, h, path in zip(base_params_rng, rpo_graph.bases, rpo_graph.base_init_paths)
        },
        "manipulator": {
            c: pickle.load(open(path, "rb")) if path else actor_network.init(g, init_x)
            for g, c, path in zip(manipulator_params_rng, rpo_graph.manipulators, rpo_graph.manipulator_init_paths)
        },
    }

    # INIT CRITIC NETWORK
    critic_class = HanabiCritic if config["ENV_NAME"] == "hanabi" else NonHanabiCritic
    critic_network = critic_class(
        action_dim=env.action_space(env.agents[0]).n,
        num_agents=env.num_agents,
        activation=config["ACTIVATION"],
    )
    init_x = jnp.zeros(env.state_feature_size())

    init_x = init_x.flatten()

    rng, _rng = jax.random.split(rng, 2)
    params_rng = jax.random.split(_rng, num_rollouts)

    critic_network_params = jax.vmap(critic_network.init, in_axes=(0, None))(params_rng, init_x)
    # transpose into list of pytrees
    critic_network_params = [tree_slice(critic_network_params, i) for i in range(num_rollouts)]
    critic_network_params = {r: p for r, p in zip(rpo_graph.all_rollouts, critic_network_params)}

    network_params = {"actor": actor_network_params, "critic": critic_network_params}

    if config["ANNEAL_LR"]:
        manipulator_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["MANIPULATOR"]),
            optax.adam(learning_rate=manipulator_lr_schedule, eps=1e-5),
        )
        if config.get("BASE_ADAM"):
            out_base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["OUT_BASE"]),
                optax.adam(learning_rate=out_base_lr_schedule, eps=1e-5, eps_root=1e-5),
            )
            in_base_tx = optax.adam(learning_rate=in_base_lr_schedule, eps=1e-5, eps_root=1e-5)
        else:
            out_base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["OUT_BASE"]),
                optax.sgd(learning_rate=out_base_lr_schedule),
            )
            in_base_tx = optax.sgd(learning_rate=in_base_lr_schedule)
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["CRITIC"]),
            optax.adam(learning_rate=critic_lr_schedule, eps=1e-5),
        )
    else:
        if config.get("MANIPULATOR_SGD"):
            manipulator_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["MANIPULATOR"]),
                optax.sgd(config["LR"]["MANIPULATOR"]),
            )
        else:
            manipulator_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["MANIPULATOR"]),
                optax.adam(config["LR"]["MANIPULATOR"], eps=1e-5),
            )
        if config.get("BASE_ADAM"):
            out_base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["OUT_BASE"]),
                optax.adam(config["LR"]["OUT_BASE"], eps=1e-5, eps_root=1e-5),
            )
            in_base_tx = optax.adam(config["LR"]["IN_BASE"], eps=1e-5, eps_root=1e-5)
        else:
            out_base_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["OUT_BASE"]),
                optax.optimistic_gradient_descent(config["LR"]["OUT_BASE"]),
            )
            in_base_tx = optax.optimistic_gradient_descent(config["LR"]["IN_BASE"])
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]["CRITIC"]),
            optax.adam(config["LR"]["CRITIC"], eps=1e-5),
        )

    txs = manipulator_tx, out_base_tx, in_base_tx, critic_tx

    manipulator_opt_state = manipulator_tx.init(actor_network_params["manipulator"])
    base_opt_state = out_base_tx.init(actor_network_params["base"])
    critic_opt_state = critic_tx.init(critic_network_params)

    opt_state = (manipulator_opt_state, base_opt_state, critic_opt_state)

    # vect_reset = jnp.vectorize(env.reset, signature='(2)->(),()')
    # vmap reset over three dimensions, would like to just use vectorize but it doesn't work on dictionary outputs
    vect_reset = jax.vmap(jax.vmap(env.reset))

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, (len(rpo_graph.base_rollouts), config["NUM_ENVS"]))
    in_obsv, in_env_state = vect_reset(reset_rng)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, (len(rpo_graph.manipulator_rollouts), config["NUM_ENVS"]))
    out_obsv, out_env_state = vect_reset(reset_rng)

    rollout_status = (in_env_state, in_obsv, out_env_state, out_obsv)

    return (
        rpo_graph,
        network_params,
        actor_network,
        critic_network,
        txs,
        opt_state,
        rollout_status,
        rew_shaping_schedule,
    )


def make_train(config, rpo_config):
    """Creates a training function for the RPG algorithm.

    This is the main entry point for RPG training. It sets up the environment,
    wraps it appropriately, and returns a training function.

    Args:
        config: Configuration dictionary containing all environment and training parameters.
        rpo_config: RPG configuration defining RPG algorithm.

    Returns:
        A function with signature train(rng) -> {"runner_state": final_state, "metrics": training_metrics}
        that runs the complete RPG training loop.
    """
    env = registration_wrapper.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = wrap_env(env, config)

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = {}
    config["MINIBATCH_SIZE"]["MANIPULATOR"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]["MANIPULATOR"]
    )
    config["MINIBATCH_SIZE"]["BASE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]["BASE"]

    def train(rng):
        rng, _rng = jax.random.split(rng)
        (
            rpo_graph,
            network_params,
            actor_network,
            critic_network,
            txs,
            opt_state,
            rollout_status,
            rew_shaping_schedule,
        ) = initialize_training(config, env, rpo_config, _rng)

        rng, _rng = jax.random.split(rng)
        update_step = 0
        full_runner_state = (
            network_params,
            opt_state,
            rollout_status,
            update_step,
            rng,
        )

        _update_step = build_full_step(
            config,
            rpo_graph,
            actor_network,
            critic_network,
            env,
            txs,
            rew_shaping_schedule,
        )

        # rollout everything in base_rollouts to get base_lookahead
        # do base_lookahead to get lookahead_params
        # rollout everything in manipulator_rollouts using lookahead_params to get manipulator shaping rollouts
        if config.get("TRACE_DIR"):
            vupdate_step = jax.jit(_update_step)
            with jax.profiler.trace(config["TRACE_DIR"]):
                full_runner_state, first_metric = vupdate_step(full_runner_state, None)
                jax.block_until_ready(full_runner_state)
            runner_state, metric = jax.lax.scan(_update_step, full_runner_state, None, config["NUM_UPDATES"] - 1)
            metric = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), metric, first_metric)
        else:
            runner_state, metric = jax.lax.scan(_update_step, full_runner_state, None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="rpg_overcooked")
def main(hydra_config):
    """Main entry point for running RPG training with Hydra configuration.

    This function handles configuration loading, environment setup, W&B initialization,
    and orchestrates the complete training pipeline including visualization and checkpointing.

    Args:
        hydra_config: Hydra configuration object containing all training parameters.
    """
    config = cast(dict, OmegaConf.to_container(hydra_config))

    if config.get("layout"):
        layout_name = config["layout"]
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    if config["ENV_KWARGS"].get("num_cards_of_rank"):
        config["ENV_KWARGS"]["num_cards_of_rank"] = np.array(config["ENV_KWARGS"]["num_cards_of_rank"])
    if config.get("PAYOFF_MATRIX"):
        payoff_matrix = jnp.array(config["PAYOFF_MATRIX"])
        if config["ENV_NAME"][0:5] == "storm":
            config["ENV_KWARGS"]["payoff_matrix"] = payoff_matrix
        else:
            config["ENV_KWARGS"]["payoffs"] = payoff_matrix

    if config.get("WANDB_DIR"):
        if not os.path.exists(config["WANDB_DIR"]):
            os.makedirs(config["WANDB_DIR"])

    wandb.init(
        project=config["PROJECT"],
        tags=config["TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config["RPG_ALG"],
        dir=config.get("WANDB_DIR"),
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_train = jax.random.split(rng, 2)

    rpo_config = make_rpg_alg(config)
    train = make_train(config, rpo_config)
    start = time.time()
    outs = train(rng_train)

    pickle.dump(outs, open("pretrained_policies/rpg_overcooked.pkl", "wb"))

    params = outs["runner_state"][0]
    metrics = outs["metrics"]

    jax.block_until_ready(metrics)
    print("compute time", time.time() - start)
    print(f"trace time: {time.time() - start}")

    rpo_graph = create_rpg_graph(*rpo_config)
    wandb_visualize(config, rpo_graph, params, rng)

    wandb_checkpoint_rppo_callback = build_wandb_checkpoint_rpg_callback(config)
    wandb_checkpoint_rppo_callback(params)


if __name__ == "__main__":
    main()
