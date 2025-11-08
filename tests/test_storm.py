from adv_div_rational import *

def build_in_lookahead(config, actor_network, critic_network, env, tx):

    _env_step = build_env_step(config, env, actor_network, critic_network)

    _horse_update_epoch = build_horse_update_epoch(config, actor_network, critic_network, tx, use_value_loss=False)
    _calculate_gae = build_calculate_gae(config)

    def in_lookahead(runner_state):
        # runner_state : (params, opt_state, env_state, obsv, rng)

        def _step_ahead(runner_state, unused):
            # TODO : could potentially vectorize this function across ego and alter for minor speedup

            params, opt_state, env_state, obsv, rng = runner_state
            horse_params = params['actor']["horse"]
            carrot_params = params['actor']["carrot"]
            ego_critic_params = tree_slice(params['critic']['horse'],0)
            alter_critic_params = tree_slice(params['critic']['horse'],1)
                        
            # obsv has shape {agent_name : (num_particles, num_envs, obs_dim)}
            # env_state has shape (num_particles, num_envs, obs_dim)

            # split env_state and obsv in half between first (ego) and second (alter) player along envs axis
            ego_env_state = tree_slice(env_state,0)
            alter_env_state = tree_slice(env_state,1)
            ego_obsv = tree_slice(obsv,0)
            alter_obsv = tree_slice(obsv,1)

            ego_actor_params = tree_stack((tree_slice(horse_params,0), tree_slice(horse_params,1)))

            ego_params = {'actor': ego_actor_params, 'critic': ego_critic_params}

            ego_state = (ego_params, ego_env_state, ego_obsv, rng)
            ego_state, ego_traj_batch = jax.lax.scan(
                _env_step, ego_state, None, config["NUM_STEPS"]
            )
            # update runner_state
            _, ego_env_state, ego_last_obsv, rng = ego_state

            # swap carrot and horse params
            alter_actor_params = tree_stack((tree_slice(carrot_params,1), tree_slice(carrot_params,1)))
            alter_params = {'actor': alter_actor_params, 'critic': alter_critic_params}

            alter_state = (alter_params, alter_env_state, alter_obsv, rng)
            alter_state, alter_traj_batch = jax.lax.scan(
                _env_step, alter_state, None, config["NUM_STEPS"]
            )
            _, alter_env_state, alter_last_obsv, rng = alter_state


            # stack runner_state
            env_state = tree_stack([ego_env_state, alter_env_state])
            last_obs = tree_stack([ego_last_obsv, alter_last_obsv])
                        
            # horse_last_obs_batch = jnp.stack([ego_last_obsv[env.agents[0]], alter_last_obsv[env.agents[1]]])
            # carrot_last_obs_batch = jnp.stack([ego_last_obsv[env.agents[1]], alter_last_obsv[env.agents[0]]])
            # last_obs_batch = jnp.stack([horse_last_obs_batch, carrot_last_obs_batch])

            horse_traj_batch = tree_stack([tree_slice(ego_traj_batch,jnp.s_[:,0]), tree_slice(ego_traj_batch,jnp.s_[:,1])])
            carrot_traj_batch = tree_stack([jax.tree.map(lambda x : jnp.zeros_like(x), tree_slice(ego_traj_batch,jnp.s_[:,1])), jax.tree.map(lambda x : jnp.zeros_like(x), tree_slice(alter_traj_batch,jnp.s_[:,0]))])
            traj_batch = tree_stack([horse_traj_batch, carrot_traj_batch])
            # swap axes to go from (horse or carrot, ego or alter, env rollouts, particles, envs, -1) to (horse or carrot, ego or alter, particles, env rollouts, envs, -1)
            traj_batch = jax.tree.map(lambda x : jnp.swapaxes(x, 2, 3), traj_batch)

            # jax.debug.print("swapped traj_batch.action[0,:,0] {x}", x=jnp.swapaxes(traj_batch.action, 3, 4)[0,:,0])

            # flatten obs dimension
            # last_obs_batch = last_obs_batch.reshape(2, env.num_agents, config['NUM_PARTICLES'], config['NUM_ENVS'], -1)
            

            # vmap over ego and alter, and all particles
            # last_state_features = jax.vmap(
            #     jax.vmap(env.get_state_features), 
            #     in_axes=(0,0), 
            # )(env_state, last_obs)

            # # vmap over ego or alter, and all particles
            # last_val = jax.vmap(
            #     jax.vmap(critic_network.apply, in_axes=(0,0), out_axes=0), 
            #     in_axes=(0,0),
            #     out_axes=0
            # )(params['critic']['horse'], last_state_features)
            # # bring agent dim to the front to get (agent, ego or alter, particles, env rollouts)
            # last_val = jnp.moveaxis(last_val, 3, 0)

            # TODO check with rewards instead of advantages

            # horse_last_val = jnp.stack([last_val[0,0], last_val[1,1]])
            # carrot_last_val = jnp.stack([last_val[1,0], last_val[0,1]])
            # last_val = jnp.stack([horse_last_val, carrot_last_val])

            # vmap over two sets of params, ego and alter, and all particles
            # advantages, targets = jax.vmap(
            #     jax.vmap(
            #         jax.vmap(_calculate_gae, in_axes=(0,0)), 
            #         in_axes=(0,0)
            #     ),
            #     in_axes=(0,0)
            # )(traj_batch, last_val)

            # TODO revert to GAE
            advantages = jnp.zeros_like(traj_batch.reward)
            for i in range(4):
                advantages = advantages.at[:,:,:,i*20:(i+1)*20,:].set(jnp.cumsum(traj_batch.reward[:,:,:,i*20:(i+1)*20,:][:,:,:,::-1,:], axis=3)[:,:,:,::-1,:])
            # jax.debug.print("{x}", x=advantages)
            targets = jnp.zeros_like(traj_batch.reward)

            # save this rng value to be used later on
            update_rng = rng

            update_state = (params, opt_state, traj_batch, advantages, targets, update_rng)
            update_state, loss_info = jax.lax.scan(
                _horse_update_epoch, update_state, None, config["UPDATE_EPOCHS"]["HORSE"]
            )

            # update runner values
            params, opt_state, _, _, _, rng = update_state

            runner_state = (params, opt_state, env_state, last_obs, rng)

            rollout_info = (traj_batch, advantages, targets, update_rng)

            return runner_state, (rollout_info, loss_info, env_state, last_obs)

        new_runner_state, (lookahead_traj_batches, loss_info, env_states, last_obs) = jax.lax.scan(
            _step_ahead, runner_state, None, config["N_LOOKAHEAD"]
        )

        new_params = new_runner_state[0]

        return lookahead_traj_batches, new_params, loss_info, (tree_slice(env_states,0), tree_slice(last_obs,0))
    
    return in_lookahead
    


def build_step_copy(config, actor_network, critic_network, env, txs):

    out_carrot_tx, out_horse_tx, in_horse_tx = txs

    horse_reg_tx = optax.chain(
        optax.clip_by_global_norm(0.5), 
        optax.sgd(config['LR']['REGULARIZER_HORSE'])
    )  
        
    in_lookahead = build_in_lookahead(config, actor_network, critic_network, env, in_horse_tx)
    in_step = build_in_lookahead_update(config, actor_network, critic_network, out_horse_tx, n_lookahead=1, use_value_loss=True)
    
    def _step_copy(full_runner_state, unused):

        params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, rng = full_runner_state
        carrot_opt_state, horse_opt_state = opt_state
        rng, in_rng, out_rng = jax.random.split(rng, 3)

        in_runner_state = (params.copy(), horse_opt_state, in_env_state, in_obsv, in_rng)
        lookahead_rollout_info, lookahead_params, loss_info, (in_env_state, in_obsv) = in_lookahead(in_runner_state)

        lookahead_traj_batch = lookahead_rollout_info[0]

        # (num_lookahead, horse or carrot, ego or alter, particles, env rollouts, envs, -1)
        inner_info = jax.tree.map(lambda x : jnp.mean(x[:,0,:,:,:,:], axis=(3,4)), lookahead_traj_batch.info)
        # loss_info = jax.tree.map(lambda x : jnp.mean(x, axis=1), loss_info)
        # first_lookahead_traj_batch = jax.tree.map(lambda x : x[:1,], lookahead_rollout_info)
        # params, horse_opt_state, horse_update = in_step(params.copy(), horse_opt_state, first_lookahead_traj_batch)
        # ego_lookahead_params = tree_slice(lookahead_params['actor']['horse'],0)
        # params['actor']['horse'] = tree_stack([ego_lookahead_params, ego_lookahead_params])
        params['actor']['horse'] = lookahead_params['actor']['horse']
        params['critic']['horse'] = lookahead_params['critic']['horse']

        carrot_metrics = 0
        # params['actor']['carrot'] = jax.tree.map(lambda x : jnp.flip(x.copy(), axis=0), params['actor']['horse'])

        metric = {"carrot": carrot_metrics, "horse": {"update": loss_info, "info": inner_info}}

        opt_state = (carrot_opt_state, horse_opt_state)
        
        full_runner_state = (params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, rng)

        return full_runner_state, metric
    
    return _step_copy


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    if config.get("CONC_SPACES"):
        env = ConcatenatePlayerSpaces(env)

    if config.get('OBS_DELAY'):
        env = DelayedObsWrapper(env, delay=config['OBS_DELAY'])

    env = make_world_state_wrapper(config['ENV_NAME'], env)
    
    if "MPE" in config["ENV_NAME"]:
        env = MPELogWrapper(env)
    else:
        env = LogWrapper(env)

    if config.get("DUALING_ENV"):
        env = DualingEnv(env, K=config["NUM_PARTICLES"])

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

    def make_linear_schedule(lr, num_minibatch, update_epochs):
        def linear_schedule(count):
            frac = 1.0 - (count // (num_minibatch * update_epochs)) / config["NUM_UPDATES"]
            return lr * frac
        return linear_schedule
    
    carrot_lr_schedule = make_linear_schedule(config['LR']["CARROT"], config['NUM_MINIBATCHES']['CARROT'], config["UPDATE_EPOCHS"]["HORSE"])
    out_horse_lr_schedule = make_linear_schedule(config['LR']["OUT_HORSE"], config['NUM_MINIBATCHES']['HORSE'], config["UPDATE_EPOCHS"]["HORSE"])
    in_horse_lr_schedule = make_linear_schedule(config['LR']["IN_HORSE"], config['NUM_MINIBATCHES']['HORSE'], config["UPDATE_EPOCHS"]["HORSE"])

    def train(rng):

        # INIT ACTOR NETWORK
        actor_network = Actor(env.action_space(env.agents[0]).n, activation=config["ACTIVATION"])
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        
        init_x = init_x.flatten()

        rng, _rng = jax.random.split(rng)
        params_rng = jax.random.split(_rng, 2*config["NUM_PARTICLES"]*env.num_agents)

        actor_network_params_flat = jax.vmap(actor_network.init, in_axes=(0,None))(params_rng, init_x)
        actor_network_params_array = jax.tree.map(lambda x : x.reshape((2,env.num_agents, config["NUM_PARTICLES"])+x.shape[1:]), actor_network_params_flat)
        actor_network_params = {"horse": tree_slice(actor_network_params_array, 0), "carrot": tree_slice(actor_network_params_array, 1)}

        # INIT CRITIC NETWORK
        critic_network = Critic(num_agents=2, activation=config["ACTIVATION"])
        init_x = jnp.zeros(env.state_feature_size())

        init_x = init_x.flatten()

        rng, _rng = jax.random.split(rng)
        params_rng = jax.random.split(_rng, config["NUM_PARTICLES"]*env.num_agents + config["NUM_PARTICLES"]**2)

        critic_network_params_flat = jax.vmap(critic_network.init, in_axes=(0,None))(params_rng, init_x)
        horse_critics = jax.tree.map(lambda x : jnp.reshape(x[:config["NUM_PARTICLES"]*env.num_agents], (env.num_agents, config["NUM_PARTICLES"])+x.shape[1:]), critic_network_params_flat)
        carrot_critics = jax.tree.map(lambda x : jnp.reshape(x[config["NUM_PARTICLES"]*env.num_agents:], (config["NUM_PARTICLES"], config["NUM_PARTICLES"])+x.shape[1:]), critic_network_params_flat)
        critic_network_params = {"horse": horse_critics, "carrot": carrot_critics}

        network_params = {"actor": actor_network_params, "critic": critic_network_params}

        if config["ANNEAL_LR"]:
            carrot_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]['CARROT']), 
                optax.adam(learning_rate=carrot_lr_schedule, eps=1e-5)
            )
            # out_horse_tx = optax.adam(learning_rate=out_horse_lr_schedule, eps=1e-5)
            # in_horse_tx = optax.adam(learning_rate=in_horse_lr_schedule, eps=1e-5)  
            out_horse_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]['CARROT']), 
                optax.sgd(learning_rate=out_horse_lr_schedule)
            )
            in_horse_tx = optax.sgd(learning_rate=in_horse_lr_schedule)
        else: 
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
        vect_reset = jax.vmap(jax.vmap(jax.vmap(env.reset)))

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, (2, config["NUM_PARTICLES"], config["NUM_ENVS"]))
        in_obsv, in_env_state = vect_reset(reset_rng)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, (config["NUM_PARTICLES"], config["NUM_PARTICLES"], config["NUM_ENVS"]))
        out_obsv, out_env_state = vect_reset(reset_rng)
    
        rng, _rng = jax.random.split(rng)
        full_runner_state = (network_params, opt_state, in_env_state, in_obsv, out_env_state, out_obsv, rng)

        _update_step = build_step_copy(config, actor_network, critic_network, env, txs)

        runner_state, metric = jax.lax.scan(
            _update_step, full_runner_state, None, config["NUM_UPDATES"]
        )
            
        return {"runner_state": runner_state, "metrics": metric}

    return train


def test_step_copy():

    seed = 36

    config = OmegaConf.load('tests/configs/test_storm_step_copy.yaml')
    config = OmegaConf.to_container(config) 

    filename = f"metrics/{config['ENV_NAME']}/ablating_to_ippo/dicelambda{config['DICE_LAMBDA']}_lookahead{config['N_LOOKAHEAD']}lr_carrot{config['LR']['CARROT']}_inhorse{config['LR']['IN_HORSE']}_outhorse{config['LR']['OUT_HORSE']}_reg{config['LR']['REGULARIZER_HORSE']}_steps{config['TOTAL_TIMESTEPS']}_env{config['NUM_ENVS']}_clip{config['CLIP_EPS']['CARROT']}_steps{config['NUM_STEPS']}"
    exp_dir = "tests"

    # create dir filename
    import os
    os.makedirs(f"{exp_dir}/{filename}", exist_ok=True)

    with open(f"{exp_dir}/{filename}/config.yaml", "w") as f:
        OmegaConf.save(config, f)
    
    if config["ENV_KWARGS"].get("layout"):
        layout_name = config["ENV_KWARGS"]["layout"]
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]
    if config.get("PAYOFF_MATRIX"):
        payoff_matrix = jnp.array(config["PAYOFF_MATRIX"])
        config["ENV_KWARGS"]["payoff_matrix"] = payoff_matrix

    rng = jax.random.PRNGKey(seed)
    rng, rng_train = jax.random.split(rng, 2)

    train = make_train(config)
    start = time.time()
    outs = train(rng_train)

    params = outs["runner_state"][0]
    metrics = outs["metrics"]

    jax.block_until_ready(metrics)
    print('compute time', time.time() - start)
    print(f"trace time: {time.time() - start}")

    visualize(config, params, rng, exp_dir, filename)

    plot_run(metrics, exp_dir, filename)


if __name__ == "__main__":
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update('jax_disable_jit', True)
    jnp.set_printoptions(suppress=True)
    start = time.time()
    test_step_copy()
    print('total time', time.time()-start)