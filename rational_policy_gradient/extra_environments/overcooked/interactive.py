import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked.overcooked import Actions


def redraw(state, obs, extras):
    extras["viz"].render(extras["agent_view_size"], state, highlight=False)


def reset(key, env, extras):
    key, subkey = jax.random.split(extras["rng"])
    obs, state = extras["jit_reset"](subkey)

    extras["rng"] = key
    extras["obs"] = obs
    extras["state"] = state

    redraw(state, obs, extras)


def step(env, action, extras):
    key, sample_key = jax.random.split(extras["rng"])

    policy = extras["policy"]
    params = extras["params"]

    pi = policy.apply(params, extras["obs"]["agent_1"].flatten())
    other_action = pi.sample(seed=sample_key)

    actions = {"agent_0": jnp.array(action), "agent_1": other_action}
    print("Actions : ", actions)
    key, step_key = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(step_key, extras["state"], actions)
    extras["obs"] = obs
    extras["state"] = state
    print(
        f"t={state.time}: reward={reward['agent_0']}, agent_dir={state.agent_dir_idx}, agent_inv={state.agent_inv}, done = {done['__all__']}, pos={state.agent_pos}"
    )

    if extras["debug"]:
        layers = [f"player_{i}_loc" for i in range(2)]
        layers.extend([f"player_{i // 4}_orientation_{i % 4}" for i in range(8)])
        layers.extend(
            [
                "pot_loc",
                "counter_loc",
                "onion_disp_loc",
                "tomato_disp_loc",
                "plate_disp_loc",
                "serve_loc",
                "onions_in_pot",
                "tomatoes_in_pot",
                "onions_in_soup",
                "tomatoes_in_soup",
                "soup_cook_time_remaining",
                "soup_done",
                "plates",
                "onions",
                "tomatoes",
                "urgency",
            ]
        )
        print("obs_shape: ", obs["agent_0"].shape)
        print("OBS: \n", obs["agent_0"])
        debug_obs = jnp.transpose(obs["agent_0"], (2, 0, 1))
        for i, layer in enumerate(layers):
            print(layer)
            print(debug_obs[i])
    # print(f"agent obs =\n {obs}")

    if done["__all__"] or (jnp.array([action, action]) == Actions.done).any():
        key, subkey = jax.random.split(key)
        reset(subkey, env, extras)
    else:
        redraw(state, obs, extras)

    extras["rng"] = key


def key_handler(env, extras, event):
    print("pressed", event.key)

    if event.key == "backspace":
        key, subkey = jax.random.split(extras["rng"])
        reset(subkey, env, extras)
        extras["rng"] = key
        return

    if event.key == "left":
        step(env, Actions.left, extras)
        return
    if event.key == "right":
        step(env, Actions.right, extras)
        return
    if event.key == "up":
        step(env, Actions.forward, extras)
        return

    # Spacebar
    if event.key == " ":
        step(env, Actions.toggle, extras)
        return
    if event.key == "[":
        step(env, Actions.pickup, extras)
        return
    if event.key == "]":
        step(env, Actions.drop, extras)
        return

    if event.key == "enter":
        step(env, Actions.done, extras)
        return


def key_handler_overcooked(env, extras, event):
    print("pressed", event.key)

    if event.key == "backspace":
        extras["jit_reset"]((env, extras))
        return

    if event.key == "left":
        step(env, Actions.left, extras)
        return
    if event.key == "right":
        step(env, Actions.right, extras)
        return
    if event.key == "up":
        # step(env, Actions.forward, extras)
        step(env, Actions.up, extras)
        return
    if event.key == "down":
        step(env, Actions.down, extras)
        return

    # Spacebar
    if event.key == " ":
        step(env, Actions.interact, extras)
        return
    if event.key == "tab":
        step(env, Actions.stay, extras)
        return
    if event.key == "enter":
        step(env, Actions.done, extras)
        return
