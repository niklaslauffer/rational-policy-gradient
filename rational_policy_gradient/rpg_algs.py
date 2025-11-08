"""A library of preconstructed Rational Policy Optimization Algorithms"""

from rational_policy_gradient.utils import RPO_Edge


def make_rpg_alg(config):
    if config["RPG_ALG"] == "doublesided_RAD":
        return build_doublesided_RAD(config)
    elif config["RPG_ALG"] == "doublesided_RAP":
        return build_doublesided_RAP(config)
    elif config["RPG_ALG"] == "singlesided_RAP":
        return build_singlesided_RAP(config)
    elif config["RPG_ALG"] == "doublesided_RAP_diff":
        return build_doublesided_RAP_diff(config)
    elif config["RPG_ALG"] == "doublesided_RAT":
        return build_doublesided_RAT(config)
    elif config["RPG_ALG"] == "doublesided_RAT_selfpartnerplay":
        return build_doublesided_RAT_selfpartnerplay(config)
    elif config["RPG_ALG"] == "singlesided_RAT":
        return build_singlesided_RAT(config)
    elif config["RPG_ALG"] == "doublesided_RPAIRED":
        return build_doublesided_RPAIRED(config)
    elif config["RPG_ALG"] == "singlesided_RPAIRED":
        return build_singlesided_RPAIRED(config)
    elif config["RPG_ALG"] == "doublesided_RPAIRED_selfpartnerplay":
        return build_doublesided_RPAIRED_selfpartnerplay(config)
    elif config["RPG_ALG"] == "doublesided_simplified_RPAIRED":
        return build_doublesided_simplified_RPAIRED(config)
    elif config["RPG_ALG"] == "fixed_selfplay":
        return build_fixed_selfplay(config)
    elif config["RPG_ALG"] == "doublesided_selfplay":
        return build_doublesided_selfplay(config)
    elif config["RPG_ALG"] == "doublesided_OAD":
        return build_doublesided_OAD(config)
    elif config["RPG_ALG"] == "singlesided_OAP":
        return build_singlesided_OAP(config)
    elif config["RPG_ALG"] == "doublesided_OAP":
        return build_doublesided_OAP(config)
    elif config["RPG_ALG"] == "singlesided_OAT":
        return build_singlesided_OAT(config)
    elif config["RPG_ALG"] == "doublesided_OAT":
        return build_doublesided_OAT(config)
    elif config["RPG_ALG"] == "doublesided_simpOPAIRED":
        return build_doublesided_simpOPAIRED(config)
    elif config["RPG_ALG"] == "doublesided_OPAIRED":
        return build_doublesided_OPAIRED(config)
    elif config["RPG_ALG"] == "singlesided_OPAIRED":
        return build_singlesided_OPAIRED(config)
    else:
        raise ValueError(f"Invalid RPG_ALG: {config['RPG_ALG']}")


def build_doublesided_RAD(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    # adversarial diversity
    N = config["NUM_PARTICLES"]
    bases = {f"base_{i}": None for i in range(N)}
    manipulators = {f"manipulator_{i}": None for i in range(N)}
    base_selfplay = [
        RPO_Edge(
            id=f"base_{i}",
            source=f"base_{i}",
            target=f"manipulator_{i}",
            weight=1.0 - N * partnerplay_ratio,
            source_player_idx=(0, 1),
        )
        for i in range(N)
    ]
    base_crossplay = [
        RPO_Edge(
            id=f"base_{i}",
            source=f"base_{i}",
            target=f"base_{j}",
            weight=partnerplay_ratio,
            source_player_idx=(0, 1),
        )
        for i in range(N)
        for j in range(N)
    ]
    base_objective = base_selfplay + base_crossplay
    manipulator_max = [
        RPO_Edge(
            id=f"manipulator_{i}",
            source=f"base_{i}",
            target=f"base_{i}",
            weight=1.0,
            source_player_idx=(0, 1),
            shaped_base=f"base_{i}",
        )
        for i in range(N)
    ]
    manipulator_min = [
        RPO_Edge(
            id=f"manipulator_{i}",
            source=f"base_{i}",
            target=f"base_{j}",
            weight=-1.0 * config["OFF_DIAG_FACTOR"] / (N - 1),
            source_player_idx=(0, 1),
            shaped_base=f"base_{i}",
        )
        for i in range(N)
        for j in range(N)
        if i != j
    ]
    manipulator_objective = manipulator_max + manipulator_min
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RAP(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - partnerplay_ratio,
            (0, 1),
        ),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RAP_diff(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {
        "victim_base": config.get("VICTIM_PARAM_PATH"),
        "adversary_base": None,
        "antagonist_base": None,
    }
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "antagonist_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - 2 * partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "antagonist_base",
            "adversary_base",
            1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_RAP(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    ego_agent = 0
    alter_agent = 1 - ego_agent
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - partnerplay_ratio,
            ego_agent,
        ),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            alter_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RAT(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": None, "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RAT_selfpartnerplay(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": None, "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "victim_base",
            "victim_base",
            "adversary_base",
            1.0 - partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("victim_base", "victim_base", "victim_base", partnerplay_ratio, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_RAT(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": None, "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    ego_agent = 0
    alter_agent = 1 - ego_agent
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, alter_agent),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            alter_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_simplified_RPAIRED(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": None, "adversary_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - 2 * partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
        RPO_Edge(
            "adversary_manipulator",
            "adversary_base",
            "adversary_base",
            1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RPAIRED(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {"victim_base": None, "adversary_base": None, "antagonist_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "antagonist_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - 2 * partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
        RPO_Edge(
            "adversary_manipulator",
            "antagonist_base",
            "adversary_base",
            1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_RPAIRED_selfpartnerplay(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {
        "victim_base": config.get("VICTIM_PARAM_PATH"),
        "adversary_base": None,
        "antagonist_base": None,
    }
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "antagonist_base",
            partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - 2 * partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge(
            "victim_base",
            "victim_base",
            "adversary_base",
            1.0 - partnerplay_ratio,
            (0, 1),
        ),
        RPO_Edge("victim_base", "victim_base", "victim_base", partnerplay_ratio, (0, 1)),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
        RPO_Edge(
            "adversary_manipulator",
            "antagonist_base",
            "adversary_base",
            1.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_RPAIRED(config):
    partnerplay_ratio = config["PARTNERPLAY_RATIO"]

    bases = {
        "victim_base": config.get("VICTIM_PARAM_PATH"),
        "adversary_base": None,
        "antagonist_base": None,
    }
    manipulators = {"adversary_manipulator": None}
    ego_agent = 0
    alter_agent = 1 - ego_agent
    base_objective = [
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "victim_base",
            partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "antagonist_base",
            partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge(
            "adversary_base",
            "adversary_base",
            "adversary_manipulator",
            1.0 - 2 * partnerplay_ratio,
            ego_agent,
        ),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, alter_agent),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, alter_agent),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "victim_base",
            "adversary_base",
            -1.0,
            alter_agent,
            shaped_base="adversary_base",
        ),
        RPO_Edge(
            "adversary_manipulator",
            "antagonist_base",
            "adversary_base",
            1.0,
            alter_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_fixed_selfplay(config):
    bases = {"fixed_base": config.get("VICTIM_PARAM_PATH"), "learning_base": None}
    manipulators = {"adversary_manipulator": None}
    base_objective = [
        RPO_Edge("learning_base", "learning_base", "fixed_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "adversary_manipulator",
            "learning_base",
            "fixed_base",
            0.0,
            (0, 1),
            shaped_base="learning_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_selfplay(config):
    bases = {"learning_base": None}
    manipulators = {"dummy_manipulator": None}
    base_objective = [
        RPO_Edge("learning_base", "learning_base", "learning_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "learning_base",
            "learning_base",
            0.0,
            (0, 1),
            shaped_base="learning_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_OAD(config):
    # adversarial diversity
    N = config["NUM_PARTICLES"]
    bases = {f"base_{i}": None for i in range(N)}
    manipulators = {"dummy_manipulator": None}
    base_max = [
        RPO_Edge(
            id=f"base_{i}",
            source=f"base_{i}",
            target=f"base_{i}",
            weight=1.0,
            source_player_idx=(0, 1),
        )
        for i in range(N)
    ]
    base_min = [
        RPO_Edge(
            id=f"base_{i}",
            source=f"base_{i}",
            target=f"base_{j}",
            weight=-1.0 * config["OFF_DIAG_FACTOR"] / (N - 1),
            source_player_idx=(0, 1),
        )
        for i in range(N)
        for j in range(N)
        if i != j
    ]
    base_objective = base_max + base_min
    manipulator_objective = [
        RPO_Edge("dummy_manipulator", "base_0", "base_0", 0.0, (0, 1), shaped_base="base_0"),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_OAP(config):
    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"dummy_manipulator": None}
    ego_agent = 0
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, ego_agent),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "adversary_base",
            "victim_base",
            0.0,
            ego_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_OAP(config):
    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"dummy_manipulator": None}
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "adversary_base",
            "victim_base",
            0.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_OAT(config):
    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"dummy_manipulator": None}
    ego_agent = 0
    alter_agent = 1 - ego_agent
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "victim_base", 1.0, ego_agent),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, alter_agent),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "adversary_base",
            "victim_base",
            0.0,
            ego_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_OAT(config):
    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"dummy_manipulator": None}
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, (0, 1)),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "adversary_base",
            "victim_base",
            0.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_simpOPAIRED(config):
    bases = {"victim_base": config.get("VICTIM_PARAM_PATH"), "adversary_base": None}
    manipulators = {"dummy_manipulator": None}
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "adversary_base", 1.0, (0, 1)),
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, (0, 1)),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "adversary_base",
            "victim_base",
            0.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_doublesided_OPAIRED(config):
    bases = {
        "victim_base": config.get("VICTIM_PARAM_PATH"),
        "adversary_base": None,
        "antagonist_base": None,
    }
    manipulators = {"dummy_manipulator": None}
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "antagonist_base", 1.0, (0, 1)),
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, (0, 1)),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, (0, 1)),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, (0, 1)),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "victim_base",
            "adversary_base",
            0.0,
            (0, 1),
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective


def build_singlesided_OPAIRED(config):
    bases = {
        "victim_base": config.get("VICTIM_PARAM_PATH"),
        "adversary_base": None,
        "antagonist_base": None,
    }
    manipulators = {"dummy_manipulator": None}
    ego_agent = 0
    alter_agent = 1 - ego_agent
    base_objective = [
        RPO_Edge("adversary_base", "adversary_base", "antagonist_base", 1.0, ego_agent),
        RPO_Edge("adversary_base", "adversary_base", "victim_base", -1.0, ego_agent),
        RPO_Edge("victim_base", "victim_base", "adversary_base", 1.0, alter_agent),
        RPO_Edge("antagonist_base", "antagonist_base", "adversary_base", 1.0, alter_agent),
    ]
    manipulator_objective = [
        RPO_Edge(
            "dummy_manipulator",
            "victim_base",
            "adversary_base",
            0.0,
            ego_agent,
            shaped_base="adversary_base",
        ),
    ]
    return bases, manipulators, base_objective, manipulator_objective
