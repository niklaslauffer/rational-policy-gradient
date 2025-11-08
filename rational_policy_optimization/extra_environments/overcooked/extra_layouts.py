from jaxmarl.environments.overcooked import layout_grid_to_dict

# Example of layout provided as a grid
hard_forced_coord = """
WWWWW
O W P
B W X
O W P
BAWAX
O W P
B W X
O W P
WWWWW
"""
medium_forced_coord = """
WBWXW
O W P
OAWAP
O W P
WBWXW
"""
dual_forced_coord = """
WBWXW
O W P
WAWAW
O W P
WBWXW
"""

overcooked_layouts = {
    "hard_forced_coord": layout_grid_to_dict(hard_forced_coord),
    "medium_forced_coord": layout_grid_to_dict(medium_forced_coord),
    "dual_forced_coord": layout_grid_to_dict(dual_forced_coord),
}
