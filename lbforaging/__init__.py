from typing import Optional
from itertools import product
from gymnasium import register


# registering all these envs takes forever, so use a reduced version
sizes = range(4, 12)
players = range(2, 5)
foods = range(1, 5)

max_food_level = [None]  # [None, 1]
coop = [True, False]
partial_obs = [True, False]
pens = [False]  # [True, False]


def get_env_id(
    s: int, p: int, f: int, c: bool, po: bool, pen: bool, mfl: Optional[int] = None
):
    """_summary_

    Parameters
    ----------
    s : int
        env size
    p : int
        number of players / agents
    f : int
        amount of food to spawn
    c : bool
        coop bool, if True then food levels are constrained to force cooperation
    po : bool
        partial observation option, if True then each agent only sees part of the map
    pen : bool
        if True, adds a reward penalty if the agents attempt to pick up the food but fail due to not having enough total levels
    mfl : Optional[int], optional
        max food level, by default None. Only used to set food to level 1 (see registration options above) so agents can pick up food without coordinating actions with others

    Returns
    -------
    str
        name of the specific version of the foraging environment to be run
    """

    mfl_str = "-ind" if mfl else ""

    env_id = "Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}-v3".format(
        s,
        p,
        f,
        "-coop" if c else "",
        "-2s" if po else "",
        mfl_str,
        "-pen" if pen else "",
    )

    # does not currently support the grid env
    # this way of setting up the envs is kinda dumb lol
    # grid env can't have penalty for some reason?
    # id="Foraging-grid{4}-{0}x{0}-{1}p-{2}f{3}{5}-v3".format(
    #     s,
    #     p,
    #     f,
    #     "-coop" if c else "",
    #     "" if sight == s else f"-{sight}s",
    #     "-ind" if mfl else "")

    return env_id


def register_envs():
    for s, p, f, mfl, c, po, pen in product(
        sizes, players, foods, max_food_level, coop, partial_obs, pens
    ):
        env_id = get_env_id(s=s, p=p, f=f, c=c, po=po, mfl=mfl, pen=pen)
        register(
            id=env_id,
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p,
                "min_player_level": 1,
                "max_player_level": 2,
                "field_size": (s, s),
                "min_food_level": 1,
                "max_food_level": mfl,
                "max_num_food": f,
                "sight": 2 if po else s,
                "max_episode_steps": 50,
                "force_coop": c,
                "grid_observation": False,
                "penalty": 0.1 if pen else 0.0,
            },
        )


def register_grid_envs():
    for s, p, f, mfl, c in product(sizes, players, foods, max_food_level, coop):
        for sight in range(1, s + 1):
            register(
                id="Foraging-grid{4}-{0}x{0}-{1}p-{2}f{3}{5}-v3".format(
                    s,
                    p,
                    f,
                    "-coop" if c else "",
                    "" if sight == s else f"-{sight}s",
                    "-ind" if mfl else "",
                ),
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": p,
                    "min_player_level": 1,
                    "max_player_level": 2,
                    "field_size": (s, s),
                    "min_food_level": 1,
                    "max_food_level": mfl,
                    "max_num_food": f,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "grid_observation": True,
                },
            )
