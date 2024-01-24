from .gymenv_core import gym_
from . import gymenv_wrapped

GymEnvID_stage_3_cont_v1 = "turtlebot3_stage_3_cont-v1"
gym_.register(
    id=GymEnvID_stage_3_cont_v1,
    entry_point=f"{gymenv_wrapped.__name__}:{gymenv_wrapped.Env_cont.__name__}",
)

GymEnvID_stage_3_disc_v1 = "turtlebot3_stage_3_disc-v1"
gym_.register(
    id=GymEnvID_stage_3_disc_v1,
    entry_point=f"{gymenv_wrapped.__name__}:{gymenv_wrapped.Env_disc.__name__}",
)

REGISTERD_ENV_IDS = [
    GymEnvID_stage_3_disc_v1,
    GymEnvID_stage_3_cont_v1,
]


del gymenv_wrapped

if __name__ == "__main__":
    raise Exception(f"不允许直接运行 {__file__}")
