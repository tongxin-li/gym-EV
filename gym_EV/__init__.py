from gym.envs.registration import register

register(
    id='EV-v0',
    entry_point='gym_EV.envs:EVEnv',
)