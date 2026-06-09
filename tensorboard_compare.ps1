$root = Split-Path -Parent $MyInvocation.MyCommand.Path
tensorboard --logdir_spec "PPO_V1:$root\ppo_tensorboard,PPO_V2:$root\ppo_tensorboard_v2,DQN_V2:$root\dqn_tensorboard_v2,DQN_V3:$root\dqn_tensorboard_v3"
