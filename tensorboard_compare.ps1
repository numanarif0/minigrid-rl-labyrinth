$root = Split-Path -Parent $MyInvocation.MyCommand.Path
tensorboard --logdir_spec "PPO:$root\ppo_tensorboard,DQN:$root\dqn_tensorboard"
