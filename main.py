import argparse

import gym
import mlflow
import torch
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformObservation
from torch.optim import Adam

from actor_critic import ActorCritic
from ppo import PPOAgent
from utils import NumpifyAction, TrajectoryBuffer

gym.logger.set_level(40)

parser = argparse.ArgumentParser(description='PyTorch PPO on CarRacing')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor (default: 0.99)')
parser.add_argument('--n_ppo_epochs', type=int, default=10,
                    help='Number of epochs for PPO optimization (default: 10)')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Clipping parameter for PPO (default: 0.1)')
parser.add_argument('--vf_coef', type=float, default=0.5,
                    help='Value function coefficient for PPO (default: 0.5)')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='Maximum value for gradient clipping (default: 0.5)')
parser.add_argument('--lam', type=float, default=0.95,
                    help='Lambda parameter for GAE (default: 0.95)')
# parser.add_argument('--seed', type=int, default=543, metavar='N',
#                     help='random seed (default: 543)')
parser.add_argument('--n_stack', type=int, default=4,
                    help='Number of frames for frame stack (default: 4)')
parser.add_argument('--n_episodes', type=int, default=1000,
                    help='Number of episodes (default: 1000)')
parser.add_argument('--n_steps', type=int, default=1000,
                    help='Number of steps per episode (default: 1000)')
parser.add_argument('--render', action='store_true',
                    help='Render the environment')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='interval between training status logs (default: 10)')
args = parser.parse_args()


def main():
    env = gym.make("CarRacing-v0", verbose=0)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 64)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=args.n_stack)
    env = NumpifyAction(env)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    traj_buf = TrajectoryBuffer(args.n_steps,
                                env.observation_space.shape,
                                env.action_space.shape,
                                args.gamma,
                                args.lam)

    actor_critic = ActorCritic(args.n_stack)
    optimizer = Adam(actor_critic.parameters(), lr=1e-3)
    agent = PPOAgent(env, actor_critic, optimizer, device, args.n_ppo_epochs, args.epsilon, args.vf_coef,
                     args.max_grad_norm)

    mlflow.log_params(vars(args))

    running_rew = 0

    for i_episode in range(args.n_episodes):

        obs = env.reset()
        ep_rew = 0.
        last_val = 0.

        for t in range(args.n_steps):
            act, act_log_prob, val = agent.act(obs)
            next_obs, rew, done, _ = env.step(act)

            ep_rew += rew

            traj_buf.store(obs, act, act_log_prob, val, rew)

            if args.render:
                env.render()

            obs = next_obs

            if done:
                break

            if t == args.n_steps - 1:  # in case ep didn't end naturally bootstrap value
                last_val = val

        running_rew = 0.05 * ep_rew + (1 - 0.05) * running_rew
        mlflow.log_metric('running_rew', running_rew)
        print(f'Episode {i_episode}\tLast reward: {ep_rew:.2f}\tAverage reward: {running_rew:.2f}')

        traj_buf.calculate_norm_advantage_and_return(last_val)

        data = traj_buf.retrieve()
        agent.learn(*data)

    env.close()


main()
if __name__ == '__main__':
    main()
