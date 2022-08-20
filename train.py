import argparse
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import Agent
from utils import moving_average, adjust_reward_and_memorize,\
    visualize_cnn, make_env

from tensorflow.python.ops.numpy_ops import np_config

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
np_config.enable_numpy_behavior()


def parse_args():
    parser = argparse.ArgumentParser("DQN experiment for Subway Surfers")
    # Environment
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e5), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-frames", type=int, default=4,
                        help="total number of frames to stack as observation to agent")
    parser.add_argument("--num-episodes", type=int, default=2000,
                        help="total number of episodes to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=1,
                        help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=300,
                        help="number of iterations between every target network update")
    # Bells and whistles
    parser.add_argument("--prioritized", type=bool, default=True,
                        help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.5,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4
                        , help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="weights/dqn",
                        help="directory in which training state and model should be saved.")
    parser.add_argument("--save-freq", type=int, default=15,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--load-on-start", type=bool, default=True,
                        help="if true and model was previously saved then training will be resumed")

    return parser.parse_args()


def learn(args):
    # making the environment with n frame stacked
    env = make_env(args.num_frames)
    # creating ddqn agent
    agent = Agent(env=env, args=args)

    rewards = []
    total_steps = 0
    for episode_n in range(args.num_episodes):
        print(f"Starting episode {episode_n + 1}")
        state = env.reset()
        state = np.array(state)
        done = False
        exp_holder = []
        episode_reward = 0
        episode_steps = 0
        while not done:
            action = agent.make_action(state, advantage=True)
            nstate, reward, done, _ = env.step(action)
            nstate = np.array(nstate)
            # storing SARS info in temporary list
            exp_holder.append([state, action, reward, done, nstate])

            state = nstate
            episode_reward += reward

            if done:
                adjust_reward_and_memorize(exp_holder, agent)

            if episode_n % agent.target_update_freq == 0:
                agent.update_target()

            episode_steps += 1
            total_steps += 1

        if episode_n + 1 > args.start_ep:
            n_replay = max(episode_steps // agent.batch_size, 1)
            loss = agent.replay(n_replay)
            print(f"loss: {loss}")

        if total_steps % agent.target_update_freq == 0:
            agent.update_target()

        if total_steps % 100 == 0:
            print(f"{total_steps} steps taken")

        if episode_n % args.save_freq == 0:
            agent.save_weights(args.save_dir)

        print(f"Finished episode {episode_n + 1} with reward: {episode_reward}")
        rewards.append(episode_reward)
        print(f"highest_run : {max(rewards)}")
        plt.plot(rewards)
        plt.show()
        ave_reward = moving_average(rewards, n=8)
        plt.plot(ave_reward)
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    learn(args)
