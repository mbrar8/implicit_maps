#!/usr/bin/env python3

import numpy as np

class Evaluator:
    def __init__(self, network, env, max_steps=501):
        self.network = network
        self.env = env
        self.max_steps = max_steps

    def evaluate(self, epoch):
        avg_reward = 0.0
        eval_episodes=10
        for _ in range(eval_episodes):
            self.env.get_logger().info(f"Evaluating Episode {_}")
            count = 0
            state = self.env.reset()
            done = False
            while not done and count < self.max_steps:
                action = self.network.get_action(np.array(state))
                a_in = [(action[0] + 1) / 2, action[1]]
                state, reward, done, _ = self.env.step(a_in)
                avg_reward += reward
                count += 1
        avg_reward /= eval_episodes
        self.env.get_logger().info("..............................................")
        self.env.get_logger().info(
            "Evaluation Info - Epoch: %i, Avg Reward: %f"
            % (epoch, avg_reward)
        )
        self.env.get_logger().info("..............................................")
        return avg_reward

def main():
    print("Evaluator Class for DREAM Project")
    print("Evaluating the performance of the trained network on the environment")
    print("..............................................")
    print("Example Usage:")
    print("1. Create an instance of the Evaluator class")
    print("2. Call the evaluate method with the epoch number")
    print("..............................................")
    print("Example:")
    print("network = ActorCriticNetwork()")
    print("env = MultiAgentEnv()")
    print("evaluator = Evaluator(network, env)")
    print("evaluator.evaluate(10)")
    print("..............................................")

if __name__ == "__main__":
    main()