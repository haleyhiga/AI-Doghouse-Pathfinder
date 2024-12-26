#!/usr/bin/env python3

import gymnasium as gym
import argparse
import logging
import sys

def create_environment(render_mode, seed, max_episode_steps):
    env = gym.make('Dog-v0', render_mode=render_mode)
    if seed:
        env.reset(seed=seed)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    return env

def destroy_environment(env):
    env.close()
    return

def run_one_episode(env, agent):
    observation, info = env.reset()
    agent.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not (terminated or truncated):
        action = agent.agent_function(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return total_reward

def run_many_episodes(env, episode_count, agent):
    reward_sum = 0
    for i in range(episode_count):
        reward_sum += run_one_episode(env, agent)
    destroy_environment(env)
    reward = reward_sum / episode_count
    return reward

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Run Doggy Environment')
    parser.add_argument(
        "--episode-count",
        "-c",
        type=int, 
        help="number of episodes to run",
        default=1
    )
    parser.add_argument(
        "--max-episode-steps",
        "-s",
        type=int, 
        help="maximum number of episode steps (defaults to environment default)",
        default=0
    )
    parser.add_argument(
        "--logging-level",
        "-l",
        type=str,
        help="logging level: warn, info, debug",
        choices=("warn", "info", "debug"),
        default="warn",
    )
    parser.add_argument(
        "--seed",
        type=int, 
        help="seed for the environment's PRNG",
        default=0
    )
    parser.add_argument(
        "--render-mode",
        "-r",
        type=str,
        help="display style (render mode): human, none",
        choices=("human", "ansi","none"),
        default="human",
    )
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        help="agent function: random, agent2",
        choices=("random", "agent2"),
        default="random",
    )

    my_args = parser.parse_args(argv[1:])
    if my_args.logging_level == "warn":
        my_args.logging_level = logging.WARN
    elif my_args.logging_level == "info":
        my_args.logging_level = logging.INFO
    elif my_args.logging_level == "debug":
        my_args.logging_level = logging.DEBUG

    if my_args.render_mode == "none":
        my_args.render_mode = None
    return my_args

from agent_random import AgentRandom
from agent2 import Agent2

def select_agent(agent_name):
    if agent_name == "random":
        agent_function = AgentRandom()
    elif agent_name == "agent2":
        agent_function = Agent2()

    else:
        raise Exception(f"unknown agent name: {agent_name}")
    return agent_function

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=my_args.logging_level)

    env = create_environment(my_args.render_mode, my_args.seed, my_args.max_episode_steps)
    agent = select_agent(my_args.agent)
    reward = run_many_episodes(env, my_args.episode_count, agent)
    print(f"Average Reward: {reward}")
    return

if __name__ == "__main__":
    main(sys.argv)
