import gym
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import deepq
from gym_breakout_pygame.wrappers.normal_space import BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace
import time







def main():
    print('-*-*-*- enjoy worker -*-*-*-')
    # tf.graph().as_default()
    # tf.reset_default_graph()
    #env = gym.make("CartPole-v0")
    env=BreakoutNMultiDiscrete()
    act = deepq.load_act("model.pkl")

    max_episodes = 5

    while max_episodes > 0:
        obs, done = env.reset(), False
        #print(obs)
        episode_rew = 0
        while not done:
            env.render()
            time.sleep(0.5)
            obs, rew, done, _ = env.step(act(obs[None])[0])
            print(rew)
            episode_rew += rew
        print("Episode reward", episode_rew)
        max_episodes = max_episodes - 1


if __name__ == '__main__':
    main()