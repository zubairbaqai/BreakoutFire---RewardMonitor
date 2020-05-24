import gym
from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


import numpy as np
import os
import itertools

import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.tf_util import get_session
from baselines.common import set_global_seeds
from monitoring_rewards.core import TraceStep
from monitoring_rewards.multi_reward_monitor import MultiRewardMonitor
from monitoring_rewards.monitoring_specification import MonitoringSpecification
from monitoring_rewards.reward_monitor import RewardMonitor
import os
import tempfile
import time

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import sys

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
import tensorflow.contrib.layers as layers
from functools import reduce




from gym_breakout_pygame.wrappers.normal_space import BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace


from flloat.parser.ltlf import LTLfParser

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")
        path="./model.pkl"

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.
    Parameters
    ----------
    path: str
        path to the act function pickle
    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

old_state=None
def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=3000,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=3000,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
    """Train a deepq model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.
    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)


    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=0.99,
        double_q=False
        #grad_norm_clipping=10,
        # param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(10000),
                                 initial_p=1.0,
                                 final_p=0.02)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()


    old_state = None





    formula_LTLf_1 = "!d U(g)"
    monitoring_RightToLeft = MonitoringSpecification(
        ltlf_formula=formula_LTLf_1,
        r=0,
        c=-0.01,
        s=10,
        f=-10
    )

    formula_LTLf_2 = "F(G(bb)) "  # break brick
    monitoring_BreakBrick = MonitoringSpecification(
        ltlf_formula=formula_LTLf_2,
        r=10,
        c=-0.01,
        s=10,
        f=0
    )

    monitoring_specifications = [monitoring_BreakBrick, monitoring_RightToLeft]




    def RightToLeftConversion(observation) -> TraceStep:

        done=False
        global old_state
        if arrays_equal(observation[-9:], np.zeros((len(observation[-9:])))):  ### Checking if all Bricks are broken
            # print('goal reached')
            goal = True  # all bricks are broken
            done = True
        else:
            goal = False

        dead = False
        if done and not goal:
            dead = True


        order = check_ordered(observation[-9:])
        if not order:
            # print('wrong order', state[5:])
            dead=True
            done = True

        if old_state is not None:  # if not the first state
            if not arrays_equal(old_state[-9:], observation[-9:]):
                brick_broken = True
                # check_ordered(state[-9:])
                # print(' a brick is broken')
            else:
                brick_broken = False
        else:
            brick_broken = False




        dictionary={'g': goal, 'd': dead, 'o': order, 'bb':brick_broken}
        #print(dictionary)
        return dictionary

    multi_monitor = MultiRewardMonitor(
        monitoring_specifications=monitoring_specifications,
        obs_to_trace_step=RightToLeftConversion
    )


    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
            # initialize
    done = False
    #monitor.get_reward(None, False) # add first state in trace
        

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        episodeCounter=0
        num_episodes=0
        for t in itertools.count():
            
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            #print(action)
            #print(action)
            new_obs, rew, done, _ = env.step(action)

            done=False
            #done=False ## FOR FIRE ONLY

            #print(new_obs)

            #new_obs.append()

            start_time = time.time()
            rew, is_perm = multi_monitor(new_obs)
            #print("--- %s seconds ---" % (time.time() - start_time))
            old_state=new_obs
            #print(rew)


            done=done or is_perm



            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew


            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if episodeCounter % 100 == 0 or episodeCounter<1:
                # Show off the result
                #print("coming here Again and Again")
                env.render()


            if done:
                episodeCounter+=1
                num_episodes+=1
                obs = env.reset()
                old_state=None
                episode_rewards.append(0)



                multi_monitor.reset()
                #monitor.get_reward(None, False)




            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(64)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("currentEpisodeReward", episode_rewards[-1])
                logger.record_tabular("mean 100 episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    act.save_act()
                    #save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        # if model_saved:
        #     if print_freq is not None:
        #         logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
        #     load_variables(model_file)

    return act



def arrays_equal(a, b):
    
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a,b):
        if ai != bi:
            return False
        
    return True

def check_ordered(bricks):
    

    zeros_index = np.where(bricks==0)[0]


    if len(zeros_index) == 0:
        return True
    zeros_index=zeros_index[0]
    for i in range(0, zeros_index):
        if(bricks[i]==0):
            print("Returning Because of Behind")
            print(bricks)
            #time.sleep(1)
            return False

    for i in range(zeros_index,len(bricks)):
        if(bricks[i]==1):
            print("Returning Because of After")
            print(bricks)
            #time.sleep(1)
            return False




    print(bricks)
    #time.sleep(2)
    return True












def model(inpt,  reuse=False):
    """This model takes as input an observation and returns values of all actions."""

    out = inpt
    out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)

    out = layers.fully_connected(out, num_outputs=4, activation_fn=None)
    return out

#env = gym.make("CartPole-v0")
env=BreakoutNMultiDiscrete()
def main():
    print('-*-*-*- train worker -*-*-*-')

    
    print(env.action_space.sample())
    # print(env.observation_space.high)
    # print(env.observation_space.low)


    
    #model = deepq.models.mlp([64])



    act = learn(
        env,
        network=model,
        lr=1e-3,
        max_timesteps=100000,
        checkpoint_freq=1000,
        buffer_size=50000   ,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        load_state="cartpole_model.pkl"
    )

    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()