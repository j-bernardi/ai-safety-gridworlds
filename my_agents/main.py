import os, sys, datetime, itertools, argparse
import numpy as np

from random import randrange

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tensorflow as tf

from pycolab.rendering import (
    BaseObservationRenderer,
    Observation)

import ai_safety_gridworlds
from ai_safety_gridworlds.environments.safe_interruptibility import (
    SafeInterruptibilityEnvironment, 
    InterruptionPolicyWrapperDrape, 
    ButtonDrape, 
    AgentSprite,
    Actions,
    GAME_ART)
from ai_safety_gridworlds.environments.shared.safety_game import (
    DEFAULT_ACTION_SET,
    timestep_termination_reason)

from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason

matplotlib.use("Agg")
# Set your ffmpeg path
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' 


class InterruptEnvWrapper(object):
    
    def __init__(self, level=1, max_episodes=500, naive=True, experiment_dir=None):
        self.env = SafeInterruptibilityEnvironment()
        self.obs_spec = self.env._compute_observation_spec()
        self.board_size = self.env._observation_spec['board'].shape

        # self.renderer = BaseObservationRenderer(*self.obs_spec['board'].shape, tuple(self.env._value_mapping.keys()))
        self.level = level

        self.max_episodes = max_episodes

        self.experiment_dir = experiment_dir if experiment_dir else os.get_cwd()

    def plot_obs_series_as_gif(self, observation_series, show=False, save_name=None, overwrite=False):
        fig = plt.figure()
        ims = []
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        for obs in observation_series:
            ims.append([plt.imshow(obs, animated=True)])
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)

        if save_name is not None:
            
            apend = 0

            if not overwrite:
                while save_name + "_" + str(apend) + '.mp4' in os.listdir(self.experiment_dir):
                    apend += 1

            saving_to = self.experiment_dir + "/" + save_name + "_" + str(apend) + '.mp4'
            print("Saving gif to", saving_to)
            ani.save(saving_to, writer=writer)

    def show_example(self, steps=99, plot=True, save=True):
        """Show a quick view of the environemt, 
        without trying to solve.
        """
        if steps > 999:
            print("I'm not sure you want that many")
            sys.exit()
        
        time_step = self.env.reset()
        
        print("INIT")
        print(time_step.observation['board'])
        print("Step type: first {}, mid {}, last {}".format(time_step.first(), time_step.mid(), time_step.last()))
        print("Reward {}, discount {}".format(time_step.reward, time_step.discount))
        print("Observation type: {}".format(type(time_step.observation)))

        if plot:
            print("Initial state:")
            in_fig = plt.figure()
            plt.title("Initial state")
            plt.imshow(time_step.observation['board'])
            plt.axis('off')
            print("Saving init state fig")
            in_fig.savefig(self.experiment_dir +  '/randInitialState.png')

        print("Let's act..")
        observations = []
        for i in range(steps):
            time_step = self.env.step(randrange(self.env._valid_actions.maximum+1))
            obs = time_step.observation['board']

            # print("Step type: first {}, mid {}, last {}".format(time_step.first(), time_step.mid(), time_step.last()))
            # print("Reward {}, discount {}".format(time_step.reward, time_step.discount))
            # print("Observation type: {}".format(type(time_step.observation)))

            observations.append(obs.copy())

        if plot:
            save_name = 'randGame' if save else None
            self.plot_obs_series_as_gif(observations, show=True, save_name=save_name, overwrite=False)

        print("Obs")
        print(observations[-1])

        plt.show()

    def run_current_agent(self, agent, steps=99, plot=True, save=True):
        # TODO - use the current state of the agent to see what's going on
        time_step = self.env.reset()
        rwd = 0
        observations = [time_step.observation['board'].copy()]
        for t in itertools.count():

            # Find the action the agent thinks we should take
            action = agent.act(time_step.observation)

            # Take the action, make observations
            time_step = self.env.step(action)
            # loss = agent.learn(time_step, action)

            # NEED observation, reward, done, info = self.env.step()
            observation = time_step.observation['board'].copy()
            observations.append(observation)

            print("\rStep {} ({}): {}".format(
                      t, agent.total_t, + 1),
                      end="")
            sys.stdout.flush()
            
            rwd += time_step.reward

            if time_step.last():

                termination = timestep_termination_reason(time_step)
                interrupted = False
                
                # Check if reached goal
                if termination == TerminationReason.TERMINATED:
                    # TODO - establish if hit interrupt button or success
                    if time_step.reward > 0: 
                        print(" - COMPLETED, step", t)
                        print(time_step.observation['board'])
                        self.plot_obs_series_as_gif(observations, show=False, 
                                                overwrite=False,
                                                save_name="SUCCEEDED_ON_SHOW")
                        return True, t, rwd
                    else:
                        interrupted = True

                # Check if maxed on episodes
                elif termination == TerminationReason.MAX_STEPS or interrupted:
                    print(" - MAXED OUT ON TIME STEP", t)
                    print(time_step.observation['board'])
                    # Failed - maxxed on episodes
                    self.plot_obs_series_as_gif(observations, show=False, 
                                                overwrite=False,
                                                save_name="FAILED_ON_SHOW")
                    return False, t, rwd

                # Shouldn't be used in this game
                elif termination == TerminationReason.INTERRUPTED:
                    raise NotImplementedError("interrupted termination not implemented.")

                else:
                    print("FAILED TO DETECT TERMINATION TYPE")
                    print("Type", type(termination))
                    print("Says", termination)

    def check_solved_on_done(self, scores, average_over, target, verbose=False):
        # TODO - check that the running average of 
        # last 50 episodes completed ( >0 reward? or >= 
        # 42 (max possible if not interrupted))
        # 38 max possible if agent pushes the button

        if len(scores) < average_over:
            return False, sum(scores) / len(scores)
        else:
            scr = sum(scores[-average_over:]) / len(scores[-average_over:])
            return (scr > target), scr

    def _solve(self, agent, verbose=False, wait=0.0, render=True, naive=True):
        """
        A generic solve function (for experience replay agents).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """

        start_time = datetime.datetime.now()

        # TODO update dqn_solver to work with the same
        
        # Keep track of scores for each episode
        # ep_lengths, scores, losses = [], [], []
        first_success = True
        for episode in range(self.max_episodes):

            # agent.save()
            
            # Initialise the environment state
            done = False
            time_step = self.env.reset()
            rwd = 0
            if render:
                observations=[time_step.observation['board'].copy()]
            
            # Take steps until failure / win
            for t in itertools.count():

                action = agent.act(time_step.observation)
                time_step = self.env.step(action)
                loss = agent.learn(time_step, action)
                
                if render:
                    observation = time_step.observation['board'].copy()
                    observations.append(observation)

                if verbose:
                    print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                          t, agent.total_t, episode + 1, self.max_episodes, loss),
                          end="")
                    sys.stdout.flush()

                # Do some return checking
                assert (time_step.reward == -1. 
                    or time_step.reward == 49.), str(time_step.reward)
                rwd += time_step.reward

                if time_step.last():
                    # Assert that interrupt doesn't just stop us
                    assert (t == 99 and time_step.reward == -1.)\
                        or time_step.reward == 49.

                    termination = timestep_termination_reason(time_step)
                    
                    if termination == TerminationReason.MAX_STEPS:
                        assert t == 99

                    # Check if reached goal
                    elif termination == TerminationReason.TERMINATED:
                        # print(" - COMPLETED, step", t, end="")
                        # print(time_step.observation['board'])

                        if time_step.reward == 49 and first_success and render:
                            self.plot_obs_series_as_gif(observations, 
                                                        show=False, 
                                                        save_name="A_SUCCESS", 
                                                        overwrite=True)
                            first_success = False

                        elif time_step.reward != 49:
                            print("THIS WAS UNEXPECTED")
                            raise NotImplementedError(
                                "Terminated with wrong reward")
                            # The agent was interrupted and the episode prompty terminated


                    # Shouldn't be used in this game
                    elif termination == TerminationReason.INTERRUPTED:
                        raise NotImplementedError(
                            "Interrupted termination not implemented.")

                    else:
                        print("FAILED TO DETECT TERMINATION TYPE")
                        print("Type", type(termination))
                        print("Says", termination)

                    done = True
                    break

            # Calculate a custom score for this episode
            agent.ep_lengths.append(t)
            agent.scores.append(rwd)
            agent.losses.append(loss)

            solved, scr = self.check_solved_on_done(agent.scores, 100, 36., verbose=verbose)

            if episode % 25 == 0:
                print("\nEpisode return: {}, and performance: {}. SCORE {}".format(
                      rwd, self.env.get_last_performance(), scr))

            if solved:
                self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="SOLVED")
                if agent.solved_on:
                    agent.solved_on = min(agent.solved_on, episode)
                else:
                    agent.solved_on = episode

                elapsed = datetime.datetime.now() - start_time
                print("\nSOLVED")
                print("\nTIME ELAPSED", elapsed)
                agent.save()
                return True, agent.ep_lengths, agent.scores, agent.losses

        elapsed = datetime.datetime.now() - start_time
        print("\nTIME ELAPSED", elapsed)

        # Failed - maxxed on episodes
        self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="NOT_SOLVED")
        print("\nNOT SOLVED")
        agent.save()
        return False, agent.ep_lengths, agent.scores, agent.losses


class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)


def make_my_dqn_agent(siw, exp_dir, checkpoint):

    from my_agents.dqn_solver.my_dqn import (
        DQNSolver)

    my_agent = DQNSolver(siw.board_size,
                         siw.env._valid_actions.maximum+1,
                         siw.env,
                         frames_state=2,
                         experiment_dir = exp_dir,
                         replay_memory_size=10000,
                         replay_memory_init_size=500,
                         update_target_estimator_every=250,
                         discount_factor=0.99,
                         epsilon_start=1.0,
                         epsilon_end=0.1,
                         epsilon_decay_steps=50000,
                         batch_size=8,
                         checkpoint=checkpoint)
                           

    return my_agent

def make_double_dqn_agent(siw, exp_dir, checkpoint):

    from my_agents.dqn_solver.side_camp_dqn_tf2Style import (
        DQNAgent)
    
    their_agent = DQNAgent(siw.board_size,
                           siw.env._valid_actions.maximum+1,
                           siw.env,
                           frames_state=2,
                           experiment_dir = exp_dir,
                           replay_memory_size=10000,
                           replay_memory_init_size=500,
                           update_target_estimator_every=250,
                           discount_factor=0.99,
                           epsilon_start=1.0,
                           epsilon_end=0.1, # TODO is this a bit high? 10% of time random? Guess it's training not complete
                           epsilon_decay_steps=50000,
                           batch_size=8,
                           checkpoint=checkpoint
                           )


    return their_agent

def make_side_camp_double_dqn_agent(siw, exp_dir, checkpoint, sess):

    from my_agents.dqn_solver.side_camp_dqn import DQNAgent

    original_agent = DQNAgent(sess,
                              siw.board_size,
                              siw.env._valid_actions.maximum+1,
                              siw.env,
                              frames_state=2,
                              experiment_dir=exp_dir,
                              replay_memory_size=10000,
                              replay_memory_init_size=500,
                              update_target_estimator_every=250,
                              discount_factor=0.99,
                              epsilon_start=1.0,
                              epsilon_end=0.1, # TODO is this a bit high? 10% of time random? Guess it's training not complete
                              epsilon_decay_steps=50000,
                              batch_size=8,
                              checkpoint=checkpoint
                           )
    return original_agent


def do_train(siw, agent):

    print("SOLVING")

    # agent.load() loads on creation now

    print("CURRENT EXP STATE")
    agent.display_param_dict()

    solved, ep_l, scrs, losses = siw._solve(agent, verbose=True)

    agent.save() # to capture solved_on

    return solved, ep_l, scrs, losses

def do_show(siw, agent):
    agent.load()
    agent.display_param_dict()
    print("SHOWING EXAMPLE")
    solved2, ep_l2, scrs2 = siw.run_current_agent(agent)
    return solved2, ep_l2, scrs2


if __name__ == "__main__":

    parser = MyParser()
    parser.add_argument("--model-suffix", type=str, 
                        help="If supplied, model "
                        "checkpoints will be saved so "
                        "training can be restarted later",
                        default=None)
    parser.add_argument("--train", dest="train", 
                        type=int, default=0, 
                        help="number of episodes to train")
    parser.add_argument("--show", action="store_true")
    # parser.add_argument("--view", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--model", type=str, default="default")
    args = parser.parse_args()

    exp_dir = args.model + "_" + args.model_suffix

    siw = InterruptEnvWrapper(level=1, max_episodes=args.train, experiment_dir=exp_dir) # 500)

    print("\nEnv data", siw.env.environment_data)
    print("Actions:", siw.env._valid_actions,(siw.env._valid_actions.maximum+1), "actions")
    print("Board size", siw.board_size)

    checkpoint = True if args.model_suffix else False

    if args.example:
        siw.show_example()

    if args.model == "default":
        agent = make_double_dqn_agent(siw, exp_dir, checkpoint)
    elif args.model == "original":
        agent = make_my_dqn_agent(siw, exp_dir, checkpoint)
    elif args.model == "side_camp_dqn":
        # sess = tf.compat.v1.Session()
        with tf.compat.v1.Session() as sess:
            agent = make_side_camp_double_dqn_agent(siw, exp_dir, checkpoint, sess)
            if args.train > 0:
                solved, ep_l, scrs, losses = do_train(siw, agent)
            if args.show:
                solved2, ep_l2, scrs2 = do_show(siw, agent)


    if args.train > 0 and args.model != "side_camp_dqn":

        solved, ep_l, scrs, losses = do_train(siw, agent)

    if args.plot:
        print("TODO - get params dict")
        raise NotImplementedError("TODO")
        x = list(range(len(ep_l)))
        
        def pltt(val, ttl):
            plt.figure()
            plt.plot(x, val)
            plt.title(ttl)
            app = 0
            new_graph = ttl + str(app) + ".png"
            while new_graph in os.listdir():
                app += 1
                new_graph = ttl + str(app) + ".png"
            
            plt.savefig(exp_dir + "/" + new_graph)

        pltt(ep_l, "lengths")
        pltt(scrs, "scores")
        pltt(losses, "losses")

    if args.show and args.model != "side_camp_dqn":
        solved2, ep_l2, scrs2 = do_show(siw, agent)
