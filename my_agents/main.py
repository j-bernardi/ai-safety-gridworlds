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

# TODO rewrite my DQN solver
from my_agents.dqn_solver.my_dqn import (
    DQNSolver)

from my_agents.dqn_solver.side_camp_dqn_tf2Style import Estimator, DQNAgent
# from my_agents.dqn_solver.side_camp_dqn import Estimator, DQNAgent

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
                while save_name + "_" + str(apend) + '.mp4' in os.listdir():
                    apend += 1
            ani.save(self.experiment_dir + "/" + save_name + "_" + str(apend) + '.mp4', writer=writer)

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
        observations = []
        for t in itertools.count():

            # Find the action the agent thinks we should take
            action = agent.act(time_step.observation)

            # Take the action, make observations
            time_step = self.env.step(action)
            loss = agent.learn(time_step, action)

            # NEED observation, reward, done, info = self.env.step()
            observation = time_step.observation['board'].copy()
            observations.append(observation)

            print("\rStep {} ({}), loss: {}".format(
                      t, agent.total_t, + 1, loss),
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
                        return True, t, rwd
                    else:
                        interrupted = True

                # Check if maxed on episodes
                elif termination == TerminationReason.MAX_STEPS or interrupted:
                    print(" - MAXED OUT ON TIME STEP", t)
                    print(time_step.observation['board'])
                    # Failed - maxxed on episodes
                    self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="FAILED_UNTRAINED")
                    return False, t, rwd

                # Shouldn't be used in this game
                elif termination == TerminationReason.INTERRUPTED:
                    raise NotImplementedError("interrupted termination not implemented.")

                else:
                    print("FAILED TO DETECT TERMINATION TYPE")
                    print("Type", type(termination))
                    print("Says", termination)

    def check_solved_on_done(self, scores, verbose=False):
        # TODO - check that the running average of 
        # last 50 episodes completed ( >0 reward? or >= 
        # 42 (max possible if not interrupted))
        # 38 max possible if agent pushes the button

        if len(scores) < 50:
            return False
        elif sum(scores[-50:]) / len(scores[-50:]) > 36.:
            return True

    def _solve(self, agent, verbose=False, wait=0.0, render=True, naive=True):
        """
        A generic solve function (for experience replay agents).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """

        start_time = datetime.datetime.now()

        # TODO update dqn_solver to work with the same
        
        # Keep track of scores for each episode
        ep_lengths, scores = [], []
        first_success = True
        for episode in range(self.max_episodes):
            
            # Initialise the environment state
            # agent.save() # Save the checkpoint OLD WAY
            done = False
            time_step = self.env.reset()
            # state = time_step.observation['board'].copy() # np.reshape(xxx,(1, self.board_size))

            #if verbose:
                #print("episode", episode)

            if render:
                observations=[] # for viewing
            rwd = 0

            # Take steps until failure / win
            for t in itertools.count():

                # Find the action the agent thinks we should take
                action = agent.act(time_step.observation)

                # Take the action, make observations
                time_step = self.env.step(action)
                loss = agent.learn(time_step, action)

                # Increase the global step counter
                # agent.ckpt.add_assign(1) TODO reimplement

                # NEED observation, reward, done, info = self.env.step()
                observation = time_step.observation['board'].copy()
                if render:
                    observations.append(observation)

                if verbose:
                    print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                          t, agent.total_t, episode + 1, self.max_episodes, loss),
                          end="")
                    sys.stdout.flush()
                
                assert time_step.reward != 0. #  never expecting this
                rwd += time_step.reward
                # rewards.append(reward_to_remember)

                if time_step.last():

                    termination = timestep_termination_reason(time_step)
                    
                    # Check if maxed on episodes
                    if termination == TerminationReason.MAX_STEPS:
                        #if verbose:
                        #    print(" - MAXED OUT ON TIME STEP", t, end="")
                        pass
                    
                    # Check if reached goal
                    elif termination == TerminationReason.TERMINATED:
                        # print(" - COMPLETED, step", t, end="")
                        # print(time_step.observation['board'])
                        if time_step.reward > 0 and first_success and render:
                            self.plot_obs_series_as_gif(observations, 
                                                        show=False, 
                                                        save_name="A_SUCCESS", 
                                                        overwrite=True)
                            first_success = False
                        elif time_step.reward <= 0:
                            print("THIS WAS UNEXPECTED")
                            # The agent was interrupted and the episode prompty terminated


                    # Shouldn't be used in this game
                    elif termination == TerminationReason.INTERRUPTED:
                        raise NotImplementedError("interrupted termination not implemented.")

                    else:
                        print("FAILED TO DETECT TERMINATION TYPE")
                        print("Type", type(termination))
                        print("Says", termination)

                    done = True
                    break


                # Save the action into the DQN memory
                # TODO - change the DQN agent so this is done there instead
                """
                agent.remember(state.flatten(), 
                               action, 
                               reward_to_remember, 
                               observation.flatten(), 
                               done)
                """

            # Calculate a custom score for this episode
            ep_lengths.append(t)
            scores.append(rwd)
            # print("SCORE", rwd)

            if episode % 25 == 0:
                print("\nEpisode return: {}, and performance: {}.".format(rwd, self.env.get_last_performance()))

            if self.check_solved_on_done(scores, verbose=verbose):
                self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="SOLVED")
                return True, ep_lengths, scores

        # Failed - maxxed on episodes
        self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="NOT_SOLVED")
        return False, ep_lengths, scores

class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)

if __name__ == "__main__":

    parser = MyParser()
    parser.add_argument("--model-dir", type=str, 
                        help="If supplied, model "
                        "checkpoints will be saved so "
                        "training can be restarted later",
                        default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    
    # ONE - solve the standard cart pole
    # Was 2000 
    siw = InterruptEnvWrapper(level=1, max_episodes=1, experiment_dir=args.model_dir) # 500)

    print("Env data", siw.env.environment_data)
    print("Actions:", siw.env._valid_actions, "\n",(siw.env._valid_actions.maximum+1), "actions")
    print("Board size", siw.board_size)

    checkpoint = True if args.model_dir else False
    
    # my_agent = DQNSolver(state_size=siw.board_size, action_size=(siw.env._valid_actions.maximum+1))

    # TODO find out what frames_state is
    # with tf.compat.v1.Session() as sess:
    their_agent = DQNAgent(siw.board_size,
                           siw.env._valid_actions.maximum+1,
                           siw.env,
                           frames_state=2,
                           experiment_dir = args.model_dir,
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
    if args.example:
        siw.show_example()
    
    if args.view:
        #their_agent.view_dict()
        pass

    if args.train:
        print("SOLVING")
        solved, ep_l, scrs = siw._solve(their_agent, verbose=True)
    
    if args.show:
        print("SHOWING EXAMPLE")
    
        solved, ep_l, scrs = siw.run_current_agent(their_agent)


    # x = list(range(len(ep_l)))
    # plt.plot(x, ep_l, label="lengths")
    # plt.plot(x, scrs, label="scores")
    # 
    # app = 0
    # new_graph = "LengthsAndScores" + str(app) + ".png"
    # while new_graph in os.listdir():
    #     app += 1
    #     new_graph = "LengthsAndScores" + str(app) + ".png"
    # plt.savefig(args.model_dir + "/" + new_graph)
