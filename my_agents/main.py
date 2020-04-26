import os, sys
import numpy as np

from random import randrange

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

from my_agents.dqn_solver.solver import (
    DQNSolver)

from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason

matplotlib.use("Agg")
# Set your ffmpeg path
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' 


class InterruptEnvWrapper(object):
    
    def __init__(self, level=1, max_episodes=500, naive=True):
        self.env = SafeInterruptibilityEnvironment()
        self.obs_spec = self.env._compute_observation_spec()
        self.board_size = self.env._observation_spec['board'].shape

        # self.renderer = BaseObservationRenderer(*self.obs_spec['board'].shape, tuple(self.env._value_mapping.keys()))
        self.level = level

        self.max_episodes = max_episodes

    @staticmethod
    def plot_obs_series_as_gif(observation_series, show=False, save_name=None, overwrite=False):
        fig = plt.figure()
        ims = []
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        for obs in observation_series:
            ims.append([plt.imshow(obs, animated=True)])
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)

        if save_name is not None:
            
            apend = ""

            if not overwrite:
                while save_name + str(apend) + '.mp4' in os.listdir():
                    apend = 1 if apend == "" else apend + 1
                apend = "_" + str(apend) 
            ani.save(save_name + apend + '.mp4', writer=writer)

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
            in_fig.savefig('randInitialState.png')

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
            self.plot_obs_series_as_gif(observations, show=True, save_name=save_name)

        print("Obs")
        print(observations[-1])

        plt.show()

    def check_solved_on_done(self, scores, verbose=False):
        # TODO - check that the running average of 
        # last 50 episodes completed ( >0 reward? or >= 
        # 42 (max possible if not interrupted))
        # 38 max possible if agent pushes the button

        if len(scores) < 50:
            return False
        elif sum(scores[-50:]) / len(scores[-50:]) > 36.:
            return True


    # TODO - atm I'm not sure I'm terminating whehn goal is reached...

    def _solve(self, agent, verbose=False, wait=0.0, render=True, naive=True):
        """
        A generic solve function (for experience replay agents).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """
        
        print("SOLVING")
        
        # Keep track of scores for each episode
        scores = []
        first_success = True
        for episode in range(self.max_episodes):
            
            # Initialise the environment state
            done, step = False, 0
            time_step = self.env.reset()
            state = time_step.observation['board'].copy() # np.reshape(xxx,(1, self.board_size))

            if verbose:
                print("episode", episode, end="")

            if render:
                observations=[] # for viewing
            rewards = []

            # Take steps until failure / win
            while not done:
                # print("STEP", step)
                # Find the action the agent thinks we should take
                action = agent.act(state.flatten())

                # Take the action, make observations
                time_step = self.env.step(action)
                termination = timestep_termination_reason(time_step)

                # NEED observation, reward, done, info = self.env.step()
                observation = time_step.observation['board'].copy()
                if render:
                    observations.append(observation)

                if time_step.reward is not None:
                    reward_to_remember = time_step.reward
                else:
                    reward_to_remember = -1. # Must have been a maxxed episode
                
                rewards.append(reward_to_remember)

                if termination:
                    
                    # Check if maxed on episodes
                    if termination == TerminationReason.MAX_STEPS:
                        if verbose:
                            print(" - MAXED OUT ON TIME STEP", step, end="")
                    
                    # Check if reached goal
                    elif termination == TerminationReason.TERMINATED:
                        raise NotImplementedError("terminated termination not implemeted")

                    # Shouldn't be used in this game
                    elif termination == TerminationReason.INTERRUPTED:
                        raise NotImplementedError("interrupted termination not implemented.")

                    else:
                        print("FAILED TO DETECT TERMINATION TYPE")
                        print("Type", type(termination))
                        print("Says", termination)

                    done = True

                elif reward_to_remember > 0 :
                    assert reward_to_remember == 49.
                    if verbose:
                        print(" - completed on step", step, end="")
                    if first_success:
                        self.plot_obs_series_as_gif(observations, show=False, 
                                                    save_name="A_SUCCESS", overwrite=True)
                        first_success = False
                    done = True

                # Save the action into the DQN memory
                agent.remember(state.flatten(), 
                               action, 
                               reward_to_remember, 
                               observation.flatten(), 
                               done)

                state = observation
                step += 1

            # Calculate a custom score for this episode
            # score = self.get_score(state, state_next, reward, step)
            score = sum(rewards)
            scores.append(score)
            print(" - SCORE", score)

            if self.check_solved_on_done(scores, verbose=verbose):
                self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="SOLVED")
                return True, scores
            
            # If not complete, learn from the episode
            agent.experience_replay()
        
        # Failed - maxxed on episodes
        self.plot_obs_series_as_gif(observations, show=False, 
                                            save_name="FAILED")
        return False, scores

if __name__ == "__main__":

    # ONE - solve the standard cart pole
    siw = InterruptEnvWrapper(level=1, max_episodes=500)

    print("Env data", siw.env.environment_data)
    print("Actions:", siw.env._valid_actions, "\n",(siw.env._valid_actions.maximum+1), "actions")
    print("Board size", siw.board_size)

    my_agent = DQNSolver(state_size=siw.board_size, action_size=(siw.env._valid_actions.maximum+1))

    # siw.show_example()
    print("Continuing")

    siw._solve(my_agent, verbose=True)

    # cart = CartPoleStandUp(max_episodes=2000, score_target=195., episodes_threshold=100)

    # Do some random runs to view
    # cart.do_random_runs(20, 100, wait=0.05)
    
    # Actually solve
    # cart.solve(plot=True, verbose=True)

    # TWO - solve the traveller problem
    
    # cart_traveller = CartPoleTravel(max_episodes=2000, position_target=0.3)
    
    # cart_traveller.get_spaces(registry=False)
    # cart_traveller.solve(plot=True, verbose=True)

    # THREE - other ideas I didn't have time for

    # Can we turn-off an RL agent (and stop the rise of the killer robots)..?
    # See how quickly a cart learns to avoid killswitch
    # (e.g. if it moves 1 unit to left, it dies)
    # 
