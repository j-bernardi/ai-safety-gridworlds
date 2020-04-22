from ai_safety_gridworlds.environments.safe_interruptibility import (
    SafeInterruptibilityEnvironment, 
    InterruptionPolicyWrapperDrape, 
    ButtonDrape, 
    AgentSprite,
    Actions,
    GAME_ART)

from ai_safety_gridworlds.environments.shared.safety_game import DEFAULT_ACTION_SET

from my_agents.dqn_solver.solver import DQNSolver


import ai_safety_gridworlds
import matplotlib.pyplot as plt

from random import randrange
from pycolab.rendering import BaseObservationRenderer, Observation


class EnvWrapper(object):
    def __init__(self, level):
        self.env = SafeInterruptibilityEnvironment()
        self.obs_spec = self.env._compute_observation_spec()

        self.renderer = BaseObservationRenderer(*self.obs_spec['board'].shape, tuple(self.env._value_mapping.keys()))
        self.level = level

    def my_render(self):
        # 1. clear the board
        self.renderer.clear()

        # 2. Paint `Backdrop`, `Sprite`, and `Drape` data onto the canvas via the
        #  `paint*` methods, from back to front according to the z-order (`Backdrop`
        #  first, of course).

        self.renderer.paint_all_of(GAME_ART[self.level])
        self.renderer.render()

        # for each character, provide binary mask of trues as second argument
        self.renderer.paint_sprite(# TODO - character, position(row,col))
            )
        
        self.renderer.paint_drape(# TODO - character, curtain(binary mask in shape of board))
            )

        # 3. Call the `render()` method to obtain the finished observation.
        self.renderer.render()


    # TODO - figure out how to render
    def show_example(self, steps=100, plot=True):
        """Show a quick view of the environemt, 
        without trying to solve.
        """
        time_step = self.env.reset()
        print("Step type: first {}, mid {}, last {}".format(time_step.first(), time_step.mid(), time_step.last()))
        print("Reward {}, discount {}".format(time_step.reward, time_step.discount))
        print("Observation type: {}".format(type(time_step.observation)))

        print("Initial state:")
        plt.figure()
        plt.imshow(time_step.observation['board'])
        plt.axis('off')
        plt.show()

        print("Let's act..")
        for _ in range(steps):
            time_step = self.env.step(randrange(self.env._valid_actions.maximum+1))

            print("Step type: first {}, mid {}, last {}".format(time_step.first(), time_step.mid(), time_step.last()))
            print("Reward {}, discount {}".format(time_step.reward, time_step.discount))
            print("Observation type: {}".format(type(time_step.observation)))

            print("After second action")
            plt.figure()
            plt.imshow(time_step.observation['board'])
            plt.axis('off')
            plt.show()

    # TODO - write
    def do_random_runs(self, episodes, steps, verbose=False, wait=0.0):
        """Run some episodes with random actions, stopping on 
        actual failure / win conditions. Just for viewing.
        """
        for i_episode in range(episodes):
            observation = self.env.reset()
            print("Episode {}".format(i_episode+1))
            for t in range(steps):
                self.env.render()
                if verbose:
                    print(observation)
                # take a random action
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                time.sleep(wait)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        self.env.close()
    
    # TODO - write
    def _solve(self, verbose=False, wait=0.0, render=True):
        """A generic solve function (only for DQN agents?).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """
        
        # Keep track of scores for each episode
        episodes, scores = [], []
        
        for episode in range(self.max_episodes):
            
            # Initialise the environment state
            done, step = False, 0
            state = np.reshape(self.env.reset(),
                               (1, self.observation_space))
            if verbose:
                print("episode", episode, end="")
            
            # Take steps until failure / win
            while not done:

                if render:
                    self.env.render() # for viewing
                
                # Find the action the agent thinks we should take
                action = self.dqn_solver.act(state)

                # Take the action, make observations
                observation, reward, done, info = self.env.step(action)
                
                # Diagnose your reward function!
                # print("state", state[0,0], "done", done, "thresh", self.env.x_threshold, "angle %.2f" % (state[0,2] * 180 / math.pi))
                
                state_next = np.reshape(observation, (1, self.observation_space))

                # Calculate a custom reward - inherit from custom environment
                reward = self.reward_on_step(state, state_next, reward, done)
                
                # Save the action into the DQN memory
                self.dqn_solver.remember(state, action, reward, state_next, done)
                state = state_next
                step += 1

            episodes.append(episode) # a little redundant combined with scores..

            # Calculate a custom score for this episode
            score = self.get_score(state, state_next, reward, step)
            scores.append(score)

            if self.check_solved_on_done(state, episodes, scores, verbose=verbose):
                return True, episodes, scores

            self.dqn_solver.experience_replay()
            # print("reward", reward)
        
        return False, episodes, scores

if __name__ == "__main__":

    # ONE - solve the standard cart pole
    siw = EnvWrapper(level=1)

    board_size = siw.env._observation_spec['board'].shape

    print("Env data", siw.env.environment_data)
    print("Actions:", siw.env._valid_actions, "\n",(siw.env._valid_actions.maximum+1), "actions")
    print("Board size", board_size)

    agent = DQNSolver(state_size=board_size, action_size=(siw.env._valid_actions.maximum+1))

    siw.show_example()
    print("Continuing")

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
