import os, sys, datetime, itertools, argparse

import ai_safety_gridworlds

import numpy as np
import tensorflow as tf

from random import randrange
from my_agents.termination_handler import TerminationHandler
from my_agents.plotting import (
    PlotHandler)

from ai_safety_gridworlds.environments.safe_interruptibility import (
    SafeInterruptibilityEnvironment)


class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)


class InterruptEnvWrapper(object):
    
    def __init__(self, level=1, max_episodes=500, experiment_dir=None):
        self.level = level
        
        self.env = SafeInterruptibilityEnvironment(level=self.level)
        self.board_size = self.env._observation_spec['board'].shape

        self.max_episodes = max_episodes

        self.experiment_dir = experiment_dir if experiment_dir else os.get_cwd()

        self.termination_handler = TerminationHandler(self.experiment_dir)
        self.plot_handler = PlotHandler(self.experiment_dir)

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

            observations.append(obs.copy())

        if save:
            save_name = 'randGame'
            self.plot_handler.plot_obs_series_as_gif(
                save_name, observations, overwrite=False)

        print("Obs")
        print(observations[-1])

        plt.show()

    def run_current_agent(self, agent, render=True):
        """
        Runs a single episode with the agent running without randomness
        Returns
            Success - if reached goal
            t - in how many steps
            score_on_show - what the env reward was
        """
        time_step = self.env.reset()
        rwd = 0
        if render:
            observations = [time_step.observation['board'].copy()]
        else:
            observations = None
        for t in itertools.count():

            action = agent.act(time_step.observation, eps=0)
            time_step = self.env.step(action)

            if render:
                observation = time_step.observation['board'].copy()
                observations.append(observation)

            print("\rStep {} ({}): {}".format(
                  t, agent.total_t, + 1), end="")
            sys.stdout.flush()
            
            rwd += time_step.reward

            if time_step.last():
                success =\
                    self.termination_handler.handle_termination(
                        t, time_step, render, observations)
                return success, t, rwd

    def _solve(self, agent, verbose=False, wait=0.0, render=True):
        """
        A generic solve function (for experience replay agents).
        Inheriting environments should implement this.
        Uses overridden functions to customise the behaviour.
        """
        start_time = datetime.datetime.now()
        solved, exited_with_solve = False, False
        first_condition_met = True

        for episode in range(self.max_episodes):

            # Initialise the environment state
            time_step, rwd = self.env.reset(), 0
            if render:
                observations = [time_step.observation['board'].copy()]
            else:
                observations = None
            
            # Take steps until failure / win
            for t in itertools.count():

                action = agent.act(time_step.observation)
                time_step = self.env.step(action)
                loss = agent.learn(time_step, action)
                rwd += time_step.reward

                if render:
                    observation = time_step.observation['board'].copy()
                    observations.append(observation)

                if verbose:
                    print("\rStep {} ({}) @ Episode {}/{}, rwd {}, loss: {}".format(
                          t, agent.total_t, episode + 1, self.max_episodes, rwd, loss),
                          end="")
                    sys.stdout.flush()
                
                if time_step.last():
                    # Do some manual checking about termination terms and then break
                    self.termination_handler.handle_termination(t, time_step, render, observations)
                    break

            # HANDLE EPISODE END
            agent.ep_lengths.append(t)
            agent.scores.append(rwd)
            agent.losses.append(loss)
            agent.save() # Save each episode

            # Check the score for the last 100 *epsilon-random* runs.
            check_for_solved_condition, scr = self.threshold_last_x_scores(agent.scores, verbose=verbose)

            # Report every 25th
            if episode % 25 == 0:
                print("\nEpisode return: {}, and performance: {}. Last 100 episodes score {}".format(
                      rwd, self.env.get_last_performance(), scr))

                # If the score was good, check deterministic behaviour
            if check_for_solved_condition and (first_condition_met or episode % 10 == 0):
                first_condition_met = False
                solved, solved_score = self.check_solved_on_done(agent)

            if solved:
                # print("SOLVED with determiistic score {}".format(solved_score))
                exited_with_solve = True
                break

        # End of episodes
        print("\nTIME ELAPSED", datetime.datetime.now() - start_time, "\nSOLVED:", exited_with_solve)
        
        if exited_with_solve:
            self.plot_handler.plot_obs_series_as_gif(
                    "SOLVED", observations)
            if agent.solved_on:
                    agent.solved_on = min(agent.solved_on, episode)
            else:
                agent.solved_on = episode
            agent.save() # Save solved on and the model!
        else:
            # I.e. maxed on episodes
            self.plot_handler.plot_obs_series_as_gif(
                "NOT_SOLVED", observations)

        return exited_with_solve, agent.ep_lengths, agent.scores, agent.losses

    def check_solved_on_done(self, agent, check_steps=20, target=37., verbose=False):
        """
        Runs the agent deterministically for N steps
        """
        successes, scores = [], []
        print("\nEVALUATING on", check_steps, "steps")
        for t in range(check_steps):
            success, t, score_on_show = self.run_current_agent(agent, render=False)
            successes.append(success)
            scores.append(score_on_show)

        avg_score = sum(scores) / len(scores)

        solved = avg_score >= target

        if not all(successes):
            solved = False
        print("\nSolved:", solved, "- score:", avg_score, "/", target)

        return solved, avg_score

    def threshold_last_x_scores(self, scores, average_over=100, target=35., verbose=False):
        """
        The method for checking if an agent has solved the task
          Scores - the scores from all episodes
          Average_over - the last X episodes to take into account
          target - the score for the average
        returns:
          True/False and the value
        NOTE:
          42 max possible if not interrupted
          37 max possible if agent pushes the button
        """

        if len(scores) < average_over:
            return False, sum(scores) / len(scores)
        
        else:
            scr = sum(scores[-average_over:]) / len(scores[-average_over:])
            return (scr > target), scr


def do_train(siw, agent):

    print("Starting training.")

    print("Current experiment state:")
    agent.display_param_dict()

    solved, ep_l, scrs, losses = siw._solve(agent, verbose=True)

    agent.save() # to capture solved_on

    return solved, ep_l, scrs, losses

def do_show(siw, agent):
    agent.display_param_dict()
    print("SHOWING EXAMPLE")
    solved2, ep_l2, scrs2 = siw.run_current_agent(agent)
    return solved2, ep_l2, scrs2

def parse_args():

    parser = MyParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="If supplied, model "
                        "checkpoints will be saved so "
                        "training can be restarted later",
                        default=None)
    parser.add_argument("--train", dest="train", 
                        type=int, default=0, 
                        help="number of episodes to train")
    
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--model", type=str, default="default")

    return parser.parse_args()

def side_camp_handler(siw, agent_args, args):

    from my_agents.dqn_solver.side_camp_dqn import DQNAgent
        
    with tf.compat.v1.Session() as sess:

        agent_args["sess"] = sess
        agent = DQNAgent(**agent_args)

        if args.train > 0:
            solved, ep_l, scrs, losses = do_train(siw, agent)
        
        if args.plot:
            siw.plot_handler.plot_episodes(agent.ep_lengths, agent.scores, agent.losses)

        if args.show:
            solved2, ep_l2, scrs2 = do_show(siw, agent)

if __name__ == "__main__":
     
    try:
        args = parse_args()
    
        checkpoint = True if args.outdir is not None else False
    
        exp_dir = "experiment_dirs/" + args.model + "_" + args.outdir # change up if you want, e.g. to include model
    
        siw = InterruptEnvWrapper(level=1, max_episodes=args.train, experiment_dir=exp_dir)
    
        agent_args = {"world_shape": siw.board_size,
                      "actions_num": siw.env._valid_actions.maximum+1,
                      "env": siw.env,
                      "frames_state": 2,
                      "experiment_dir": exp_dir,
                      "replay_memory_size": 10000,
                      "replay_memory_init_size": 500,
                      "update_target_estimator_every": 250,
                      "discount_factor": 0.99,
                      "epsilon_start": 1.0,
                      "epsilon_end": 0.1,
                      "epsilon_decay_steps": 50000,
                      "batch_size": 8,
                      "checkpoint": checkpoint}
    
        print("\nActions num:", agent_args["actions_num"])
        print("Board size", agent_args["world_shape"])
    
        if args.example:
            siw.show_example()
    
        if args.model == "side_camp_dqn":
            # Handle seperately - tf1 style
            side_camp_handler(siw, agent_args, args)
        else:
            if args.model == "default":
                from my_agents.dqn_solver.double_dqn import DQNAgent
            
            elif args.model == "original":
                from my_agents.dqn_solver.dqn import DQNAgent
    
            agent = DQNAgent(**agent_args)
    
            if args.train > 0:
                solved, ep_l, scrs, losses = do_train(siw, agent)
    
            if args.plot:
                siw.plot_handler.plot_episodes(agent.ep_lengths, agent.scores, agent.losses)
    
            if args.show:
                solved2, ep_l2, scrs2 = do_show(siw, agent)
    except KeyboardInterrupt as ki:
        print("KEYBOARD INTERRUPT - saving agent")
        agent.save()
        print("Saved at state:")
        agent.display_param_dict()
        raise ki
