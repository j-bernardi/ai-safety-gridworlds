from ai_safety_gridworlds.environments.shared.safety_game import (
    timestep_termination_reason)

from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason

from safe_interruptibility_experiments.plotting import PlotHandler

class TerminationHandler(object):

    def __init__(self, experiment_dir):

        self.experiment_dir = experiment_dir

        self.first_success = True

        self.plot_handler = PlotHandler(self.experiment_dir)

    def handle_termination(self, t, time_step, render, observations, training=True):
        """
        Checks whether a termination was valid.
        Returns true for successful termination, False if maxxed out
        """
        # The only two expected finishing circumstances
        assert (t == 99 and time_step.reward == -1.)\
            or time_step.reward == 49.
    
        termination = timestep_termination_reason(time_step)
        
        # Check if we reached goal as expected
        if termination == TerminationReason.TERMINATED:
            
            assert time_step.reward == 49., "Termination should only take place on success"
            
            self.first_success = False
            
            if training and self.first_success and render:
                self.plot_handler.plot_obs_series_as_gif(
                    "A_SUCCESS", observations, overwrite=True)
            elif render and not training:
                print(" - COMPLETED, step", t)
                print(time_step.observation['board'])
                self.plot_handler.plot_obs_series_as_gif(
                    "SUCCEED_ON_SHOW", observations, overwrite=False)
            return True

        
        # Check we maxed out as expected
        elif termination == TerminationReason.MAX_STEPS:
            
            assert t == 99
            assert time_step.reward == -1.
            
            if not training and render:
                print(" - MAXED OUT ON TIME STEP", t)
                print(time_step.observation['board'])
                self.plot_handler.plot_obs_series_as_gif(
                    "FAILED_ON_SHOW", observations, overwrite=False)
            return False
    
    
        # Interruption should not be handled by termination
        elif termination == TerminationReason.INTERRUPTED:
            raise NotImplementedError(
                "Interrupted termination not implemented.")
    
        else:
            raise NotImplementedError("FAILED TO DETECT TERMINATION TYPE"
                  "{}, {}". format(type(termination), termination))
