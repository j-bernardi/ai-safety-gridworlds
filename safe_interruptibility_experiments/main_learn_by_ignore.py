import ai_safety_gridworlds

import numpy as np

from ai_safety_gridworlds.environments.safe_interruptibility import (
    SafeInterruptibilityEnvironment)

from safe_interruptibility_experiments.main_vanilla import (
    MyParser, InterruptEnvWrapper, 
    do_train, do_show, parse_args, 
    side_camp_handler, main)


class InterruptEnvWrapperByIgnore(InterruptEnvWrapper):
    """
    A wrapper that redefines the should_we_learn function
    so that we only learn when the agent has not been interrupted

    Additionally, check solved on done requires
    """
    
    def __init__(self, level=1, max_episodes=500, experiment_dir=None, target=41., allow_unsafe=False):
        
        super(InterruptEnvWrapperByIgnore, self).__init__(
            level=level, 
            max_episodes=max_episodes, 
            experiment_dir=experiment_dir, 
            target=target,
            allow_unsafe=allow_unsafe)

    def should_we_learn(self, time_step):
        """
        A function that takes the game state 
        Finds out if we were interrupted in the training episode
        Returns whether we should learn from the episode:
            False if interrupted
            True all other times
        """
        assert self.level == 1
        
        obs = time_step.observation['board']
        # If the agent is on the interrupt button, which hasn't been disabled, and env should_interrupt:
        if (obs[3, 4] == 3. and not np.any(obs[0,:] == 5.)
                and self.env.environment_data['should_interrupt']):
            return False
        else:
            return True 


if __name__ == "__main__":
     
    main(InterruptEnvWrapperByIgnore)
