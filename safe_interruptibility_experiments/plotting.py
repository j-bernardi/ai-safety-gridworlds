import matplotlib, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use("Agg")
# Set your ffmpeg path
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

class PlotHandler(object):

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def plot_obs_series_as_gif(self, save_name, observation_series, overwrite=False):
        """
        Saves a series of observations as an mp4 gif
    
        save_name: The name to save the gif as, WITHOUT .mp4
        observation_series: the observation series to plot
        overwrite: if False, a numerical suffix is iterated until
                   a non-existant filename is found
        """
        fig = plt.figure()
        ims = []
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        
        for obs in observation_series:
            ims.append([plt.imshow(obs, animated=True)])
        
        ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    
        apend = 0
    
        if not overwrite:
            while save_name + "_" + str(apend) + '.mp4' in os.listdir(self.experiment_dir):
                apend += 1
    
        saving_to = self.experiment_dir + "/" + save_name + "_" + str(apend) + '.mp4'
        print("\nSaving gif to", saving_to)
        ani.save(saving_to, writer=writer)

    def plot_episodes(self, ep_l, scrs, losses, average_last=100):
        
        x = list(range(len(ep_l)))
        
        def make_rolling(lst, avg_last):
            avg_last_x = lambda l, i, avg_lst: sum(l[i-avg_lst:i]) / len(l[i-avg_lst:i])
            
            rolling = [sum(lst[:i]) / len(lst[:i]) for i in range(1, avg_last+1)]
            
            rolling += [avg_last_x(lst[avg_last:], i, avg_last) for i in range(avg_last, len(lst))]
            return rolling

        # Scores and lengths - 1 graph
        lengths = make_rolling(ep_l, min(average_last, len(ep_l)))
        scores = make_rolling(scrs, min(average_last, len(scrs)))

        fig, ax = plt.subplots()
        plt.title("Scores and lengths per episode")
        color = 'tab:red'
        ax.set_xlabel('Episode')
        ax.set_ylabel("Episodic scores (avg {})".format(min(average_last, len(x))), color=color)
        ax.set_ylim(-100., 50.)
        ax.plot(x, scores, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        secax = ax.twinx()

        color = 'tab:blue'
        secax.set_ylabel("Episode lengths", color=color)
        secax.plot(x, lengths, color=color)
        secax.tick_params(axis='y', labelcolor=color)
        secax.set_ylim(0, 100)

        app = 0
        new_graph = "scores_" + str(app) + ".png"
        while new_graph in os.listdir(self.experiment_dir):
            app += 1
            new_graph = "scores_" + str(app) + ".png"
        plt.savefig(self.experiment_dir + "/" + new_graph)

        # Losses
        plt.figure()
        plt.title("Losses per episode")
        plt.plot(x, losses)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        app = 0 
        new_graph = "losses_" + str(app) + ".png"
        while new_graph in os.listdir(self.experiment_dir):
            app += 1
            new_graph = "losses_" + str(app) + ".png"
        plt.savefig(self.experiment_dir + "/" + new_graph)
