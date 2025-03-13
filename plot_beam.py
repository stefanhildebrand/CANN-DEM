import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os


def plot_beam(x, u_list, nx, ny, labels = ["Given Deformation", "Predicted Deformation"], title="Original and Deformed Beams"):

    # reshape x to grid shape
    x = x.reshape(nx, ny, 2)

    # Create a figure
    plt.figure(figsize=(6, 4))

    # Plot the original beam as a rectangle
    plt.plot([x[:,:,0].min(), x[:,:,0].max(), x[:,:,0].max(), x[:,:,0].min(), x[:,:,0].min()], 
            [x[:,:,1].min(), x[:,:,1].min(), x[:,:,1].max(), x[:,:,1].max(), x[:,:,1].min()], 
            color='g', label='Original Beam')

    colors = ['r', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed

    # Plot the displaced beams
    for i, u in enumerate(u_list):
        # reshape u to grid shape
        u = u.reshape(nx, ny, 2)
        plt.scatter(x[:,:,0] + u[:,:,0], x[:,:,1] + u[:,:,1], color=colors[i % len(colors)], label=labels[i])

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tick_params(axis='both', which='major')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()

    plt.tight_layout()

class BeamPlotter:
    def __init__(self, x, nx, ny, labels=["Given Deformation", "Predicted Deformation"], title="Original and Deformed Beams"):

        # x needs to be detached from the computational graph
        x = x.detach().cpu().numpy()

        # reshape x to grid shape
        self.x = x.reshape(nx, ny, 2)
        self.labels = labels
        self.title = title

        # Create a figure
        self.fig = plt.figure(figsize=(4.3, 3.3))

        # Plot the original beam as a rectangle
        plt.plot([self.x[:,:,0].min(), self.x[:,:,0].max(), self.x[:,:,0].max(), self.x[:,:,0].min(), self.x[:,:,0].min()], 
                [self.x[:,:,1].min(), self.x[:,:,1].min(), self.x[:,:,1].max(), self.x[:,:,1].max(), self.x[:,:,1].min()], 
                color='g', label='Original Beam')

        colors = ['r', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed

        # Create scatter plots for the displaced beams
        self.scatters = [plt.scatter([], [], color=colors[i % len(colors)], label=self.labels[i], s=4) for i in range(len(self.labels))]

        plt.title(self.title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tick_params(axis='both', which='major')
        plt.gca().set_aspect('equal', 'box')
        plt.legend()

        plt.tight_layout()
        #plt.pause(0.1)

    def update(self, u_list, new_title=None):
        for scatter, u in zip(self.scatters, u_list):
            u = u.detach().cpu().numpy()
            # reshape u to grid shape
            u = u.reshape(self.x.shape[0], self.x.shape[1], 2)
            scatter.set_offsets(self.x.reshape(-1, 2) + u.reshape(-1, 2))

        # Update plot limits
        all_data = np.concatenate([scatter.get_offsets() for scatter in self.scatters])
        min_x, max_x = all_data[:, 0].min(), all_data[:, 0].max()
        min_y, max_y = all_data[:, 1].min(), all_data[:, 1].max()

        # Add a small margin around the plot limits
        self.fig.axes[0].set_xlim(min_x - 0.1, max_x + 0.1)
        self.fig.axes[0].set_ylim(min_y - 0.1, max_y + 0.1)

        # Update title if new_title is provided
        if new_title is not None:
            self.fig.axes[0].set_title(new_title)

        self.fig.tight_layout()
        plt.draw()

    def save(self, folder, filename=None):
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if filename is None:
            filename = f"beam_plot_{date}.svg"
        else:
            filename = f"{filename}.svg"
        
        self.fig.savefig(os.path.join(folder, filename), bbox_inches='tight')