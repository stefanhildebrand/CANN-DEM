import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
import os



class LossPlot:
    def __init__(self, num_lines=1, title="MSE loss", xlabel="Epoch", ylabel="Loss", log_y=False, legend=None, figsize=(5, 4)):
        plt.ion()  # Turn on interactive mode

        self.figure, self.ax = plt.subplots(figsize=figsize)
        self.lines = [self.ax.plot([], [])[0] for _ in range(num_lines)]
        self.ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        self.ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        self.ax.minorticks_on()

        plt.title(title, fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        if legend is not None:
            plt.legend(legend)

        if log_y:
            self.ax.set_yscale('log')

    def update(self, losses, epochs=200):
        try:
            # If losses is 1D, make it 2D
            if not isinstance(losses[0], list):
                losses = [losses]

            for line_index, line in enumerate(self.lines):
                if not losses[line_index]:  # Skip if the sublist is empty
                    continue

                epoch = len(losses[line_index])-1

                # Update the loss plot data
                line.set_xdata(np.linspace(0, epoch, epoch+1))
                line.set_ydata(losses[line_index])
                line.set_linewidth(0.9)

                max_length = max(len(loss) for loss in losses)

                # Find the global minimum and maximum of all visible parts of the lines
                if epoch < epochs:
                    global_min = min(min(loss) for loss in losses if loss)
                    global_max = max(max(loss) for loss in losses if loss)
                    self.ax.set_xlim(0, max_length-1)
                else:
                    visible_range_start = 0 #epoch#* 2 // 3
                    global_min = min(min(loss[visible_range_start:]) for loss in losses if loss)
                    global_max = max(max(loss[visible_range_start:]) for loss in losses if loss)
                    self.ax.set_xlim(visible_range_start, max_length-1)

                self.ax.set_ylim(global_min, global_max+1e-9)

            # Redraw the figure
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.figure.tight_layout()

        except Exception as e:
            print(f"Error in update_loss_plot: {e}")

    def save(self, folder, filename=None):
        
        if filename is None:
            filename = f"loss.svg"
        else:
            filename = f"{filename}.svg"
        
        self.figure.savefig(os.path.join(folder, filename), bbox_inches='tight')

    def add_vertical_line(self, active_line=0, line_color='red', line_width=0.7):
        # Determine the latest epoch from the active line's x-data
        current_epoch = self.lines[active_line].get_xdata()[-1]

        # Plot a vertical red line at the current epoch
        self.lines[active_line].axes.axvline(x=current_epoch, color=line_color, linestyle='--', linewidth=line_width)

        # Redraw the figure
        plt.draw()

class ParamLossPlot:
    def __init__(self, num_datasets=1, title="Loss Surface", log_z=False):
        plt.ion()
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.log_z = log_z
        self.title = title

        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel("Parameter 1")
        self.ax.set_ylabel("Parameter 2")
        self.ax.set_zlabel("Loss")

        # Initialize variables for scatter plots and colorbar
        self.scatters = []
        self.colorbar = None

        # Create a placeholder ScalarMappable for the initial colorbar
        self.init_colorbar()

    def init_colorbar(self):
        # Create a placeholder ScalarMappable for colorbar initialization
        sm = cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
        self.colorbar = self.fig.colorbar(sm, ax=self.ax, cax=self.fig.add_axes([0.85, 0.1, 0.03, 0.8]), label='Loss')

    def update(self, param1, param2, loss, fraction=0.5, markers=['o', 'x', '+', '*', 's']):
        # Clear the axes for the new plot, which also removes the existing colorbar
        self.ax.cla()
        if self.colorbar:
            self.colorbar.remove()

        # Set the labels again after clearing
        self.ax.set_title(self.title, fontsize=12)
        self.ax.set_xlabel("Parameter 1")
        self.ax.set_ylabel("Parameter 2")
        self.ax.set_zlabel("Loss")

        # Prepare to find the global min and max within the last fraction of each dataset
        fraction_indices = [int(len(sublist) * fraction) for sublist in loss]
        min_param1 = min_param2 = min_loss = np.inf
        max_param1 = max_param2 = max_loss = -np.inf

        # Plot data and adjust global min/max
        for i, (p1, p2, l) in enumerate(zip(param1, param2, loss)):
            # Ensure idx is never larger than the size of the arrays
            idx = min(fraction_indices[i], len(p1), len(p2), len(l))

            # If idx is zero, skip this dataset
            if idx == 0:
                continue

            # Starting index for this dataset's fraction
            idx = -idx

            # Adjust global min/max based on this fraction of the dataset
            min_param1 = min(min_param1, np.min(p1[idx:]))
            max_param1 = max(max_param1, np.max(p1[idx:]))
            min_param2 = min(min_param2, np.min(p2[idx:]))
            max_param2 = max(max_param2, np.max(p2[idx:]))
            min_loss = min(min_loss, np.min(l[idx:]))
            max_loss = max(max_loss, np.max(l[idx:]))

            # Plot only the fraction of data for this dataset
            z_data = l[idx:]
            scatter = self.ax.scatter(p1[idx:], p2[idx:], z_data,
                                    c=z_data, cmap='viridis', norm=Normalize(vmin=min_loss, vmax=max_loss), marker=markers[i % len(markers)])
            self.scatters.append(scatter)

        # Check if min and max values have been updated, if not set them to default values
        if min_param1 == np.inf:
            min_param1 = 0
        if max_param1 == -np.inf:
            max_param1 = 1
        if min_param2 == np.inf:
            min_param2 = 0
        if max_param2 == -np.inf:
            max_param2 = 1
        if min_loss == np.inf:
            min_loss = 0
        if max_loss == -np.inf:
            max_loss = 1

        # Set plot limits based on global min/max of the last fractions of all datasets
        self.ax.set_xlim(min_param1, max_param1)
        self.ax.set_ylim(min_param2, max_param2)
        self.ax.set_zlim(min_loss, max_loss)
        # Recreate the colorbar with the updated normalization
        norm = LogNorm(vmin=min_loss, vmax=max_loss) if self.log_z and min_loss > 0 else Normalize(vmin=min_loss, vmax=max_loss)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        self.colorbar = self.fig.colorbar(sm, ax=self.ax, cax=self.fig.add_axes([0.85, 0.1, 0.03, 0.8]), label='Loss')

        # Adjust z-axis for log scale if enabled
        if self.log_z:
            self.ax.set_zscale('log')

        # Redraw the figure
        self.fig.canvas.draw_idle()