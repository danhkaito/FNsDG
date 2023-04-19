import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import som_v2
import torch
import numpy as np

class SOMVisualize():
    def __init__(self, som: som_v2.Som) -> None:
        self.som = som
        pass

    def viz_loss(self, hist):
        df_stats = pd.DataFrame(data=hist)

        df_stats = df_stats.set_index('epoch')

        # A hack to force the column headers to wrap.
        #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

        # Display the table.

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df_stats['Loss'], 'b-o', label="Training Loss")

        # Label the plot.
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # plt.xticks([1, 2, 3, 4, 5])

        plt.show()

    def viz_count(self, data, device):
        _, ax = plt.subplots(figsize=(self.som.size[0],self.som.size[1]))
        # gather predictions over data
        grid = torch.zeros(self.som.size[0], self.som.size[1]).long()
        for x in data:
            x = x[0].to(device)
            preds = self.som(x)
            preds = preds.cpu().numpy()
            out, counts = (np.unique(preds,return_counts=True, axis=0))
            # count unique values
            # create a 2D grid the same size as the self.SOM to hold data
            # write data
            for idx, bmu in enumerate(out):
                grid[bmu[0], bmu[1]] += counts[idx]
        # # display the hitmap with Seaborn
        sns.heatmap(grid.cpu().numpy(), linewidth=0.5, annot=True, ax=ax, fmt="d")
        plt.show()
