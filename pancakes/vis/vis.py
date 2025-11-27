import neurite as ne


def plotpreds(x,
              predictions,
              y,
              M,
              num_sets,
              label_cmap='turbo'):
    
    """
    Plot predictions along with inputs and available labels.
    Args:
        x: Input tensor of shape (num_sets, 1, H, W)
        predictions: Prediction tensor of shape (num_sets, M, 1, H, W)
        y: Label tensor of shape (num_sets, 1, H, W)
        M: Number of prediction maps
        num_sets: Number of images in the set
        label_cmap: Colormap for the labels
    """
    pred2plot = []
    titles = ['Input' for i in range(num_sets)]
    for i in range(M):
        titles.extend([f'Map {i+1}' for j in range(num_sets)])
        pred2plot.extend(predictions[:, i])

    titles.extend(['Label Available' for i in range(num_sets)])
    cmaps = [
                *['gray'] * num_sets,
                *[label_cmap] * len(pred2plot),
                * ['gray'] * num_sets,
            ]

    nrows = M+2

    # Plot with neurite
    fig, _ = ne.plot.slices(
        slices_in=[
            *x[:, 0].cpu().numpy(),
            *pred2plot,
            *y[:, 0],
        ],
        cmaps=cmaps,
        grid = (nrows, num_sets),
        titles=titles,
        width=7
    );
    return fig