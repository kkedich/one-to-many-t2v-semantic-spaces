

def plot_at_ax(dataframe, ax, x_axis_label, y_axis_label, distributions, title,
               z_axis=None, put_id_on_points=False):
    assert len(distributions) > 0, 'There is no distribution to sample'

    plots = []
    last_id_index = 0

    for dist in distributions:
        x_values = dataframe[dataframe["distribution"] == dist][x_axis_label]
        y_values = dataframe[dataframe["distribution"] == dist][y_axis_label]

        z_values = None
        if z_axis is not None:
            z_values = dataframe[dataframe["distribution"] == dist][z_axis]
        current_color = dataframe[dataframe["distribution"] == dist]["color"]

        if z_axis is None:
            current_plot = ax.scatter(x_values, y_values, c=current_color, marker=".", alpha=0.6)

            if put_id_on_points:
                list_ids = list(range(len(x_values)))

                for i, current_id in enumerate(list_ids):
                    ax.annotate(
                        current_id, (x_values[i + last_id_index], y_values[i + last_id_index]))
                last_id_index += len(list_ids)
        else:
            current_plot = ax.scatter(x_values, y_values, z_values,
                                      c=current_color, marker=".", alpha=0.6)

            # Make simple, bare axis lines through space:
            # https://python-graph-gallery.com/372-3d-pca-result/
            if len(x_values) > 0 and len(y_values) > 0 and len(z_values) > 0:
                xAxisLine = ((min(x_values), max(x_values)), (0, 0), (0, 0))
                ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], "#000000")
                yAxisLine = ((0, 0), (min(y_values), max(y_values)), (0, 0))
                ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], "#000000")
                zAxisLine = ((0, 0), (0, 0), (min(z_values), max(z_values)))
                ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], "#000000")

            ax.set_zlabel(z_axis)
        plots.append(current_plot)

    ax.grid(True)
    ax.legend(plots, distributions, loc="best")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    if title is not None:
        ax.set_title(title)
