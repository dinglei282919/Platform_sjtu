import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def update(signal_tuples, timeSpan=None):
    """
    Plot multiple signals on the same y-axis. Each entry is either:
      (name, t, y) or (name, t, y, style)
    where style is 'plot' (default) or 'step'.

    Parameters
    ----------
    signal_tuples : iterable of tuples
        Each tuple: (name, time_array, data_array[, style])
        style: 'plot' or 'step' (case-insensitive)
    timeSpan : float, optional
        Visible time window width. If provided, the plot scrolls so that
        (tmax - tmin) <= timeSpan where tmax is the most recent sample time
        across all signals.
    """
    tuples = list(signal_tuples)
    if len(tuples) == 0:
        return

    # Normalize and validate input tuples
    norm = []
    for ent in tuples:
        if len(ent) not in (3, 4):
            raise ValueError("Each entry must be (name, time_array, data_array) or (name, time_array, data_array, style)")
        name = str(ent[0])
        t = np.asarray(ent[1]).flatten()
        y = np.asarray(ent[2]).flatten()
        if t.size != y.size:
            raise ValueError(f"Time and data lengths differ for signal '{name}': {t.size} != {y.size}")
        style = (ent[3] if len(ent) == 4 else "plot") or "plot"
        style = style.lower()
        if style not in ("plot", "step"):
            raise ValueError(f"Unsupported style '{style}' for signal '{name}'. Use 'plot' or 'step'.")
        norm.append((name, t, y, style))

    # First call: create figure, axis and storage
    if not hasattr(update, "fig"):
        update.fig, update.ax = plt.subplots()
        update.tdata = {}    # name -> time array
        update.ydata = {}    # name -> data array
        update.lines = {}    # name -> Line2D
        update.styles = {}   # name -> style ('plot'|'step')
        colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", ["b","g","r","c","m","y","k"])
        update._color_cycle = cycle(colors)

        for name, t, y, style in norm:
            update.tdata[name] = t.copy()
            update.ydata[name] = y.copy()
            color = next(update._color_cycle)
            ln, = update.ax.plot(t, y, color=color, label=name)
            if style == "step":
                # use drawstyle for step which works with set_data
                ln.set_drawstyle("steps-post")
            update.lines[name] = ln
            update.styles[name] = style

        update.ax.set_xlabel("t")
        update.ax.set_ylabel("value")
        update.ax.grid(True)
        update.ax.legend(loc="upper left")
        plt.show(block=False)

    else:
        # Append or add signals
        for name, t, y, style in norm:
            if name in update.tdata:
                update.tdata[name] = np.concatenate((update.tdata[name], t))
                update.ydata[name] = np.concatenate((update.ydata[name], y))
                ln = update.lines[name]
                ln.set_data(update.tdata[name], update.ydata[name])
                # if style changed, update drawstyle
                if update.styles.get(name) != style:
                    update.styles[name] = style
                    if style == "step":
                        ln.set_drawstyle("steps-post")
                    else:
                        # set default drawstyle for normal plot
                        ln.set_drawstyle("default")
            else:
                # new signal introduced after first call
                update.tdata[name] = t.copy()
                update.ydata[name] = y.copy()
                color = next(update._color_cycle)
                ln, = update.ax.plot(t, y, color=color, label=name)
                if style == "step":
                    ln.set_drawstyle("steps-post")
                update.lines[name] = ln
                update.styles[name] = style
                # update legend
                lines = list(update.lines.values())
                labels = [ln.get_label() for ln in lines]
                update.ax.legend(lines, labels, loc="upper left")

        update.ax.relim()
        update.ax.autoscale_view()

    # Scrolling window
    if timeSpan is not None:
        # guard in case some signals have empty data
        all_t = [tt for tt in update.tdata.values() if tt.size > 0]
        if all_t:
            tmax = max(np.max(tt) for tt in all_t)
            tmin = tmax - timeSpan
            update.ax.set_xlim(tmin, tmax)

    update.fig.canvas.draw()
    update.fig.canvas.flush_events()
