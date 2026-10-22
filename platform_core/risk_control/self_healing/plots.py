"""Object-based figures: no pyplot/backend or global style changes."""
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties


_LABELS = {
    "total_flow": ("Total supply (priority output)", r"Outlet flow, $Q$"),
    "Q1": ("Tank 1 outlet", r"Outlet flow, $Q_1$"),
    "Q2": ("Tank 2 outlet", r"Outlet flow, $Q_2$"),
    "Q3": ("Tank 3 outlet (blockage)", r"Outlet flow, $Q_3$"),
    "h1_setpoint": ("Tank 1 level setpoint", r"Level setpoint, $h_{1,\mathrm{sp}}$"),
    "h2_setpoint": ("Tank 2 level setpoint", r"Level setpoint, $h_{2,\mathrm{sp}}$"),
    "h3_setpoint": ("Tank 3 level setpoint", r"Level setpoint, $h_{3,\mathrm{sp}}$"),
    "C": ("Concentration (priority output)", r"Concentration, $C$"),
    "T": ("Reactor temperature (priority output)", r"Temperature, $T$"),
    "Tc": ("Cooling-water outlet temperature", r"Temperature, $T_{\mathrm{c}}$"),
    "Ci": ("Feed concentration", r"Concentration, $C_{\mathrm{i}}$"),
    "Ti": ("Feed temperature", r"Temperature, $T_{\mathrm{i}}$"),
    "Tci": ("Cooling-water inlet temperature", r"Temperature, $T_{\mathrm{ci}}$"),
    "Tsp": ("Reactor temperature setpoint", r"Temperature, $T_{\mathrm{sp}}$"),
}


def figures(result, *, duration=None):
    colors = {"normal": "#718096", "fault": "#d1495b", "scsc": "#007f8b", "ccsc": "#7a4cb0"}
    # Explicit local fonts keep platform-wide Chinese/monospace defaults out
    # of these scientific axes without changing global Matplotlib settings.
    label_font = FontProperties(family="DejaVu Sans", size=11)
    tick_font = FontProperties(family="DejaVu Sans", size=10)
    title_font = FontProperties(family="DejaVu Sans", size=11, weight="semibold")
    legend_font = FontProperties(family="DejaVu Sans", size=10)
    plots = {}
    for kind, labels in (("outputs", result["output_names"]), ("inputs", result["input_names"])):
        fig = Figure(figsize=(11, 7.5), constrained_layout=True)
        if len(labels) == 3:
            grid = fig.add_gridspec(2, 2)
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
        else:
            axes = list(fig.subplots(2, 2).flat)
        for j, ax in enumerate(axes):
            for name, run in result["runs"].items():
                ax.plot(result["time"], run[kind][:, j], label=name.upper(), color=colors[name],
                        lw=1.6 if name == "normal" else 1.3,
                        ls="--" if name == "normal" else "-",
                        zorder=4 if name == "normal" else 2)
            if kind == "outputs":
                ax.axhline(result["reference"][j], color="#222222", ls=":", lw=.9,
                           label="Reference", zorder=5)
            ax.axvline(result["config"]["fault_start"], color="#999999", ls="--", lw=.8)
            title, ylabel = _LABELS.get(labels[j], (labels[j], labels[j]))
            ax.set_title(f"({chr(97 + j)}) {title}", loc="left", pad=10,
                         fontproperties=title_font, color="#243447")
            ax.set_ylabel(ylabel, fontproperties=label_font, math_fontfamily="dejavusans",
                          labelpad=8, color="#334155")
            ax.set_xlabel("Model time", fontproperties=label_font, labelpad=7, color="#334155")
            ax.tick_params(axis="both", colors="#475569", direction="out", length=4, width=.8)
            for text in [*ax.get_xticklabels(), *ax.get_yticklabels(),
                         ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
                text.set_fontproperties(tick_font)
            ax.grid(color="#e2e8f0", linewidth=.7, alpha=.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color("#94a3b8")
                ax.spines[side].set_linewidth(.8)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside upper center", ncol=5, prop=legend_font,
                   frameon=False, handlelength=2.8, columnspacing=1.6, labelcolor="#334155")
        plots[kind] = fig
    return plots
