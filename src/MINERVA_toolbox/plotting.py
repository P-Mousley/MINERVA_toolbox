import os
from pathlib import Path
from typing import List

import h5py
import ipywidgets as wg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# from matplotlib import cm as mpl_cm
from IPython.display import display
from matplotlib.colors import LogNorm

from MINERVA_toolbox.processing import fit_peaks


def view_2D_image(img_data, cmap="viridis", flip_vertical=False, flip_horizontal=False):
    # Load Fabio image
    data = np.flipud(img_data.astype(float))

    # Optional flipping
    if flip_vertical:
        data = np.flipud(data)
    if flip_horizontal:
        data = np.fliplr(data)

    # Basic stats
    dmin, dmax = np.nanmin(data), np.nanmax(data)

    # Widgets -------------------------------------------------------------------

    vmin_slider = wg.FloatSlider(
        min=dmin,
        max=dmax,
        value=dmin,
        description="Min",
        continuous_update=False,
        readout=False,
    )
    vmax_slider = wg.FloatSlider(
        min=dmin,
        max=dmax,
        value=dmax,
        description="Max",
        continuous_update=False,
        readout=False,
    )

    vmin_text = wg.FloatText(value=dmin)
    vmax_text = wg.FloatText(value=dmax)

    log_box = wg.Checkbox(value=False, description="Log scale")

    # Sync slider <->  text ------------------------------------------------------

    def sync_vmin_slider(change):
        vmin_text.value = change["new"]

    vmin_slider.observe(sync_vmin_slider, names="value")

    def sync_vmin_text(change):
        if change["new"] <= vmax_slider.value:
            vmin_slider.value = change["new"]

    vmin_text.observe(sync_vmin_text, names="value")

    def sync_vmax_slider(change):
        vmax_text.value = change["new"]

    vmax_slider.observe(sync_vmax_slider, names="value")

    def sync_vmax_text(change):
        if change["new"] >= vmin_slider.value:
            vmax_slider.value = change["new"]

    vmax_text.observe(sync_vmax_text, names="value")

    # Update function -----------------------------------------------------------

    def update(vmin, vmax, log_scale):
        plt.figure(figsize=(7, 7))

        if log_scale:
            # Avoid non-positive values for LogNorm
            safe_data = np.clip(data, a_min=1e-12, a_max=None)
            norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
            plt.imshow(safe_data, cmap=cmap, norm=norm)
        else:
            plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        plt.colorbar(label="Intensity")
        plt.axis("off")
        plt.show()

    # Connect widgets  to updater
    out = wg.interactive_output(
        update, {"vmin": vmin_slider, "vmax": vmax_slider, "log_scale": log_box}
    )

    # Layout --------------------------------------------------------------------

    layout = wg.VBox(
        [
            wg.HBox([wg.Label(""), vmin_slider, vmin_text]),
            wg.HBox([wg.Label(""), vmax_slider, vmax_text]),
            log_box,
            out,
        ]
    )

    display(layout)


def remove_colorbars(fig):
    for ax in fig.axes:
        if ax.get_label() == "colorbar":
            ax.remove()


def remove_axes(fig):
    for ax in fig.axes:
        fig.delaxes(ax)


def parse_i07_giwaxs(data, title=None):
    startkeys = data.keys()
    if "qpara_qperp" in startkeys:
        return plot_qmap_i07, title
    elif "exit_angles" in startkeys:
        return plot_exitmap_i07, title
    elif "integrations" in startkeys:
        return plot_ivq_i07, title


def parse_i07_giwaxs_plotly(data, title=None):
    startkeys = data.keys()
    if "qpara_qperp" in startkeys:
        return plot_qmap_i07, title
    elif "exit_angles" in startkeys:
        return plot_exitmap_i07_plotly, title
    elif "integrations" in startkeys:
        return plot_ivsq_i07_plotly, title


def plot_i07_giwaxs(h5file: Path):
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
    with h5py.File(h5file) as data:
        outfunc, title = parse_i07_giwaxs(data, h5file.stem)
        outfunc(data, title, figax=[fig, ax1])


def get_files(folder, scantype):
    return [f for f in os.listdir(folder) if scantype in f]


def compare_qmap_list(paths: List[Path]):
    row_num = int(np.ceil(len(paths) / 2))
    fig, axs = plt.subplots(row_num, 2, figsize=(10, 5 * row_num))
    axlist = axs.flatten()
    for i, path in enumerate(paths):
        with h5py.File(path) as h5data:
            plot_qmap_i07(h5data, path.name, (fig, axlist[i]))
    # plt.tight_layout()
    plt.show()


def compare_ivq_list(paths: List[Path], logscale=False):
    fig, axs = plt.subplots(figsize=(10, 5))
    for path in paths:
        h5data = h5py.File(path)
        plot_ivq_i07(h5data, path.name, (fig, axs), log=logscale)
    if logscale:
        axs.set_yscale("log")
    plt.show()


def plot_i07_list(dirpath: Path, scantype: str, title=None):

    folder_text = wg.Text(value=str(dirpath), layout=wg.Layout(width="100%"))

    # initial file list
    file_list = wg.Dropdown(
        options=get_files(folder_text.value, scantype), description=f"{scantype} files"
    )

    # --- Update list when folder changes ---
    def update_list(folder):
        if not os.path.exists(folder):
            file_list.options = ["folder not found"]
            return
        try:
            new_files = get_files(folder, scantype)
            file_list.options = new_files
        except FileNotFoundError:
            file_list.options = []

    wg.interact(update_list, folder=folder_text)

    # --- Plotting callback ---
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=100)

    def update_plot(filename, log_scale):
        if filename == "folder not found":
            return
        remove_axes(fig)
        ax1 = fig.add_subplot(1, 1, 1)
        folder_path = Path(folder_text.value)
        h5file = folder_path / filename

        with h5py.File(h5file) as data:
            outfunc, title = parse_i07_giwaxs(data, h5file.stem)
            outfunc(data, title, figax=[fig, ax1], log=log_scale)
            plt.show()

        fig.canvas.draw_idle()

    log_box = wg.Checkbox(value=False, description="Log y-scale")
    return wg.interact(update_plot, filename=file_list, log_scale=log_box)


def get_i07_i_q_data(data):
    int_data = np.array(data["integrations/Intensity"])
    q_data = np.array(data["integrations/Q_angstrom^-1"])
    return int_data, q_data


def get_i07_exitmap_data(data):
    map2d = np.array(data["exit_angles/exit_angles_map"])

    exit_para = np.array(data["exit_angles/map_para"])
    exit_perp = np.array(data["exit_angles/map_perp"])
    return map2d, exit_para, exit_perp


def get_i07_qmap_data(data):
    map2d = np.array(data["qpara_qperp/qpara_qperp_map"])
    q_para = np.array(data["qpara_qperp/map_para"])
    q_perp = -1 * np.array(data["qpara_qperp/map_perp"])
    return map2d, q_para, q_perp


def plot_ivq_i07(data, title, figax, log):
    int_data, q_data = get_i07_i_q_data(data)
    while len(np.shape(q_data)) > 1:
        q_data = q_data[0]
    while len(np.shape(int_data)) > 1:
        int_data = int_data[0]
    # wg.interact(plot_1D_profile,q=q_data,intensity=int_data,title=title,ax=ax)
    plot_1D_profile(
        q=q_data, intensity=int_data, title=title, figax=figax, log_scale=log
    )
    # plt.show()


def plot_exitmap_i07(data, title, figax):
    map2d, exit_para, exit_perp = get_i07_exitmap_data(data)
    while len(np.shape(map2d)) > 2:
        map2d = map2d[0]
    while len(np.shape(exit_para)) > 1:
        exit_para = exit_para[0]
    while len(np.shape(exit_perp)) > 1:
        exit_perp = exit_perp[0]
    if "map_para_unit" in data["exit_angles"].keys():
        axlabel_para = data["exit_angles/map_para_unit"][()].decode()
        axlabel_perp = data["exit_angles/map_perp_unit"][()].decode()
        axlabels = [axlabel_para, axlabel_perp]
        plot_2D_map(
            map2d, exit_para, exit_perp, labels=axlabels, title=title, figax=figax
        )
    else:
        plot_2D_map(map2d, exit_para, exit_perp, title=title, figax=figax)


def plot_qmap_i07(data, title, figax, log_scale=False):
    map2d, q_para, q_perp = get_i07_qmap_data(data)
    while len(np.shape(map2d)) > 2:
        map2d = map2d[0]
    while len(np.shape(q_para)) > 1:
        q_para = q_para[0]
    while len(np.shape(q_perp)) > 1:
        q_perp = q_perp[0]

    if "map_para_unit" not in data["qpara_qperp/"].keys():
        plot_2D_map(map2d, q_para, q_perp, title=title, figax=figax)
    else:
        axlabel_para = data["qpara_qperp/map_para_unit"][()].decode()
        axlabel_perp = data["qpara_qperp/map_perp_unit"][()].decode()
        axlabels = [axlabel_para, axlabel_perp]

        plot_2D_map(map2d, q_para, q_perp, labels=axlabels, title=title, figax=figax)


def plot_1D_profile(
    q,
    intensity,
    qmin=None,
    qmax=None,
    log_scale=False,
    title="1D Profile",
    label=None,
    figax=None,
):
    """
    Plot a 1D intensity profile.
    """
    if figax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig, ax = figax
    ax.plot(q, intensity, label=label)
    ax.set_xlabel("Q (A^-1)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
        # ax.set_xscale("log")
    if qmin is not None and qmax is not None:
        ax.set_xlim(qmin, qmax)
    if label:
        ax.legend()
    # plt.show()


def plot_chi_profile(
    chi,
    intensity,
    chi_min=None,
    chi_max=None,
    log_scale=False,
    title="Chi Profile",
    label=None,
):
    """
    Plot a 1D intensity profile versus chi.
    """
    plt.figure(figsize=(10, 10))
    plt.plot(chi, intensity, label=label)
    plt.xlabel("Chi (degrees)")
    plt.ylabel("Intensity")
    plt.title(title)
    if chi_min is not None and chi_max is not None:
        plt.xlim(chi_min, chi_max)
    if label:
        plt.legend()
    plt.show()


def plot_2D_map(
    intensity,
    xy,
    z,
    vmin=None,
    vmax=None,
    dpi=100,
    limits=None,
    cm="viridis",
    labels=["$q_{xy}$  [Å⁻¹]", "$q_z$  [Å⁻¹]"],
    title="Q_para Vs Q_perp map",
    figax=None,
):

    extent = [xy.min(), xy.max(), z.min(), z.max()]
    if vmax is None:
        vmax = intensity.max()
    if vmin is None:
        vmin = np.max(np.array([intensity.min(), 0.01]))
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    else:
        fig, ax = figax
    im = ax.imshow(
        intensity,
        cmap=cm,
        extent=extent,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect=1,
    )
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    cbar = plt.colorbar(im, ax=ax, location="right")
    ax.set_aspect("auto")
    # plt.show()


def peakfit_and_plot(peaklist, x, y):
    # pars += peak2.guess(y, x=x)

    result, comps, y_fit, xnew = fit_peaks(peaklist, x, y)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    # ax1.plot(x, y, "k.", ms=3, label="Data")
    # ax1.plot(x, y_fit, "r-", lw=2, label="Total fit")
    result.plot_fit(ax=ax1)

    # individual components
    ax1.plot(
        xnew, comps["bkg_"], color="tab:blue", ls="--", lw=2, label="Background (bkg_)"
    )
    for pnum in np.arange(len(peaklist)) + 1:
        ax1.plot(
            xnew,
            comps[f"p{pnum}_"],
            color="tab:green",
            lw=2,
            label=f"Peak {pnum} (p{pnum}_)",
        )
    ax1.legend()
    # ax1.plot(xnew, comps['p2_'], color='tab:orange', lw=2, label='Peak 2 (p2_)')
    # ax1.plot(xnew,(comps['{}'.format(fit_type)]))
    # ax1.plot(xnew,(comps['line_']))
    # ax1.plot(xnew,(comps['p1_{}'.format(fit_type)]+comps['line_']))
    # ax2.plot(x, result.residual)
    # ax2.set_title("residuals")
    result.plot_residuals(ax=ax2)
    plt.tight_layout()
    plt.show()

    return result


def plot_contour(csv_file, outfile, cmap="viridis", levels=100, figsize=(10, 6)):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Extract q and intensity values
    q = df.iloc[:, 0].values
    intensity = df.iloc[:, 1:].values  # shape: (n_q, n_columns)
    num_columns = intensity.shape[1]

    # X-axis: column indices
    x = np.arange(num_columns)
    y = q

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Plot contour
    plt.figure(figsize=figsize)
    contour = plt.contourf(X, Y, intensity, levels=levels, cmap=cmap)
    plt.xlabel("Column index")
    plt.ylabel("q [Å⁻¹]")
    # cbar = plt.colorbar(contour, label="Intensity")

    # Save figure
    plt.savefig(outfile + ".png", bbox_inches="tight")
    plt.savefig(outfile + ".pdf", bbox_inches="tight")
    plt.show()

    #     folder_text=wg.Text(value='/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/dev_2026-01-23')


#     folder_path=Path(folder_text.value)
#     ivqfiles = [file for file in os.listdir(folder_text.value) if f'{scantype}' in file]
#     file_list = wg.Dropdown(options=ivqfiles, description=f"{scantype} files")
#     #display(file_list)
#     # index1=wg.IntSlider(value=0)
#     # index2=wg.IntSlider(value=0)
#     fig, ax1 = plt.subplots(1,1,figsize=(6,3), dpi=100)
#     def update_list(folder):
#         file_list.options=[file for file in os.listdir(folder) if f'{scantype}' in file]
#     wg.interact(update_list,folder=folder_text)
#     def update_plot(filename):
#         remove_axes(fig)
#         ax1 = fig.add_subplot(1, 1, 1)
#         h5file=folder_path / filename

#         title=h5file.stem
#         with h5py.File(h5file) as data:
#             outfunc,title=parse_i07_giwaxs(data,title)
#             outfunc(data,title,figax=[fig,ax1])
#         fig.canvas.draw_idle()

#     wg.interact(update_plot, filename=file_list)


##==========plotly functions


def plot_ivsq_i07_plotly(data, title, log_y):
    # fig = go.Figure()
    range_color = wg.FloatRangeSlider(value=[5, 7.5], min=0, max=10.0)
    int_data, q_data = get_i07_i_q_data(data)
    while len(np.shape(q_data)) > 1:
        q_data = q_data[0]
    while len(np.shape(int_data)) > 1:
        int_data = int_data[0]

    fig = px.line(
        x=q_data, y=int_data, template="simple_white", title=title, log_y=log_y
    )
    fig.update_layout(xaxis_title="Q (A^-1)", yaxis_title="Intensity")
    # fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def plot_i07_list_plotly(dirpath: Path, scantype: str, title=None):

    folder_text = wg.Text(value=str(dirpath), layout=wg.Layout(width="100%"))

    # initial file list
    file_list = wg.Dropdown(
        options=get_files(folder_text.value, scantype), description=f"{scantype} files"
    )

    # --- Update list when folder changes ---
    def update_list(folder):
        if not os.path.exists(folder):
            file_list.options = ["folder not found"]
            return
        try:
            new_files = get_files(folder, scantype)
            file_list.options = new_files
        except FileNotFoundError:
            file_list.options = []

    wg.interact(update_list, folder=folder_text)
    logy_check = wg.Checkbox(value=False, description="log_y", disabled=False)

    def update_plot(filename, set_logy):
        if filename == "folder not found":
            return
        folder_path = Path(folder_text.value)
        h5file = folder_path / filename

        with h5py.File(h5file) as data:
            outfunc, title = parse_i07_giwaxs(data, h5file.stem)
            outfunc(data, title, log_y=set_logy)

    wg.interact(update_plot, filename=file_list, set_logy=logy_check)


def plot_exitmap_i07_plotly(data, title, log_y):
    # fig = go.Figure()
    map2d, exit_para, exit_perp = get_i07_exitmap_data(data)

    for item in [map2d, exit_para, exit_perp]:
        print(np.shape(item))

    while len(np.shape(map2d)) > 2:
        map2d = map2d[0]
    while len(np.shape(exit_para)) > 1:
        exit_para = exit_para[0]
    while len(np.shape(exit_perp)) > 1:
        exit_perp = exit_perp[0]
    print(f"mean ={np.mean(map2d)}")
    print(f"max = {np.max(map2d)}")
    fig = px.imshow(
        map2d,
        x=exit_para,
        y=exit_perp,
        template="simple_white",
        title=title,
        range_color=[0, 10],
    )
    fig.update_layout(xaxis_title="exit angle para", yaxis_title="exit angle perp")
    # fig.update_xaxes(rangeslider_visible=True)
    fig.show()
