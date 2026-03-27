import logging
import os
from pathlib import Path
from typing import List

import h5py
import ipywidgets as wg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import cm as mpl_cm
from IPython.display import display
from ipywidgets import fixed, interact
from matplotlib.colors import LogNorm, Normalize

from MINERVA_toolbox.processing import data_loader, result1d, result2d


def get_logger(folderpath, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:  # <- prevents duplicate handlers
        file_handler = logging.FileHandler(folderpath / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    # --- Silence third‑party libraries ---
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return logger


def plot_2d_map(loaded_data, loaded_axis, filename, fig, ax, logscale, axlabels):
    map2d = loaded_data
    q_para, q_perp = loaded_axis
    cm = "viridis"
    vmax, vmin = None, None
    limits = None
    extent = [q_para.min(), q_para.max(), q_perp.min(), q_perp.max()]
    if vmax is None:
        vmax = np.mean(map2d) + 2 * np.std(map2d)
    if vmin is None:
        vmin = np.max(np.array([map2d.min(), 0.01]))
    if logscale:
        normvals = LogNorm(vmin=vmin, vmax=vmax)
    else:
        normvals = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        map2d,
        cmap=cm,
        extent=extent,
        norm=normvals,
        aspect=1,
    )
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_title(filename)
    ax.set_xlabel(axlabels[0])
    ax.set_ylabel(axlabels[1])
    fig.colorbar(im, ax=ax, location="right")
    ax.set_aspect("auto")
    return fig


def plot_1d_profile(
    loaded_data, loaded_axis, filename, fig, ax, logscale, axlabels, label=None
):
    """
    Plot a 1D intensity profile.
    """
    y_data = loaded_data
    x_data = loaded_axis
    ax.plot(x_data, y_data, label=label)
    ax.set_xlabel(axlabels[1])
    ax.set_ylabel(axlabels[0])
    ax.set_title(filename)

    qmin, qmax = None, None
    if logscale:
        ax.set_yscale("log")
    if qmin is not None and qmax is not None:
        ax.set_xlim(qmin, qmax)
    if label:
        ax.set_title("")
        ax.legend(loc="upper left", bbox_to_anchor=(0.1, 1.15))
    return fig


def get_1d_data(data, paths, dataind, axisind):
    y_out = data[paths[0]][dataind]
    x_out = data[paths[1]][axisind]
    return y_out, x_out


def get_2d_data(data, paths, dataind, axisind):
    dataout = data[paths[0]][dataind]
    para_out = data[paths[1]][axisind]
    perp_out = data[paths[2]][axisind]
    return dataout, np.array([para_out, perp_out])


def check_shape(inshape, expected_shape, index1, index2):

    if len(inshape) == expected_shape:
        ind1max = np.int32(0)
        ind2max = np.int32(0)
        outind = slice(None)

    if len(inshape) == expected_shape + 1:
        ind1max = inshape[0] - 1
        ind2max = np.int32(0)
        if index1 > ind1max:
            index1 = np.int32(0)
        outind = (index1, slice(None))

    if len(inshape) == expected_shape + 2:
        ind1max = inshape[0] - 1
        ind2max = inshape[1] - 1
        if index1 > ind1max:
            index1 = np.int32(0)
        if index2 > ind2max:
            index2 = np.int32(0)
        outind = (index1, index2, slice(None))

    return outind, index1, index2, ind1max, ind2max


def reset_plots():
    plt.close("all")
    wg.Widget.close_all()


class individual_plotter:
    def __init__(self, datafolder: str):
        reset_plots()
        self.datafolder = Path(datafolder)
        self.plot_list_data()

    def plot_list_data(self):
        list_plotter_multi = ind_list_plotter(self.datafolder)
        list_plotter_multi.create_plot()


class ind_list_plotter:
    def __init__(self, folderpath):
        self._updating = False
        self.folderpath = folderpath
        self.loader = data_loader(folderpath)

        # self.logger = self.logger = get_logger(
        #     self.folderpath, name=f"ind_list_plotter_{id(self)}"
        # )

    def get_files(self):
        return [f for f in os.listdir(self.folderpath) if self.scantype in f]

    def set_plot_callback(self):
        callback_dict = {
            "IvsQ": self._plot_ivsq,
            "Qmap": self._plot_qmap,
            "exitmap": self._plot_exitmap,
        }
        self._plot_callback = callback_dict[self.scantype]

    def set_dataloader(self):
        loaders_dict = {
            "IvsQ": self.loader.get_ivsq,
            "Qmap": self.loader.get_qmap,
            "exitmap": self.loader.get_exitmap,
        }
        self._dataloader = loaders_dict[self.scantype]

    def set_scantype(self, scantype):
        self.scantype = scantype
        self.filelist = self.get_files()
        self.set_plot_callback()
        self.set_dataloader()

    def _plot_qmap(self, data_result: result2d, filename: str, fig, ax, logscale: bool):
        axlabels = ["$_{para}$  [Å⁻¹]", "$q_{perp}$  [Å⁻¹]"]
        return plot_2d_map(
            data_result.data,
            [data_result.x_axis, data_result.y_axis],
            filename,
            fig,
            ax,
            logscale,
            axlabels,
        )

    def _plot_exitmap(
        self, data_result: result2d, filename: str, fig, ax, logscale: bool
    ):

        axlabels = ["$exitangle_{para}$  [deg]", "$exitangle_{perp}$  [deg]"]
        return plot_2d_map(
            data_result.data,
            [data_result.x_axis, -1 * data_result.y_axis],
            filename,
            fig,
            ax,
            logscale,
            axlabels,
        )

    def _plot_ivsq(self, data_result: result1d, filename, fig, ax, logscale):
        axlabels = ["Intensity", "Q (A^-1)"]
        return plot_1d_profile(
            data_result.data,
            data_result.x_axis,
            filename,
            fig,
            ax,
            logscale,
            axlabels,
        )

    def create_plot(self):
        # out = Output()  # persistent output area
        fig, ax = plt.subplots()
        self.set_scantype("IvsQ")
        files = wg.Dropdown(
            options=self.filelist,
            description=f"{self.scantype} files",
        )
        self.index1 = wg.IntSlider(max=10)
        self.index2 = wg.IntSlider(max=10)
        self.index1.style.handle_color = "lightblue"
        self.index2.style.handle_color = "lightblue"

        @interact(
            scantype=wg.Dropdown(
                options=["IvsQ", "Qmap", "exitmap"],
                description="scantype",
            ),
            filename=files,
            logscale=wg.Checkbox(value=False),
            fig=fixed(fig),
            ax=fixed(ax),
            index1=self.index1,
            index2=self.index2,
        )
        def do_plot(scantype, filename, logscale, index1, index2):
            if self._updating:
                return
            # out.clear_output()
            self._updating = True
            self.set_scantype(scantype)
            files.options = self.filelist
            self._updating = False
            fig.clear()
            ax = fig.add_subplot(111)
            self.index1.layout.visibility = "hidden"
            self.index2.layout.visibility = "hidden"

            # SKIP invalid combinations when interact misfires
            if filename not in self.filelist:
                filename = files.options[0]
            # h5file = self.folderpath / filename
            # with h5py.File(h5file) as data:
            result, index1, index2, indmax1, indmax2 = self._dataloader(
                filename, index1, index2
            )
            self.index1.max = indmax1
            self.index2.max = indmax2
            self.index1.layout.visibility = "hidden" if indmax1 == 0 else "visible"
            self.index2.layout.visibility = "hidden" if indmax2 == 0 else "visible"
            # your existing plot functions already draw correctly
            self._plot_callback(result, filename, fig, ax, logscale)

        # display(VBox([out]))


class comparison_plotter:
    def __init__(
        self,
        datafolder: str,
    ):
        reset_plots()
        self.datafolder = Path(datafolder)

    def plot_files(self, filenames, index1vals, index2vals, logscale):
        combo_file_plotter = combo_plotter(self.datafolder)
        combo_file_plotter.plot_files(filenames, index1vals, index2vals, logscale)


class combo_plotter:
    def __init__(
        self,
        datafolder: str,
    ):
        reset_plots()
        self.datafolder = Path(datafolder)
        self.loader = data_loader(datafolder)

    def plot_files(
        self,
        filenames: str,
        index1vals: np.ndarray | None = None,
        index2vals: np.ndarray | None = None,
        logscale=False,
    ):
        if all(["IvsQ" in file for file in filenames]):
            self.plot_ivsq(filenames, index1vals, index2vals, logscale)
        elif all(["Qmap" in file for file in filenames]):
            self.plot_qmap(filenames, index1vals, index2vals, logscale)
        elif all(["exitmap" in file for file in filenames]):
            self.plot_exitmap(filenames, index1vals, index2vals, logscale)

    def plot_ivsq(
        self,
        filenames: list,
        index1vals: np.ndarray | None = None,
        index2vals: np.ndarray | None = None,
        logscale: bool = False,
    ):
        # plotpaths = [self.datafolder / name for name in filenames]
        axlabels = ["Intensity", "Q (A^-1)"]
        if index1vals is None:
            index1vals = np.zeros(len(filenames)).astype(np.int32)
        if index2vals is None:
            index2vals = np.zeros(len(filenames)).astype(np.int32)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        for file_num, file in enumerate(filenames):
            index1 = index1vals[file_num]
            index2 = index2vals[file_num]
            result, *_ = self.loader.get_ivsq(file, index1, index2)

            plot_1d_profile(
                result.data,
                result.x_axis,
                file,
                self.fig,
                self.ax,
                logscale,
                axlabels,
                label=f"{filenames[file_num]} {index1} {index2}",
            )

        # fig.canvas.draw_idle()

    def plot_exitmap(
        self,
        filenames: list,
        index1vals: np.ndarray | None = None,
        index2vals: np.ndarray | None = None,
        logscale: bool = False,
    ):
        axlabels = ["$exitangle_{para}$  [deg]", "$exitangle_{perp}$  [deg]"]
        if index1vals is None:
            index1vals = np.zeros(len(filenames)).astype(np.int32)
        if index2vals is None:
            index2vals = np.zeros(len(filenames)).astype(np.int32)
        fig, ax = plt.subplots(figsize=(10, 5))
        for file_num, filename in enumerate(filenames):
            index1 = index1vals[file_num]
            index2 = index2vals[file_num]

            result, *_ = self.loader.get_exitmap(filename, index1, index2)
            plot_2d_map(
                result.data,
                [result.x_axis, result.y_axis],
                filename,
                fig,
                ax,
                logscale,
                axlabels,
            )

    def plot_qmap(
        self,
        filenames: list,
        index1vals: np.ndarray | None = None,
        index2vals: np.ndarray | None = None,
        logscale: bool = False,
    ):

        axlabels = ["$q_{para}$  [Å⁻¹]", "$q_{perp}$  [Å⁻¹]"]
        if index1vals is None:
            index1vals = np.zeros(len(filenames)).astype(np.int32)
        if index2vals is None:
            index2vals = np.zeros(len(filenames)).astype(np.int32)
        row_num = int(np.ceil(len(filenames) / 2))
        fig, axs = plt.subplots(row_num, 2, figsize=(10, 5 * row_num))
        axlist = axs.flatten()
        for file_num, filename in enumerate(filenames):
            index1 = index1vals[file_num]
            index2 = index2vals[file_num]

            result, *_ = self.loader.get_qmap(filename, index1, index2)
            plot_2d_map(
                result.data,
                [result.x_axis, result.y_axis],
                filename,
                fig,
                axlist[file_num],
                logscale,
                axlabels,
            )


# class data_loader:
#     def __init__(self, datafolder: str):
#         self.datafolder = Path(datafolder)

#     def get_ivsq(self, filenames: list):
#         outvals = dict()
#         for i, testivqfile in enumerate(filenames):
#             h5data = h5py.File(self.datafolder / testivqfile)
#             int_vals, q_vals = get_i07_i_q_data(h5data)
#             outvals[testivqfile] = [int_vals, q_vals]
#         return outvals


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
    elif "exitmap" in startkeys:
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


def compare_qmap_list(paths: List[Path], logscale=False):
    row_num = int(np.ceil(len(paths) / 2))
    fig, axs = plt.subplots(row_num, 2, figsize=(10, 5 * row_num))
    axlist = axs.flatten()
    for i, path in enumerate(paths):
        with h5py.File(path) as h5data:
            plot_qmap_i07(h5data, path.name, (fig, axlist[i]))
    # plt.tight_layout()
    fig.canvas.draw_idle()


def compare_exitmap_list(paths: List[Path], logscale=False):
    row_num = int(np.ceil(len(paths) / 2))
    fig, axs = plt.subplots(row_num, 2, figsize=(10, 5 * row_num))
    axlist = axs.flatten()
    for i, path in enumerate(paths):
        with h5py.File(path) as h5data:
            plot_exitmap_i07(h5data, path.name, (fig, axlist[i]))
    # plt.tight_layout()
    fig.canvas.draw_idle()


def compare_ivq_list(paths: List[Path], logscale=False):
    fig, ax = plt.subplots(figsize=(10, 5))

    for path in paths:
        with h5py.File(path) as h5data:
            plot_ivq_i07(h5data, path.name, (fig, ax), log=logscale)

    if logscale:
        ax.set_yscale("log")

    fig.canvas.draw_idle()


def plot_i07_list(dirpath: Path, scantype: str, title=None):

    folder_text = wg.Text(value=str(dirpath), layout=wg.Layout(width="100%"))

    file_list = wg.Dropdown(
        options=get_files(folder_text.value, scantype), description=f"{scantype} files"
    )

    def update_list(folder):
        if not os.path.exists(folder):
            file_list.options = ["folder not found"]
            return
        try:
            files = get_files(folder, scantype)
            file_list.options = files

            if files:
                file_list.value = None  # force trait change event
                file_list.value = files[0]

        except FileNotFoundError:
            file_list.options = []

    folder_interact = wg.interactive_output(update_list, {"folder": folder_text})

    def update_plot(filename, logscale):

        if filename in (None, "", "folder not found"):
            return

        fig, ax = plt.subplots()

        h5file = Path(folder_text.value) / filename

        with h5py.File(h5file) as data:
            outfunc, title = parse_i07_giwaxs(data, h5file.stem)
            fig, ax = outfunc(data, title, figax=[fig, ax], log=logscale)
        return fig

    log_box = wg.Checkbox(value=False, description="Log y-scale")
    plot_interact = wg.interactive_output(
        update_plot, {"filename": file_list, "logscale": log_box}
    )

    ui = wg.VBox([folder_interact, file_list, log_box, plot_interact])

    display(ui)


def plot_ivq_i07(data, title, figax, log):
    int_data, q_data = get_i07_i_q_data(data)
    while len(np.shape(q_data)) > 1:
        q_data = q_data[0]
    while len(np.shape(int_data)) > 1:
        int_data = int_data[0]
    # wg.interact(plot_1D_profile,q=q_data,intensity=int_data,title=title,ax=ax)
    fig, ax = plot_1D_profile(
        q=q_data, intensity=int_data, title=title, figax=figax, logscale=log
    )

    return fig, ax


def plot_qmap_i07(data, title, figax, log=False):
    map2d, q_para, q_perp = get_i07_qmap_data(data)
    while len(np.shape(map2d)) > 2:
        map2d = map2d[0]
    while len(np.shape(q_para)) > 1:
        q_para = q_para[0]
    while len(np.shape(q_perp)) > 1:
        q_perp = q_perp[0]

    if "map_para_unit" not in data["qpara_qperp"].keys():
        fig, ax = plot_2D_map(map2d, q_para, q_perp, title=title, figax=figax)
    else:
        axlabel_para = data["qpara_qperp/map_para_unit"][()].decode()
        axlabel_perp = data["qpara_qperp/map_perp_unit"][()].decode()
        axlabels = [axlabel_para, axlabel_perp]

        fig, ax = plot_2D_map(
            map2d, q_para, q_perp, labels=axlabels, title=title, figax=figax
        )
    return fig, ax


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
    return fig, ax


def plot_exitmap_i07(data, title, figax, log=False):
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


def plot_chi_profile(
    chi,
    intensity,
    chi_min=None,
    chi_max=None,
    logscale=False,
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


def plot_i07_list_old(dirpath: Path, scantype: str, title=None):

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

    def update_plot(filename, logscale):
        if filename == "folder not found":
            return
        remove_axes(fig)
        ax1 = fig.add_subplot(1, 1, 1)
        folder_path = Path(folder_text.value)
        h5file = folder_path / filename

        with h5py.File(h5file) as data:
            outfunc, title = parse_i07_giwaxs(data, h5file.stem)
            outfunc(data, title, figax=[fig, ax1], log=logscale)
            plt.show()

        fig.canvas.draw_idle()

    log_box = wg.Checkbox(value=False, description="Log y-scale")
    wg.interact(update_plot, filename=file_list, logscale=log_box)


def plot_i07_list_colab(dirpath: Path, scantype: str, title=None):
    """
    Colab-friendly version:
    - Avoids %matplotlib widget/ipympl
    - Uses ipywidgets.Output() for reliable redraws
    - Avoids wg.interact auto-display (returns a VBox you can display())
    - Observes changes from Text/Dropdown/Checkbox instead of using interact
    """

    # --- Controls ---
    folder_text = wg.Text(
        value=str(dirpath), description="Folder", layout=wg.Layout(width="100%")
    )

    # initial file list
    try:
        initial_options = get_files(folder_text.value, scantype)
        if not initial_options:
            initial_options = ["<no files>"]
    except FileNotFoundError:
        initial_options = ["<folder not found>"]

    file_list = wg.Dropdown(
        options=initial_options,
        description=f"{scantype} files",
        layout=wg.Layout(width="60%"),
    )
    log_box = wg.Checkbox(value=False, description="Log y-scale")

    # Where plots go
    out = wg.Output(layout=wg.Layout(border="1px solid #ddd"))

    # --- Helpers ---
    def _safe_files(folder_str: str):
        """Refresh the file dropdown safely for a new folder."""
        if not os.path.exists(folder_str):
            return ["<folder not found>"]
        try:
            files = get_files(folder_str, scantype)
            return files if files else ["<no files>"]
        except FileNotFoundError:
            return ["<no files>"]

    # --- Observers / Callbacks ---
    def on_folder_change(change):
        new_folder = change["new"]
        file_list.options = _safe_files(new_folder)
        # If the new options are non-empty strings, set value to first
        if isinstance(file_list.options, (list, tuple)) and file_list.options:
            file_list.value = file_list.options[0]

    folder_text.observe(on_folder_change, names="value")

    def draw_plot():
        """(Re)draw the plot for current selections inside Output()."""
        filename = file_list.value
        if filename in ("<no files>", "<folder not found>", None):
            with out:
                out.clear_output(wait=True)
                print("No valid file to display.")
            return

        folder_path = Path(folder_text.value)
        h5file = folder_path / filename

        with out:
            out.clear_output(wait=True)
            # fresh figure for each draw (most reliable in Colab)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
            try:
                with h5py.File(h5file, "r") as data:
                    outfunc, the_title = parse_i07_giwaxs(data, h5file.stem)
                    outfunc(
                        data,
                        the_title if title is None else title,
                        figax=[fig, ax],
                        log=log_box.value,
                    )
                plt.tight_layout()
                plt.show()
            except FileNotFoundError:
                print(f"File not found: {h5file}")
            except Exception as e:
                # Don't crash the widget; show the error in the output pane
                print(f"Error while plotting {h5file}:\n{e}")

    def on_file_change(change):
        draw_plot()

    def on_log_change(change):
        draw_plot()

    file_list.observe(on_file_change, names="value")
    log_box.observe(on_log_change, names="value")

    # Initial draw (if possible)
    draw_plot()

    # --- Layout ---
    controls = wg.HBox([file_list, log_box])
    ui = wg.VBox([folder_text, controls, out])
    return ui


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

    def update(vmin, vmax, logscale):
        plt.figure(figsize=(7, 7))

        if logscale:
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
        update, {"vmin": vmin_slider, "vmax": vmax_slider, "logscale": log_box}
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


if __name__ == "__main__":
    ind_plotter = individual_plotter(
        datafolder="/dls/science/users/rpy65944/I07_work/MINERVA_analysis/MINERVA_training/example_data"
    )
    ind_plotter.plot_qmap()


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


# def plot_ivsq_i07_plotly(data, title, log_y):
#     # fig = go.Figure()
#     range_color = wg.FloatRangeSlider(value=[5, 7.5], min=0, max=10.0)
#     int_data, q_data = get_i07_i_q_data(data)
#     while len(np.shape(q_data)) > 1:
#         q_data = q_data[0]
#     while len(np.shape(int_data)) > 1:
#         int_data = int_data[0]

#     fig = px.line(
#         x=q_data, y=int_data, template="simple_white", title=title, log_y=log_y
#     )
#     fig.update_layout(xaxis_title="Q (A^-1)", yaxis_title="Intensity")
#     # fig.update_xaxes(rangeslider_visible=True)
#     fig.show()


# def plot_i07_list_plotly(dirpath: Path, scantype: str, title=None):

#     folder_text = wg.Text(value=str(dirpath), layout=wg.Layout(width="100%"))

#     # initial file list
#     file_list = wg.Dropdown(
#         options=get_files(folder_text.value, scantype), description=f"{scantype} files"
#     )

#     # --- Update list when folder changes ---
#     def update_list(folder):
#         if not os.path.exists(folder):
#             file_list.options = ["folder not found"]
#             return
#         try:
#             new_files = get_files(folder, scantype)
#             file_list.options = new_files
#         except FileNotFoundError:
#             file_list.options = []

#     wg.interact(update_list, folder=folder_text)
#     logy_check = wg.Checkbox(value=False, description="log_y", disabled=False)

#     def update_plot(filename, set_logy):
#         if filename == "folder not found":
#             return
#         folder_path = Path(folder_text.value)
#         h5file = folder_path / filename

#         with h5py.File(h5file) as data:
#             outfunc, title = parse_i07_giwaxs(data, h5file.stem)
#             outfunc(data, title, log_y=set_logy)

#     wg.interact(update_plot, filename=file_list, set_logy=logy_check)


# def plot_exitmap_i07_plotly(data, title, log_y):
#     # fig = go.Figure()
#     map2d, exit_para, exit_perp = get_i07_exitmap_data(data)

#     for item in [map2d, exit_para, exit_perp]:
#         print(np.shape(item))

#     while len(np.shape(map2d)) > 2:
#         map2d = map2d[0]
#     while len(np.shape(exit_para)) > 1:
#         exit_para = exit_para[0]
#     while len(np.shape(exit_perp)) > 1:
#         exit_perp = exit_perp[0]
#     print(f"mean ={np.mean(map2d)}")
#     print(f"max = {np.max(map2d)}")
#     fig = px.imshow(
#         map2d,
#         x=exit_para,
#         y=exit_perp,
#         template="simple_white",
#         title=title,
#         range_color=[0, 10],
#     )
#     fig.update_layout(xaxis_title="exit angle para", yaxis_title="exit angle perp")
#     # fig.update_xaxes(rangeslider_visible=True)
#     fig.show()


class dummy_plotter:
    def __init__(self, datafolder: str):
        self.datafolder = Path(datafolder)

    def plot_ivsq(self, option_values):
        list_plot = DummyListPlotter(option_values, scantype="IvsQ")
        list_plot.create_plot()


class DummyListPlotter:
    def __init__(self, options, scantype):
        self.scantype = scantype
        self.options = options

    def _plot_callback(self, number):

        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()
            return
        fig, ax = self.fig, self.ax
        # Plot something simple
        ax.clear()
        ax.bar(2, number)
        ax.set_title(f"{self.scantype} -> value {number}")
        ax.set_xlabel("Q (A^-1)")

        # ipympl: request redraw
        # fig.canvas.draw_idle()
        return fig

    def create_plot(self):
        # Create a fresh figure for ipympl (important)

        # decorator-created UI
        @interact
        def do_plot(
            number=wg.Dropdown(options=self.options, description=f"{self.scantype}:"),
        ):
            self._plot_callback(number)


class dummy_list_plotter:
    def __init__(self, scantype: str):
        self.type = scantype

    def create_plot(self):
        # Dropdown
        self.file_list = wg.Dropdown(
            options=np.arange(0, 5, 1), description=f"{self.type} files"
        )

        # Output area for the figure
        self.out = wg.Output()

        # Create initial figure
        with self.out:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("testing start")

        # Link callback
        self.file_list.observe(self.update_plot, names="value")

        # Display UI
        display(wg.VBox([self.file_list, self.out]))

        # Trigger initial plot
        self.update_plot({"new": self.file_list.value})

    def update_plot(self, change):
        number = change["new"]
        if number is None:
            return

        # Update inside output area
        with self.out:
            self.ax.clear()
            self.ax.bar(2, number)
            self.ax.set_xlabel("Q (A^-1)")
            self.fig.canvas.draw_idle()


def plot_dummy_list(datafolder, type):

    file_list = wg.Dropdown(options=np.arange(0, 5, 1), description=f"{type} files")

    # # Create a separate figure for THIS instance
    # fig, ax1 = plt.subplots(1, 1, figsize=(6, 3), dpi=100)

    def update_plot(number):

        if number in (None, "", "folder not found"):
            return

        fig, ax = plt.subplots()
        fig, ax = plot_dummy_vals(number, [fig, ax])
        fig.canvas.draw_idle()

    plot_interact = wg.interactive_output(update_plot, {"number": file_list})

    ui = wg.VBox([file_list, plot_interact])

    display(ui)


def plot_dummy_vals(number, figax):
    if figax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig, ax = figax
    ax.bar(2, number)
    ax.set_xlabel("Q (A^-1)")
    return fig, ax
