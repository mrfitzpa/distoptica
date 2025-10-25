# -*- coding: utf-8 -*-
# Copyright 2025 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
"""This module contains helper functions for the Jupyter notebook located at
``<root>/examples/basic_usage.ipynb``, where ``<root>`` is the root of the
``distoptica`` repository.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For general array handling.
import numpy as np
import torch

# For creating hyperspy signals.
import hyperspy.signals
import hyperspy.axes

# For creating quiver plots.
import matplotlib.pyplot as plt



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["generate_undistorted_image_set_signal",
           "convert_torch_tensor_to_signal",
           "visualize_flow_field"]



_title_font_size = 15



def generate_undistorted_image_set_signal():
    kwargs = {"data": _generate_undistorted_image_set_signal_data(),
              "metadata": _generate_undistorted_image_set_signal_metadata()}
    undistorted_image_set_signal = hyperspy.signals.Signal2D(**kwargs)

    axes = _generate_undistorted_image_set_signal_axes()

    for axis_idx, axis in enumerate(axes):
        undistorted_image_set_signal.axes_manager[axis_idx].update_from(axis)
        undistorted_image_set_signal.axes_manager[axis_idx].name = axis.name

    return undistorted_image_set_signal



def _generate_undistorted_image_set_signal_data():
    signal_data_shape = _generate_undistorted_image_set_signal_data_shape()
    Y_dim, X_dim, v_dim, h_dim = signal_data_shape

    undistorted_image_supports = _generate_undistorted_image_supports(h_dim,
                                                                      v_dim)
    
    metadata = \
        _generate_undistorted_image_set_signal_metadata()
    max_pixel_vals_of_channels_of_undistorted_image = \
        metadata["max_pixel_vals_of_channels_of_undistorted_image"]

    kwargs = {"shape": signal_data_shape, "dtype": "float"}
    signal_data = np.zeros(**kwargs)

    for X_idx in range(X_dim):
        max_pixel_val_of_channel = \
            max_pixel_vals_of_channels_of_undistorted_image[X_idx]
        signal_data[:, X_idx] = \
            (undistorted_image_supports[:, X_idx] * max_pixel_val_of_channel)

    undistorted_image_set_signal_data = signal_data

    return undistorted_image_set_signal_data



def _generate_undistorted_image_set_signal_data_shape():
    undistorted_image_set_signal_data_shape = (3, 2, 140, 140)

    return undistorted_image_set_signal_data_shape



def _generate_undistorted_image_supports(h_dim, v_dim):
    signal_data_shape = _generate_undistorted_image_set_signal_data_shape()
    Y_dim, X_dim, _, _ = signal_data_shape

    metadata = \
        _generate_undistorted_image_set_signal_metadata()
    undistorted_disk_centers = \
        metadata["undistorted_disk_centers"]
    undistorted_disk_radius = \
        metadata["undistorted_disk_radius"]

    kwargs = {"shape": (Y_dim, X_dim, v_dim, h_dim), "dtype": "bool"}
    undistorted_image_supports = np.zeros(**kwargs)

    u_x, u_y = _generate_coord_meshgrid(h_dim, v_dim)

    for Y_idx in range(Y_dim):
        for X_idx in range(X_dim):
            u_x_c, u_y_c = undistorted_disk_centers[Y_idx]
            u_xy = np.sqrt((u_x-u_x_c)**2 + (u_y-u_y_c)**2)
            u_R = undistorted_disk_radius
            undistorted_image_supports[Y_idx, X_idx] = (u_xy <= u_R)
            
    return undistorted_image_supports



def _generate_coord_meshgrid(h_dim, v_dim):
    m_range = np.arange(h_dim)
    n_range = np.arange(v_dim)

    horizontal_coords_of_meshgrid = \
        (m_range + 0.5) / m_range.size
    vertical_coords_of_meshgrid = \
        1 - (n_range + 0.5) / n_range.size

    pair_of_1d_coord_arrays = (horizontal_coords_of_meshgrid,
                               vertical_coords_of_meshgrid)
    coord_meshgrid = np.meshgrid(*pair_of_1d_coord_arrays,
                                 indexing="xy")

    return coord_meshgrid



def _generate_undistorted_image_set_signal_metadata():
    metadata = {"General": \
                {"title": "Undistorted Image Set"},
                "Signal": \
                dict(),
                "undistorted_disk_centers": \
                _generate_undistorted_disk_centers(),
                "undistorted_disk_radius": \
                _generate_undistorted_disk_radius(),
                "max_pixel_vals_of_channels_of_undistorted_image": \
                _generate_max_pixel_vals_of_channels_of_undistorted_image()}

    undistorted_image_set_signal_metadata = metadata

    return undistorted_image_set_signal_metadata



def _generate_undistorted_disk_centers():
    undistorted_disk_centers = ((0.5, 0.7),
                                (0.5, 0.5),
                                (0.5, 0.3))

    return undistorted_disk_centers



def _generate_undistorted_disk_radius():
    undistorted_disk_radius = 1/6

    return undistorted_disk_radius



def _generate_max_pixel_vals_of_channels_of_undistorted_image():
    max_pixel_vals_of_channels_of_undistorted_image = (1, 3)

    return max_pixel_vals_of_channels_of_undistorted_image



def _generate_undistorted_image_set_signal_axes():
    signal_data_shape = _generate_undistorted_image_set_signal_data_shape()
    Y_dim, X_dim, v_dim, h_dim = signal_data_shape

    d_h = 1/h_dim
    d_v = -1/v_dim

    axes_sizes = (X_dim, Y_dim, h_dim, v_dim)
    axes_scales = (1, 1, d_h, d_v)
    axes_offsets = (0, 0, 0.5*d_h, 1+0.5*d_v)
    axes_names = ("$X$",
                  "$Y$",
                  "fractional horizontal coordinate",
                  "fractional vertical coordinate")

    axes = tuple()
    for axis_idx, _ in enumerate(axes_names):
        kwargs = {"size": axes_sizes[axis_idx],
                  "scale": axes_scales[axis_idx],
                  "offset": axes_offsets[axis_idx],
                  "name": axes_names[axis_idx]}
        axis = hyperspy.axes.UniformDataAxis(**kwargs)
        axes += (axis,)

    undistorted_image_set_signal_axes = axes

    return undistorted_image_set_signal_axes



def convert_torch_tensor_to_signal(torch_tensor, title):
    numpy_array = torch_tensor.detach().numpy()

    metadata = {"General": {"title": title}, "Signal": dict()}

    kwargs = {"data": numpy_array, "metadata": metadata}
    signal = hyperspy.signals.Signal2D(**kwargs)

    num_axes = len(numpy_array.shape)

    v_dim, h_dim = numpy_array.shape[-2:]

    d_h = 1/h_dim
    d_v = -1/v_dim

    axes_sizes = (h_dim, v_dim)
    axes_scales = (d_h, d_v)
    axes_offsets = (0.5*d_h, 1+0.5*d_v)
    axes_names = ("fractional horizontal coordinate",
                  "fractional vertical coordinate")

    if num_axes == 4:
        Y_dim, X_dim = numpy_array.shape[:2]
        axes_sizes = (X_dim, Y_dim) + axes_sizes
        axes_scales = (1, 1) + axes_scales
        axes_offsets = (0, 0) + axes_offsets
        axes_names = ("$X$", "$Y$") + axes_names

    for axis_idx in range(num_axes):
        signal.axes_manager[axis_idx].size = axes_sizes[axis_idx]
        signal.axes_manager[axis_idx].scale = axes_scales[axis_idx]
        signal.axes_manager[axis_idx].offset = axes_offsets[axis_idx]
        signal.axes_manager[axis_idx].name = axes_names[axis_idx]

    return signal



def visualize_flow_field(sampling_grid, flow_field, title):
    sampling_grid = (sampling_grid[0].numpy(), sampling_grid[1].numpy())
    flow_field = (flow_field[0].numpy(), flow_field[1].numpy())

    slice_step = 8
    quiver_kwargs = {"angles": "uv", "pivot": "middle", "scale_units": "width"}

    X = sampling_grid[0][::slice_step, ::slice_step]
    Y = sampling_grid[1][::slice_step, ::slice_step]

    fig, ax = plt.subplots()

    U = flow_field[0][::slice_step, ::slice_step]
    V = flow_field[1][::slice_step, ::slice_step]

    kwargs = quiver_kwargs
    ax.quiver(X, Y, U, V, **kwargs)

    _update_quiver_plot_title_and_axes(ax, title)
            
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

    return None



def _update_quiver_plot_title_and_axes(ax, title):
    _update_quiver_plot_title_and_axes_labels(ax, title)
    _update_quiver_plot_axes_ticks_and_spines(ax)

    return None



def _update_quiver_plot_title_and_axes_labels(ax, title):
    kwargs = {"label": title, "fontsize": _title_font_size}
    ax.set_title(**kwargs)

    kwargs = {"ax": ax,
              "x_label": "fractional horizontal coordinate",
              "y_label": "fractional vertical coordinate"}
    _update_axes_labels(**kwargs)

    return None



def _update_axes_labels(ax, x_label, y_label):
    axis_label_font_size = _title_font_size

    kwargs = {"xlabel": x_label, "fontsize": axis_label_font_size}
    ax.set_xlabel(**kwargs)

    kwargs = {"ylabel": y_label, "fontsize": axis_label_font_size}
    ax.set_ylabel(**kwargs)

    return None



def _update_quiver_plot_axes_ticks_and_spines(ax):
    kwargs = {"axis": "x",
              "which": "both",
              "bottom": False,
              "top": False,
              "labelbottom": False}
    ax.tick_params(**kwargs)

    kwargs = {"axis": "y",
              "which": "both",
              "left": False,
              "right": False,
              "labelleft": False}
    ax.tick_params(**kwargs)

    _update_axes_spines(ax)

    return None



def _update_axes_spines(ax):
    for side in ['top','bottom','left','right']:
        linewidth = 1.5
        ax.spines[side].set_linewidth(linewidth)

    return



###########################
## Define error messages ##
###########################
