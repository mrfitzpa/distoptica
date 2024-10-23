# -*- coding: utf-8 -*-
# Copyright 2024 Matthew Fitzpatrick.
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
r"""Contains tests for the root of the package :mod:`distoptica`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For operations related to unit tests.
import pytest

# For accessing attributes of functions.
import inspect



# For general array handling.
import numpy as np
import torch



# For modelling optical distortions.
import distoptica



##################################
## Define classes and functions ##
##################################

def generate_undistorted_images():
    Y_dim, X_dim, v_dim, h_dim = generate_undistorted_images_shape()

    undistorted_image_supports = generate_undistorted_image_supports(h_dim, 
                                                                     v_dim)

    metadata = \
        generate_undistorted_images_metadata()
    max_pixel_vals_of_channels_of_undistorted_image = \
        metadata["max_pixel_vals_of_channels_of_undistorted_image"]

    kwargs = {"shape": (Y_dim, X_dim, v_dim, h_dim), "dtype": "float"}
    undistorted_images = np.zeros(**kwargs)

    for X_idx in range(X_dim):
        max_pixel_val_of_channel = \
            max_pixel_vals_of_channels_of_undistorted_image[X_idx]
        undistorted_images[:, X_idx] = \
            (undistorted_image_supports[:, X_idx] * max_pixel_val_of_channel)

    return undistorted_images



def generate_undistorted_images_shape():
    undistorted_images_shape = (3, 2, 140, 131)
    
    return undistorted_images_shape



def generate_undistorted_image_supports(h_dim, v_dim):
    Y_dim, X_dim, _, _ = generate_undistorted_images_shape()

    metadata = \
        generate_undistorted_images_metadata()
    undistorted_disk_centers = \
        metadata["undistorted_disk_centers"]
    undistorted_disk_radius = \
        metadata["undistorted_disk_radius"]

    kwargs = {"shape": (Y_dim, X_dim, v_dim, h_dim), "dtype": "bool"}
    undistorted_image_supports = np.zeros(**kwargs)

    u_x, u_y = generate_coord_meshgrid(h_dim, v_dim)

    for Y_idx in range(Y_dim):
        for X_idx in range(X_dim):
            u_x_c, u_y_c = undistorted_disk_centers[Y_idx]
            u_xy = np.sqrt((u_x-u_x_c)**2 + (u_y-u_y_c)**2)
            u_R = undistorted_disk_radius
            undistorted_image_supports[Y_idx, X_idx] = (u_xy <= u_R)

    return undistorted_image_supports



def generate_coord_meshgrid(h_dim, v_dim):
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



def generate_undistorted_images_metadata():
    metadata = {"undistorted_disk_centers": \
                generate_undistorted_disk_centers(),
                "undistorted_disk_radius": \
                generate_undistorted_disk_radius(), 
                "max_pixel_vals_of_channels_of_undistorted_image": \
                generate_max_pixel_vals_of_channels_of_undistorted_image()}

    undistorted_images_metadata = metadata

    return undistorted_images_metadata



def generate_undistorted_disk_centers():
    elliptical_distortion_center = generate_elliptical_distortion_center()
    x_c_D, y_c_D = elliptical_distortion_center
    
    undistorted_disk_centers = ((x_c_D, y_c_D),
                                (x_c_D, 0.5),
                                (x_c_D, 1-y_c_D))

    return undistorted_disk_centers



def generate_elliptical_distortion_center():
    elliptical_distortion_center = (0.5, 0.7)

    return elliptical_distortion_center



def generate_undistorted_disk_radius():
    undistorted_disk_radius = 1/6

    return undistorted_disk_radius



def generate_max_pixel_vals_of_channels_of_undistorted_image():
    max_pixel_vals_of_channels_of_undistorted_image = (1, 3)

    return max_pixel_vals_of_channels_of_undistorted_image



def generate_distorted_images():
    Y_dim, X_dim, v_dim, h_dim = generate_distorted_images_shape()

    distorted_image_supports = generate_distorted_image_supports(h_dim, 
                                                                 v_dim)

    metadata = \
        generate_distorted_images_metadata()
    max_pixel_vals_of_channels_of_distorted_image = \
        metadata["max_pixel_vals_of_channels_of_distorted_image"]

    kwargs = {"shape": (Y_dim, X_dim, v_dim, h_dim), "dtype": "float"}
    distorted_images = np.zeros(**kwargs)

    for X_idx in range(X_dim):
        max_pixel_val_of_channel = \
            max_pixel_vals_of_channels_of_distorted_image[X_idx]
        distorted_images[:, X_idx] = \
            distorted_image_supports[:, X_idx] * max_pixel_val_of_channel

    return distorted_images



def generate_distorted_images_shape():
    distorted_images_shape = generate_undistorted_images_shape()
    
    return distorted_images_shape



def generate_distorted_image_supports(h_dim, v_dim):
    Y_dim, X_dim, _, _ = generate_distorted_images_shape()

    metadata = \
        generate_distorted_images_metadata()
    distorted_disk_centers = \
        metadata["distorted_disk_centers"]
    undistorted_disk_radius = \
        metadata["undistorted_disk_radius"]
    elliptical_distortion_vector  = \
        metadata["elliptical_distortion_vector"]

    kwargs = {"shape": (Y_dim, X_dim, v_dim, h_dim), "dtype": "bool"}
    distorted_image_supports = np.zeros(**kwargs)

    q_x, q_y = generate_coord_meshgrid(h_dim, v_dim)

    for Y_idx in range(Y_dim):
        for X_idx in range(X_dim):
            q_x_c, q_y_c = distorted_disk_centers[Y_idx]
            u_R = undistorted_disk_radius
            A = np.linalg.norm(elliptical_distortion_vector)

            distorted_image_supports[Y_idx, X_idx] = \
                ((((q_x-q_x_c)/(u_R-A*u_R))**2 
                  + ((q_y-q_y_c)/(u_R+A*u_R))**2) <= 1)

    return distorted_image_supports



def generate_distorted_images_metadata():
    metadata = {"elliptical_distortion_vector": \
                generate_elliptical_distortion_vector(),
                "distorted_disk_centers": \
                generate_distorted_disk_centers(),
                "undistorted_disk_radius": \
                generate_undistorted_disk_radius(), 
                "max_pixel_vals_of_channels_of_distorted_image": \
                generate_max_pixel_vals_of_channels_of_distorted_image()}
        
    distorted_images_metadata = metadata
    
    return distorted_images_metadata



def generate_elliptical_distortion_vector():
    amplitude = 0.2
    phase = np.pi/2
    elliptical_distortion_vector = (amplitude*np.cos(2*phase).item(), 
                                    amplitude*np.sin(2*phase).item())

    return elliptical_distortion_vector



def generate_distorted_disk_centers():
    undistorted_disk_centers = generate_undistorted_disk_centers()

    elliptical_distortion_center = \
        generate_elliptical_distortion_center()
    elliptical_distortion_vector = \
        generate_elliptical_distortion_vector()
    
    x_c_D, y_c_D = elliptical_distortion_center
    A = np.linalg.norm(elliptical_distortion_vector)

    distorted_disk_centers = tuple()
    for undistorted_disk_center in undistorted_disk_centers:
        u_x_c, u_y_c = undistorted_disk_center
        u_r = np.sqrt((u_x_c-x_c_D)**2 + (u_y_c-y_c_D)**2)
        q_x_c = u_x_c
        q_y_c = (y_c_D + np.sign(u_y_c-y_c_D)*(u_r+A*u_r)).item()
        distorted_disk_centers += ((q_x_c, q_y_c),)

    return distorted_disk_centers



def generate_max_pixel_vals_of_channels_of_distorted_image():
    max_pixel_vals_of_channels_of_distorted_image = \
        generate_max_pixel_vals_of_channels_of_undistorted_image()

    return max_pixel_vals_of_channels_of_distorted_image



def generate_standard_coord_transform_params_1_ctor_params():
    center = (0.49, 0.55)
    
    quadratic_radial_distortion_amplitude = -0.8

    elliptical_distortion_vector = \
        generate_elliptical_distortion_vector()

    spiral_distortion_amplitude = -1.5

    amplitude = 0.42
    phase = 4*np.pi/3
    parabolic_distortion_vector = (amplitude*np.cos(phase), 
                                   amplitude*np.sin(phase))
    
    standard_coord_transform_params_ctor_params = \
        {"center": \
         center,
         "quadratic_radial_distortion_amplitude": \
         quadratic_radial_distortion_amplitude,
         "elliptical_distortion_vector": \
         elliptical_distortion_vector,
         "spiral_distortion_amplitude": \
         spiral_distortion_amplitude,
         "parabolic_distortion_vector": \
         parabolic_distortion_vector, 
         "skip_validation_and_conversion": \
         False}

    return standard_coord_transform_params_ctor_params



def generate_standard_coord_transform_params_2_ctor_params():
    center = (0.55, 0.5)
    
    quadratic_radial_distortion_amplitude = -0.6

    elliptical_distortion_vector = (0, 0)

    spiral_distortion_amplitude = 0

    amplitude = 0
    phase = 0
    parabolic_distortion_vector = (amplitude*np.cos(phase), 
                                   amplitude*np.sin(phase))
    
    standard_coord_transform_params_ctor_params = \
        {"center": \
         center,
         "quadratic_radial_distortion_amplitude": \
         quadratic_radial_distortion_amplitude,
         "elliptical_distortion_vector": \
         elliptical_distortion_vector,
         "spiral_distortion_amplitude": \
         spiral_distortion_amplitude,
         "parabolic_distortion_vector": \
         parabolic_distortion_vector, 
         "skip_validation_and_conversion": \
         False}

    return standard_coord_transform_params_ctor_params



def generate_standard_coord_transform_params_3_ctor_params():
    center = (0.5, 0.5)
    
    quadratic_radial_distortion_amplitude = -1

    elliptical_distortion_vector = (0, 0)

    spiral_distortion_amplitude = 0

    amplitude = 0
    phase = 0
    parabolic_distortion_vector = (amplitude*np.cos(phase), 
                                   amplitude*np.sin(phase))
    
    standard_coord_transform_params_ctor_params = \
        {"center": \
         center,
         "quadratic_radial_distortion_amplitude": \
         quadratic_radial_distortion_amplitude,
         "elliptical_distortion_vector": \
         elliptical_distortion_vector,
         "spiral_distortion_amplitude": \
         spiral_distortion_amplitude,
         "parabolic_distortion_vector": \
         parabolic_distortion_vector, 
         "skip_validation_and_conversion": \
         False}

    return standard_coord_transform_params_ctor_params



def generate_standard_coord_transform_params_4_ctor_params():
    center = generate_undistorted_disk_centers()[0]
    
    quadratic_radial_distortion_amplitude = 0

    elliptical_distortion_vector = \
        generate_elliptical_distortion_vector()

    spiral_distortion_amplitude = 0

    amplitude = 0
    phase = 0
    parabolic_distortion_vector = (amplitude*np.cos(phase), 
                                   amplitude*np.sin(phase))
    
    standard_coord_transform_params_ctor_params = \
        {"center": \
         center,
         "quadratic_radial_distortion_amplitude": \
         quadratic_radial_distortion_amplitude,
         "elliptical_distortion_vector": \
         elliptical_distortion_vector,
         "spiral_distortion_amplitude": \
         spiral_distortion_amplitude,
         "parabolic_distortion_vector": \
         parabolic_distortion_vector, 
         "skip_validation_and_conversion": \
         False}

    return standard_coord_transform_params_ctor_params



def generate_center(instance_idx):
    if instance_idx <= 3:
        func_name = ("generate_standard_coord_transform_params_{}_"
                     "ctor_params".format(instance_idx+1))
        func_alias = globals()[func_name]
        ctor_params = func_alias()
        
        center = ctor_params["center"]
    else:
        center = (0.46, 0.53)

    return center



def generate_radial_cosine_coefficient_matrix(instance_idx):
    if instance_idx <= 3:
        func_name = ("generate_standard_coord_transform_params_{}_"
                     "ctor_params".format(instance_idx+1))
        func_alias = globals()[func_name]
        ctor_params = func_alias()
        
        A_r_0_2 = ctor_params["quadratic_radial_distortion_amplitude"]
        A_r_1_1 = ctor_params["parabolic_distortion_vector"][0]
        A_r_2_0 = ctor_params["elliptical_distortion_vector"][0]

        radial_cosine_coefficient_matrix = ((0.00000, 0.00000, A_r_0_2),
                                            (0.00000, A_r_1_1, 0.00000),
                                            (A_r_2_0, 0.00000, 0.00000))
    else:
        radial_cosine_coefficient_matrix = ((0.00, 0.00, 0.10, 0.01), 
                                            (0.00, 0.00, 0.00, 0.00))

    return radial_cosine_coefficient_matrix



def generate_radial_sine_coefficient_matrix(instance_idx):
    if instance_idx <= 3:
        func_name = ("generate_standard_coord_transform_params_{}_"
                     "ctor_params".format(instance_idx+1))
        func_alias = globals()[func_name]
        ctor_params = func_alias()
        
        B_r_0_1 = ctor_params["parabolic_distortion_vector"][1]
        B_r_1_0 = ctor_params["elliptical_distortion_vector"][1]

        radial_sine_coefficient_matrix = ((0.00000, B_r_0_1),
                                          (B_r_1_0, 0.00000))
    else:
        radial_sine_coefficient_matrix = ((0, 0),
                                          (0, 0))

    return radial_sine_coefficient_matrix



def generate_tangential_cosine_coefficient_matrix(instance_idx):
    if instance_idx <= 3:
        func_name = ("generate_standard_coord_transform_params_{}_"
                     "ctor_params".format(instance_idx+1))
        func_alias = globals()[func_name]
        ctor_params = func_alias()
        
        B_r_1_0 = ctor_params["elliptical_distortion_vector"][1]
        B_r_0_1 = ctor_params["parabolic_distortion_vector"][1]
    
        A_t_0_2 = ctor_params["spiral_distortion_amplitude"]
        A_t_1_1 = B_r_0_1 / 3
        A_t_2_0 = B_r_1_0

        tangential_cosine_coefficient_matrix = ((0.00000, 0.00000, A_t_0_2),
                                                (0.00000, A_t_1_1, 0.00000),
                                                (A_t_2_0, 0.00000, 0.00000))
    else:
        tangential_cosine_coefficient_matrix = np.array((tuple(),))

    return tangential_cosine_coefficient_matrix



def generate_tangential_sine_coefficient_matrix(instance_idx):
    if instance_idx <= 3:
        func_name = ("generate_standard_coord_transform_params_{}_"
                     "ctor_params".format(instance_idx+1))
        func_alias = globals()[func_name]
        ctor_params = func_alias()
        
        A_r_1_1 = ctor_params["parabolic_distortion_vector"][0]
        A_r_2_0 = ctor_params["elliptical_distortion_vector"][0]
    
        B_t_0_1 = -A_r_1_1 / 3
        B_t_1_0 = -A_r_2_0

        tangential_sine_coefficient_matrix = ((0.00000, B_t_0_1),
                                              (B_t_1_0, 0.00000))
    else:
        tangential_sine_coefficient_matrix = ((0.00, 0.00),)

    return tangential_sine_coefficient_matrix



def generate_coord_transform_params_ctor_params(instance_idx):
    coord_transform_params_ctor_params = \
        {"center": \
         generate_center(instance_idx),
         "radial_cosine_coefficient_matrix": \
         generate_radial_cosine_coefficient_matrix(instance_idx),
         "radial_sine_coefficient_matrix": \
         generate_radial_sine_coefficient_matrix(instance_idx), 
         "tangential_cosine_coefficient_matrix": \
         generate_tangential_cosine_coefficient_matrix(instance_idx),
         "tangential_sine_coefficient_matrix": \
         generate_tangential_sine_coefficient_matrix(instance_idx), 
         "skip_validation_and_conversion": \
         False}

    return coord_transform_params_ctor_params



def test_1_of_CoordTransformParams():
    cls_alias = distoptica.CoordTransformParams

    coord_transform_params = cls_alias()

    coord_transform_params.validation_and_conversion_funcs
    coord_transform_params.pre_serialization_funcs
    coord_transform_params.de_pre_serialization_funcs

    kwargs = {"serializable_rep": coord_transform_params.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    instance_idx = 0

    kwargs = \
        generate_coord_transform_params_ctor_params(instance_idx)
    coord_transform_params = \
        cls_alias(**kwargs)

    kwargs_keys = tuple(kwargs.keys())

    key_1 = kwargs_keys[-1]
    for key_2 in kwargs_keys:
        if key_1 != "skip_validation_and_conversion":
            func_name = "generate_"+key_1
            func_alias = globals()[func_name]
            kwargs[key_1] = func_alias(instance_idx)
        if key_2 != "skip_validation_and_conversion":
            kwargs[key_2] = slice(None)
            with pytest.raises(TypeError) as err_info:
                coord_transform_params = cls_alias(**kwargs)
        key_1 = key_2

    return None



def test_2_of_CoordTransformParams():
    cls_alias = distoptica.CoordTransformParams

    coord_transform_params = cls_alias()

    attr_name_set = ("is_corresponding_model_azimuthally_symmetric", 
                     "is_corresponding_model_trivial", 
                     "is_corresponding_model_standard")

    for attr_name in attr_name_set:
        attr_val = getattr(coord_transform_params, attr_name)
        assert (attr_val == True)
    
    expected_attr_val_superset = ((False, False, True), 
                                  (True, False, True), 
                                  (True, False, True),
                                  (False, False, True), 
                                  (True, False, False))

    num_instance_indices = len(expected_attr_val_superset)

    for instance_idx in range(num_instance_indices):
        new_core_attr_subset_candidate = \
            generate_coord_transform_params_ctor_params(instance_idx)

        coord_transform_params.update(new_core_attr_subset_candidate)

        expected_attr_val_set = expected_attr_val_superset[instance_idx]
        zip_obj = zip(attr_name_set, expected_attr_val_set)

        for attr_name, expected_attr_val in zip_obj:
            attr_val = getattr(coord_transform_params, attr_name)
            assert (attr_val == expected_attr_val)

    return None



def generate_least_squares_alg_params_ctor_params():
    least_squares_alg_params_ctor_params = \
        {"max_num_iterations": 20,
         "initial_damping": 1e-3,
         "factor_for_decreasing_damping": 9,
         "factor_for_increasing_damping": 11,
         "improvement_tol": 0.1, 
         "rel_err_tol": 1e-2, 
         "plateau_tol": 1e-3, 
         "plateau_patience": 2, 
         "skip_validation_and_conversion": False}

    return least_squares_alg_params_ctor_params



def test_1_of_LeastSquaresAlgParams():
    cls_alias = distoptica.LeastSquaresAlgParams

    least_squares_alg_params = cls_alias()

    least_squares_alg_params.validation_and_conversion_funcs
    least_squares_alg_params.pre_serialization_funcs
    least_squares_alg_params.de_pre_serialization_funcs

    kwargs = {"serializable_rep": least_squares_alg_params.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    least_squares_alg_params_ctor_params = \
        generate_least_squares_alg_params_ctor_params()

    kwargs = least_squares_alg_params_ctor_params.copy()
    least_squares_alg_params = cls_alias(**kwargs)

    kwargs_keys = tuple(kwargs.keys())

    key_1 = kwargs_keys[-1]
    for key_2 in kwargs_keys:
        if key_1 != "skip_validation_and_conversion":
            kwargs[key_1] = least_squares_alg_params_ctor_params[key_1]
        if key_2 != "skip_validation_and_conversion":
            kwargs[key_2] = -1
            with pytest.raises(ValueError) as err_info:
                coord_transform_params = cls_alias(**kwargs)
        key_1 = key_2

    new_core_attr_subset_candidate = {"max_num_iterations": 20}
    least_squares_alg_params.update(new_core_attr_subset_candidate)

    return None



def test_1_of_StandardCoordTransformParams():
    cls_alias = distoptica.StandardCoordTransformParams

    standard_coord_transform_params = cls_alias()

    standard_coord_transform_params.validation_and_conversion_funcs
    standard_coord_transform_params.pre_serialization_funcs
    standard_coord_transform_params.de_pre_serialization_funcs

    kwargs = {"serializable_rep": \
              standard_coord_transform_params.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    standard_coord_transform_params_ctor_params = \
        generate_standard_coord_transform_params_1_ctor_params()

    kwargs = standard_coord_transform_params_ctor_params.copy()
    standard_coord_transform_params = cls_alias(**kwargs)

    kwargs_keys = tuple(kwargs.keys())

    key_1 = kwargs_keys[-1]
    for key_2 in kwargs_keys:
        if key_1 != "skip_validation_and_conversion":
            kwargs[key_1] = standard_coord_transform_params_ctor_params[key_1]
        if key_2 != "skip_validation_and_conversion":
            kwargs[key_2] = slice(None)
            with pytest.raises(TypeError) as err_info:
                standard_coord_transform_params = cls_alias(**kwargs)
        key_1 = key_2

    return None


def test_2_of_StandardCoordTransformParams():
    cls_alias = distoptica.StandardCoordTransformParams

    standard_coord_transform_params = cls_alias()

    attr_name_set = ("is_corresponding_model_azimuthally_symmetric", 
                     "is_corresponding_model_trivial")

    expected_attr_val_superset = ((True, True),
                                  (False, False),
                                  (True, False),
                                  (True, False),
                                  (False, False))

    num_instance_indices = len(expected_attr_val_superset)

    for instance_idx in range(num_instance_indices):
        expected_attr_val_set = expected_attr_val_superset[instance_idx]
        zip_obj = zip(attr_name_set, expected_attr_val_set)

        for attr_name, expected_attr_val in zip_obj:
            attr_val = getattr(standard_coord_transform_params, attr_name)
            assert (attr_val == expected_attr_val)

        if instance_idx < num_instance_indices-1:
            func_name = ("generate_standard_coord_transform_params_"
                         "{}_ctor_params").format(instance_idx+1)
            func_alias = globals()[func_name]
            kwargs = {"new_core_attr_subset_candidate": func_alias()}
            standard_coord_transform_params.update(**kwargs)

    return None



def generate_coord_transform_params(instance_idx):
    kwargs = generate_coord_transform_params_ctor_params(instance_idx)
    coord_transform_params = distoptica.CoordTransformParams(**kwargs)

    return coord_transform_params



def generate_sampling_grid_dims_in_pixels():
    sampling_grid_dims_in_pixels = (200, 190)

    return sampling_grid_dims_in_pixels



def generate_device_name(instance_idx):
    cuda_is_available = torch.cuda.is_available()
    default_device = "cuda"*cuda_is_available + "cpu"*(1-cuda_is_available)
    device_name = default_device if (instance_idx <= 3) else "cpu"

    return device_name



def generate_least_squares_alg_params():
    kwargs = generate_least_squares_alg_params_ctor_params()
    least_squares_alg_params = distoptica.LeastSquaresAlgParams(**kwargs)

    return least_squares_alg_params



def generate_distortion_model_ctor_params(instance_idx):
    kwargs = \
        {"instance_idx": instance_idx}
    coord_transform_params = \
        generate_coord_transform_params(**kwargs)

    kwargs = \
        {"instance_idx": instance_idx}
    sampling_grid_dims_in_pixels = \
        generate_sampling_grid_dims_in_pixels()
    device_name = \
        generate_device_name(instance_idx)
    least_squares_alg_params = \
        generate_least_squares_alg_params()
    
    distortion_model_ctor_params = {"coord_transform_params": \
                                    coord_transform_params,
                                    "sampling_grid_dims_in_pixels": \
                                    sampling_grid_dims_in_pixels,
                                    "device_name": \
                                    device_name,
                                    "least_squares_alg_params": \
                                    least_squares_alg_params, 
                                    "skip_validation_and_conversion": \
                                    False}

    return distortion_model_ctor_params



def test_1_of_DistortionModel():
    cls_alias = distoptica.DistortionModel

    distortion_model = cls_alias()

    assert distortion_model.is_azimuthally_symmetric
    assert distortion_model.is_trivial

    distortion_model.validation_and_conversion_funcs
    distortion_model.pre_serialization_funcs
    distortion_model.de_pre_serialization_funcs

    kwargs = {"serializable_rep": distortion_model.pre_serialize()}
    cls_alias.de_pre_serialize(**kwargs)

    distortion_model_ctor_params = \
        generate_distortion_model_ctor_params(instance_idx=0)

    kwargs = distortion_model_ctor_params.copy()
    distortion_model = cls_alias(**kwargs)

    kwargs_keys = tuple(kwargs.keys())

    key_1 = kwargs_keys[-1]
    for key_2 in kwargs_keys:
        if key_1 != "skip_validation_and_conversion":
            kwargs[key_1] = distortion_model_ctor_params[key_1]
        if key_2 != "skip_validation_and_conversion":
            kwargs[key_2] = slice(None)
            with pytest.raises(TypeError) as err_info:
                distortion_model = cls_alias(**kwargs)
        key_1 = key_2

    new_core_attr_subset_candidate = {"device_name": None}
    distortion_model.update(new_core_attr_subset_candidate)

    return None



def test_2_of_DistortionModel():
    kwargs = generate_distortion_model_ctor_params(instance_idx=3)
    distortion_model = distoptica.DistortionModel(**kwargs)

    multi_dim_slice_set_1 = generate_multi_dim_slice_set_1()
    multi_dim_slice_set_2 = generate_multi_dim_slice_set_2()
    zip_obj = zip(multi_dim_slice_set_1, multi_dim_slice_set_2)

    for multi_dim_slice_1, multi_dim_slice_2 in zip_obj:
        undistorted_images = generate_undistorted_images()[multi_dim_slice_1]

        method_alias = distortion_model.distort_then_resample_images
        kwargs = {"undistorted_images": undistorted_images}
        distorted_then_resampled_images = method_alias(**kwargs)
        
        distorted_then_resampled_images = \
            distorted_then_resampled_images.cpu().detach().numpy()
        distorted_then_resampled_image_supports = \
            (distorted_then_resampled_images > 1e-6)

        kwargs = \
            {"axis": (-2, -1)}
        integrated_undistorted_images = \
            undistorted_images.sum(**kwargs)
        integrated_distorted_then_resampled_images = \
            distorted_then_resampled_images[multi_dim_slice_2].sum(**kwargs)

        kwargs = \
            {"h_dim": distorted_then_resampled_image_supports.shape[-1],
             "v_dim": distorted_then_resampled_image_supports.shape[-2]}
        expected_distorted_then_resampled_image_supports = \
            generate_distorted_image_supports(**kwargs)[multi_dim_slice_1]

        kwargs = {"array_1": distorted_then_resampled_image_supports, 
                  "array_2": expected_distorted_then_resampled_image_supports}
        dice_scores = calc_dice_scores(**kwargs)
        assert np.all(dice_scores > 0.95)
    
        rel_diff = (np.abs(integrated_distorted_then_resampled_images
                           - integrated_undistorted_images)
                    / np.abs(integrated_undistorted_images))
        assert np.all(rel_diff < 1e-3)

    return None



def generate_multi_dim_slice_set_1():
    multi_dim_slice_set_1 = ((slice(None), slice(None)), 
                             (1, 1), 
                             (2, slice(None)))

    return multi_dim_slice_set_1



def generate_multi_dim_slice_set_2():
    multi_dim_slice_set_1 = generate_multi_dim_slice_set_1()
    
    multi_dim_slice_set_2 = tuple()
    for multi_dim_slice_1 in multi_dim_slice_set_1:
        multi_dim_slice_2 = tuple()
        for single_dim_slice_1 in multi_dim_slice_1:
            multi_dim_slice_2 += ((single_dim_slice_1,)
                                  if isinstance(single_dim_slice_1, slice)
                                  else (0,))
        multi_dim_slice_set_2 += (multi_dim_slice_2,)

    return multi_dim_slice_set_2



def calc_dice_scores(array_1, array_2):
    numerator = (2*array_1*array_2).sum(axis=(-2, -1))
    denominator = ((array_1*array_1).sum(axis=(-2, -1)) 
                   + (array_2*array_2).sum(axis=(-2, -1)))
    dice_scores = numerator/denominator

    return dice_scores



def test_3_of_DistortionModel():
    kwargs = generate_distortion_model_ctor_params(instance_idx=3)
    distortion_model = distoptica.DistortionModel(**kwargs)

    multi_dim_slice_set_1 = generate_multi_dim_slice_set_1()
    multi_dim_slice_set_2 = generate_multi_dim_slice_set_2()
    zip_obj = zip(multi_dim_slice_set_1, multi_dim_slice_set_2)

    for multi_dim_slice_1, multi_dim_slice_2 in zip_obj:
        distorted_images = generate_distorted_images()[multi_dim_slice_1]

        method_alias = distortion_model.undistort_then_resample_images
        kwargs = {"distorted_images": distorted_images}
        undistorted_then_resampled_images = method_alias(**kwargs)
        
        undistorted_then_resampled_images = \
            undistorted_then_resampled_images.cpu().detach().numpy()
        undistorted_then_resampled_image_supports = \
            (undistorted_then_resampled_images > 1e-6)

        kwargs = \
            {"axis": (-2, -1)}
        integrated_distorted_images = \
            distorted_images.sum(**kwargs)
        integrated_undistorted_then_resampled_images = \
            undistorted_then_resampled_images[multi_dim_slice_2].sum(**kwargs)

        kwargs = \
            {"h_dim": undistorted_then_resampled_image_supports.shape[-1],
             "v_dim": undistorted_then_resampled_image_supports.shape[-2]}
        expected_undistorted_then_resampled_image_supports = \
            generate_undistorted_image_supports(**kwargs)[multi_dim_slice_1]

        kwargs = {"array_1": undistorted_then_resampled_image_supports, 
                  "array_2": expected_undistorted_then_resampled_image_supports}
        dice_scores = calc_dice_scores(**kwargs)
        assert np.all(dice_scores > 0.95)
    
        rel_diff = (np.abs(integrated_undistorted_then_resampled_images
                           -integrated_distorted_images)
                    / np.abs(integrated_distorted_images))
        assert np.all(rel_diff < 1e-3)

    return None



def test_4_of_DistortionModel():
    kwargs = generate_distortion_model_ctor_params(instance_idx=1)
    distortion_model = distoptica.DistortionModel(**kwargs)

    undistorted_images = generate_undistorted_images()

    convergence_map = \
        distortion_model.convergence_map_of_distorted_then_resampled_images
    convergence_map = \
        convergence_map.cpu().detach().numpy()
    
    L, R, B, T = \
        distortion_model.mask_frame_of_distorted_then_resampled_images

    v_dim, h_dim = convergence_map.shape

    assert np.all(convergence_map[T:v_dim-B, L:h_dim-R])

    kwargs = generate_distortion_model_ctor_params(instance_idx=2)
    distortion_model = distoptica.DistortionModel(**kwargs)
        
    with pytest.raises(RuntimeError) as err_info:
        kwargs = {"undistorted_images": undistorted_images}
        _ = distortion_model.distort_then_resample_images(**kwargs)

    return None



def test_5_of_DistortionModel():
    distortion_model_ctor_params = \
        generate_distortion_model_ctor_params(instance_idx=3)
    
    attr_names = ("sampling_grid", 
                  "convergence_map_of_distorted_then_resampled_images",
                  "mask_frame_of_distorted_then_resampled_images",
                  "flow_field_of_coord_transform", 
                  "flow_field_of_coord_transform_right_inverse",
                  "out_of_bounds_map_of_distorted_then_resampled_images",
                  "out_of_bounds_map_of_undistorted_then_resampled_images")
    for attr_name in attr_names:
        kwargs = distortion_model_ctor_params
        distortion_model = distoptica.DistortionModel(**kwargs)
        for iteration_idx in range(2):
            attr = getattr(distortion_model, attr_name)
            assert (attr is not None)

    kwargs = distortion_model_ctor_params
    distortion_model = distoptica.DistortionModel(**kwargs)

    method_alias = distortion_model.undistort_then_resample_images
    kwargs = {"distorted_images": generate_distorted_images()}
    output_array_1 = method_alias(**kwargs).cpu().detach().numpy()
    output_array_2 = method_alias(**kwargs).cpu().detach().numpy()
    abs_diff = np.abs(output_array_1-output_array_2)
    assert np.all(abs_diff < 1e-12)

    kwargs = {"distorted_images": torch.from_numpy(generate_distorted_images())}
    _ = method_alias(**kwargs)

    with pytest.raises(TypeError) as err_info:
        kwargs = {"distorted_images": np.random.rand(2, 2, 2, 2, 2)}
        _ = method_alias(**kwargs)

    kwargs = distortion_model_ctor_params
    distortion_model = distoptica.DistortionModel(**kwargs)

    method_alias = distortion_model.distort_then_resample_images
    kwargs = {"undistorted_images": generate_undistorted_images()}
    output_array_1 = method_alias(**kwargs).cpu().detach().numpy()
    output_array_2 = method_alias(**kwargs).cpu().detach().numpy()
    abs_diff = np.abs(output_array_1-output_array_2)
    assert np.all(abs_diff < 1e-12)

    return None



def test_6_of_DistortionModel():
    cls_alias = distoptica.DistortionModel

    distortion_model = cls_alias()

    attr_name_set = ("is_azimuthally_symmetric", 
                     "is_trivial", 
                     "is_standard")

    for attr_name in attr_name_set:
        attr_val = getattr(distortion_model, attr_name)
        assert (attr_val == True)
    
    expected_attr_val_superset = ((False, False, True), 
                                  (True, False, True), 
                                  (True, False, True),
                                  (False, False, True), 
                                  (True, False, False))

    num_instance_indices = len(expected_attr_val_superset)

    for instance_idx in range(num_instance_indices):
        new_core_attr_subset_candidate = \
            generate_distortion_model_ctor_params(instance_idx)

        distortion_model.update(new_core_attr_subset_candidate)

        expected_attr_val_set = expected_attr_val_superset[instance_idx]
        zip_obj = zip(attr_name_set, expected_attr_val_set)

        for attr_name, expected_attr_val in zip_obj:
            attr_val = getattr(distortion_model, attr_name)
            assert (attr_val == expected_attr_val)

    return None



def test_7_of_DistortionModel():
    kwargs = generate_distortion_model_ctor_params(instance_idx=4)
    distortion_model = distoptica.DistortionModel(**kwargs)

    kwargs = {"undistorted_images": generate_undistorted_images()}
    _ = distortion_model.distort_then_resample_images(**kwargs)

    kwargs = generate_distortion_model_ctor_params(instance_idx=4)
    distortion_model = distoptica.DistortionModel(**kwargs)

    kwargs = {"distorted_images": generate_distorted_images()}
    _ = distortion_model.undistort_then_resample_images(**kwargs)

    distortion_model_ctor_params = \
        generate_distortion_model_ctor_params(instance_idx=3)
    
    attr_names = ("sampling_grid", 
                  "convergence_map_of_distorted_then_resampled_images",
                  "flow_field_of_coord_transform", 
                  "flow_field_of_coord_transform_right_inverse",
                  "out_of_bounds_map_of_distorted_then_resampled_images",
                  "out_of_bounds_map_of_undistorted_then_resampled_images")
    for attr_name in attr_names:
        kwargs = distortion_model_ctor_params
        distortion_model = distoptica.DistortionModel(**kwargs)
        for iteration_idx in range(2):
            attr_1 = getattr(distortion_model, attr_name)
            
            method_alias = getattr(distortion_model, "get_"+attr_name)
            attr_2 = method_alias(deep_copy=False)

            attr_3 = getattr(distortion_model, "_"+attr_name)
            
            assert (attr_2 is not attr_1)
            assert (attr_2 is attr_3)

            if isinstance(attr_1, tuple):
                assert torch.all(attr_1[0] == attr_2[0])
                assert torch.all(attr_1[1] == attr_2[1])
            else:
                assert torch.all(attr_1 == attr_2)

    distortion_model.device

    return None



def test_1_of_generate_standard_distortion_model():
    distortion_model_A = \
        distoptica.DistortionModel()
    
    serializable_rep_of_distortion_model_A = \
        distortion_model_A.pre_serialize()
    
    distortion_model_B = \
        distoptica.generate_standard_distortion_model()
    
    serializable_rep_of_distortion_model_B = \
        distortion_model_B.pre_serialize()

    core_attr_names = tuple(distortion_model_A.core_attrs.keys())

    for core_attr_name in core_attr_names:
        if core_attr_name != "coord_transform_params":
            serializable_obj_A = \
                serializable_rep_of_distortion_model_A[core_attr_name]
            serializable_obj_B = \
                serializable_rep_of_distortion_model_B[core_attr_name]

            assert (serializable_obj_A == serializable_obj_B)
    
    assert distortion_model_A.is_trivial
    assert distortion_model_B.is_trivial

    return None



def test_2_of_generate_standard_distortion_model():
    kwargs = \
        generate_standard_coord_transform_params_1_ctor_params()
    standard_coord_transform_params = \
        distoptica.StandardCoordTransformParams(**kwargs)
    
    distortion_model_ctor_params = \
        generate_distortion_model_ctor_params(instance_idx=0)

    kwargs = distortion_model_ctor_params
    distortion_model_A = distoptica.DistortionModel(**kwargs)
    
    serializable_rep_of_distortion_model_A = \
        distortion_model_A.pre_serialize()

    kwargs = distortion_model_ctor_params.copy()
    del kwargs["coord_transform_params"]
    del kwargs["skip_validation_and_conversion"]
    kwargs["standard_coord_transform_params"] = standard_coord_transform_params
    distortion_model_B = distoptica.generate_standard_distortion_model(**kwargs)

    kwargs["skip_validation_and_conversion"] = True
    distortion_model_C = distoptica.generate_standard_distortion_model(**kwargs)
    
    serializable_rep_of_distortion_model_B = \
        distortion_model_B.pre_serialize()
    serializable_rep_of_distortion_model_C = \
        distortion_model_C.pre_serialize()

    core_attr_names = tuple(distortion_model_A.core_attrs.keys())

    for core_attr_name in core_attr_names:
        serializable_obj_A = \
            serializable_rep_of_distortion_model_A[core_attr_name]
        serializable_obj_B = \
            serializable_rep_of_distortion_model_B[core_attr_name]
        serializable_obj_C = \
            serializable_rep_of_distortion_model_C[core_attr_name]

        assert (serializable_obj_A == serializable_obj_B)
        assert (serializable_obj_B == serializable_obj_C)
    
    assert distortion_model_A.is_standard
    assert distortion_model_B.is_standard

    return None



def test_3_of_generate_standard_distortion_model():
    kwargs = \
        generate_standard_coord_transform_params_1_ctor_params()
    standard_coord_transform_params = \
        distoptica.StandardCoordTransformParams(**kwargs)
    
    distortion_model_ctor_params = \
        generate_distortion_model_ctor_params(instance_idx=4)

    kwargs = distortion_model_ctor_params.copy()
    del kwargs["coord_transform_params"]
    del kwargs["skip_validation_and_conversion"]
    kwargs["standard_coord_transform_params"] = standard_coord_transform_params
    distortion_model = distoptica.generate_standard_distortion_model(**kwargs)

    return None



###########################
## Define error messages ##
###########################
