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
"""``distoptica`` is a Python library for modelling optical distortions.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For accessing attributes of functions.
import inspect

# For randomly selecting items in dictionaries.
import random

# For performing deep copies.
import copy



# For general array handling.
import numpy as np
import torch

# For validating and converting objects.
import czekitout.check
import czekitout.convert

# For defining classes that support enforced validation, updatability,
# pre-serialization, and de-serialization.
import fancytypes



# Get version of current package.
from distoptica.version import __version__



##################################
## Define classes and functions ##
##################################

# List of public objects in package.
__all__ = ["CoordTransformParams",
           "LeastSquaresAlgParams",
           "DistortionModel",
           "StandardCoordTransformParams",
           "generate_standard_distortion_model"]



class _Polynomials(torch.nn.Module):
    def __init__(self, coefficient_matrix):
        super().__init__()

        coefficient_matrix = torch.tensor(coefficient_matrix,
                                          dtype=torch.float32)
        self.coefficient_matrix = torch.nn.Parameter(coefficient_matrix)

        self.M = self.coefficient_matrix.shape[1]

        self.forward_output = None
        self.derivative_wrt_u_r = None

        return None


    
    def eval_forward_output(self, inputs):
        powers_of_u_r = inputs["powers_of_u_r"]

        M = self.M

        output_tensor = torch.einsum("nm, mij -> nij",
                                     self.coefficient_matrix,
                                     powers_of_u_r[1:M+1])

        return output_tensor



    def eval_derivative_wrt_u_r(self, inputs):
        derivative_of_powers_of_u_r_wrt_u_r = \
            inputs["derivative_of_powers_of_u_r_wrt_u_r"]

        M = self.M

        output_tensor = torch.einsum("nm, mij -> nij",
                                     self.coefficient_matrix,
                                     derivative_of_powers_of_u_r_wrt_u_r[0:M])

        return output_tensor



    def forward(self, inputs):
        output_tensor = self.eval_forward_output(inputs)

        return output_tensor



class _FourierSeries(torch.nn.Module):
    def __init__(self, cosine_amplitudes, sine_amplitudes):
        super().__init__()

        self.cosine_amplitudes = cosine_amplitudes
        self.sine_amplitudes = sine_amplitudes

        self.N_cos = cosine_amplitudes.coefficient_matrix.shape[0]-1
        self.N_sin = sine_amplitudes.coefficient_matrix.shape[0]
        self.num_azimuthal_orders = max(self.N_cos+1, self.N_sin+1)

        self.M = max(cosine_amplitudes.M, sine_amplitudes.M)

        self.forward_output = None
        self.derivative_wrt_u_r = None
        self.derivative_wrt_u_theta = None

        return None



    def eval_forward_output(self, inputs):
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]

        N_cos = self.N_cos
        N_sin = self.N_sin

        intermediate_tensor_1 = (self.cosine_amplitudes.forward_output
                                 * cosines_of_scaled_u_thetas[0:N_cos+1])
        intermediate_tensor_1 = intermediate_tensor_1.sum(dim=0)

        intermediate_tensor_2 = (self.sine_amplitudes.forward_output
                                 * sines_of_scaled_u_thetas[0:N_sin])
        intermediate_tensor_2 = intermediate_tensor_2.sum(dim=0)

        output_tensor = intermediate_tensor_1+intermediate_tensor_2

        return output_tensor



    def eval_derivative_wrt_u_r(self, inputs):
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]

        N_cos = self.N_cos
        N_sin = self.N_sin

        intermediate_tensor_1 = (self.cosine_amplitudes.derivative_wrt_u_r
                                 * cosines_of_scaled_u_thetas[0:N_cos+1])
        intermediate_tensor_1 = intermediate_tensor_1.sum(dim=0)

        intermediate_tensor_2 = (self.sine_amplitudes.derivative_wrt_u_r
                                 * sines_of_scaled_u_thetas[0:N_sin])
        intermediate_tensor_2 = intermediate_tensor_2.sum(dim=0)

        output_tensor = intermediate_tensor_1+intermediate_tensor_2

        return output_tensor



    def eval_derivative_wrt_u_theta(self, inputs):
        derivative_of_cosines_of_scaled_u_thetas_wrt_u_theta = \
            inputs["derivative_of_cosines_of_scaled_u_thetas_wrt_u_theta"]
        derivative_of_sines_of_scaled_u_thetas_wrt_u_theta = \
            inputs["derivative_of_sines_of_scaled_u_thetas_wrt_u_theta"]

        N_cos = self.N_cos
        N_sin = self.N_sin

        intermediate_tensor_1 = \
            (self.cosine_amplitudes.forward_output[1:N_cos+1]
             * derivative_of_cosines_of_scaled_u_thetas_wrt_u_theta[0:N_cos])
        intermediate_tensor_1 = \
            intermediate_tensor_1.sum(dim=0)

        intermediate_tensor_2 = \
            (self.sine_amplitudes.forward_output
             * derivative_of_sines_of_scaled_u_thetas_wrt_u_theta[0:N_sin])
        intermediate_tensor_2 = \
            intermediate_tensor_2.sum(dim=0)

        output_tensor = intermediate_tensor_1+intermediate_tensor_2

        return output_tensor



    def forward(self, inputs):
        output_tensor = self.eval_forward_output(inputs)

        return output_tensor



class _CoordTransform(torch.nn.Module):
    def __init__(self,
                 center,
                 radial_fourier_series,
                 tangential_fourier_series):
        super().__init__()

        device = radial_fourier_series.sine_amplitudes.coefficient_matrix.device

        center = torch.tensor(center, dtype=torch.float32, device=device)

        self.center = torch.nn.Parameter(center)
        self.radial_fourier_series = radial_fourier_series
        self.tangential_fourier_series = tangential_fourier_series

        args = (radial_fourier_series.num_azimuthal_orders,
                tangential_fourier_series.num_azimuthal_orders)
        num_azimuthal_orders = max(*args)
        azimuthal_orders = torch.arange(0, num_azimuthal_orders, device=device)
        self.register_buffer("azimuthal_orders", azimuthal_orders)

        self.M = max(radial_fourier_series.M, tangential_fourier_series.M)
        exponents = torch.arange(0, self.M+1, device=device)
        self.register_buffer("exponents", exponents)

        self.forward_output = None
        self.jacobian = None

        return None



    def eval_forward_output(self, inputs):
        u_x = inputs["u_x"]
        u_y = inputs["u_y"]
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]

        cos_u_theta = cosines_of_scaled_u_thetas[1]
        sin_u_theta = sines_of_scaled_u_thetas[0]

        output_tensor_shape = (2,) + cos_u_theta.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=cos_u_theta.dtype,
                                    device=cos_u_theta.device)

        output_tensor[0] = (u_x
                            + (self.radial_fourier_series.forward_output
                               * cos_u_theta)
                            - (self.tangential_fourier_series.forward_output
                               * sin_u_theta))
        output_tensor[1] = (u_y
                            + (self.radial_fourier_series.forward_output
                               * sin_u_theta)
                            + (self.tangential_fourier_series.forward_output
                               * cos_u_theta))

        return output_tensor



    def eval_jacobian(self, inputs):
        inputs["derivative_wrt_u_r"] = \
            self.eval_derivative_wrt_u_r(inputs)
        inputs["derivative_wrt_u_theta"] = \
            self.eval_derivative_wrt_u_theta(inputs)

        derivative_wrt_u_x = self.eval_derivative_wrt_u_x(inputs)
        derivative_wrt_u_y = self.eval_derivative_wrt_u_y(inputs)

        del inputs["derivative_wrt_u_r"]
        del inputs["derivative_wrt_u_theta"]
        
        output_tensor_shape = (2,) + derivative_wrt_u_x.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=derivative_wrt_u_x.dtype,
                                    device=derivative_wrt_u_x.device)
        
        output_tensor[0, 0] = derivative_wrt_u_x[0]
        output_tensor[1, 0] = derivative_wrt_u_x[1]
        output_tensor[0, 1] = derivative_wrt_u_y[0]
        output_tensor[1, 1] = derivative_wrt_u_y[1]

        return output_tensor



    def eval_derivative_wrt_u_r(self, inputs):
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]

        cos_u_theta = cosines_of_scaled_u_thetas[1]
        sin_u_theta = sines_of_scaled_u_thetas[0]

        output_tensor_shape = (2,) + cos_u_theta.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=cos_u_theta.dtype,
                                    device=cos_u_theta.device)

        intermediate_tensor = 1+self.radial_fourier_series.derivative_wrt_u_r

        output_tensor[0] = \
            ((intermediate_tensor
              * cos_u_theta)
             - (self.tangential_fourier_series.derivative_wrt_u_r
                * sin_u_theta))
        output_tensor[1] = \
            ((intermediate_tensor
              * sin_u_theta)
             + (self.tangential_fourier_series.derivative_wrt_u_r
                * cos_u_theta))

        return output_tensor



    def eval_derivative_wrt_u_theta(self, inputs):
        powers_of_u_r = inputs["powers_of_u_r"]
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]

        u_r = powers_of_u_r[1]
        cos_u_theta = cosines_of_scaled_u_thetas[1]
        sin_u_theta = sines_of_scaled_u_thetas[0]

        output_tensor_shape = (2,) + cos_u_theta.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=cos_u_theta.dtype,
                                    device=cos_u_theta.device)

        output_tensor[0] = \
            (-u_r*sin_u_theta
             + ((self.radial_fourier_series.derivative_wrt_u_theta
                 * cos_u_theta)
                - (self.radial_fourier_series.forward_output
                   * sin_u_theta))
             - ((self.tangential_fourier_series.derivative_wrt_u_theta
                 * sin_u_theta)
                + (self.tangential_fourier_series.forward_output
                   * cos_u_theta)))
        output_tensor[1] = \
            (u_r*cos_u_theta
             + ((self.radial_fourier_series.derivative_wrt_u_theta
                 * sin_u_theta)
                + (self.radial_fourier_series.forward_output
                   * cos_u_theta))
             + ((self.tangential_fourier_series.derivative_wrt_u_theta
                 * cos_u_theta)
                - (self.tangential_fourier_series.forward_output
                   * sin_u_theta)))

        return output_tensor



    def eval_derivative_wrt_u_x(self, inputs):
        derivative_of_u_theta_wrt_u_x = inputs["derivative_of_u_theta_wrt_u_x"]
        cosines_of_scaled_u_thetas = inputs["cosines_of_scaled_u_thetas"]
        derivative_wrt_u_r = inputs["derivative_wrt_u_r"]
        derivative_wrt_u_theta = inputs["derivative_wrt_u_theta"]

        cos_u_theta = cosines_of_scaled_u_thetas[1]

        output_tensor_shape = (2,) + cos_u_theta.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=cos_u_theta.dtype,
                                    device=cos_u_theta.device)

        output_tensor[0] = \
            (cos_u_theta*derivative_wrt_u_r[0]
             + derivative_of_u_theta_wrt_u_x*derivative_wrt_u_theta[0])
        output_tensor[1] = \
            (cos_u_theta*derivative_wrt_u_r[1]
             + derivative_of_u_theta_wrt_u_x*derivative_wrt_u_theta[1])

        return output_tensor



    def eval_derivative_wrt_u_y(self, inputs):
        derivative_of_u_theta_wrt_u_y = inputs["derivative_of_u_theta_wrt_u_y"]
        sines_of_scaled_u_thetas = inputs["sines_of_scaled_u_thetas"]
        derivative_wrt_u_r = inputs["derivative_wrt_u_r"]
        derivative_wrt_u_theta = inputs["derivative_wrt_u_theta"]

        sin_u_theta = sines_of_scaled_u_thetas[0]

        output_tensor_shape = (2,) + sin_u_theta.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=sin_u_theta.dtype,
                                    device=sin_u_theta.device)

        output_tensor[0] = \
            (sin_u_theta*derivative_wrt_u_r[0]
             + derivative_of_u_theta_wrt_u_y*derivative_wrt_u_theta[0])
        output_tensor[1] = \
            (sin_u_theta*derivative_wrt_u_r[1]
             + derivative_of_u_theta_wrt_u_y*derivative_wrt_u_theta[1])

        return output_tensor



    def forward(self, inputs):
        output_tensor = self.eval_forward_output(inputs)

        return output_tensor



def _update_coord_transform_input_subset_1(coord_transform_inputs,
                                           coord_transform,
                                           u_x,
                                           u_y):
    x_c_D, y_c_D = coord_transform.center
    delta_u_x = u_x - x_c_D
    delta_u_y = u_y - y_c_D
    u_r = torch.sqrt(delta_u_x*delta_u_x + delta_u_y*delta_u_y)
    exponents = coord_transform.exponents
    powers_of_u_r = torch.pow(u_r[None, :, :], exponents[:, None, None])
    
    u_theta = torch.atan2(delta_u_y, delta_u_x)
    azimuthal_orders = coord_transform.azimuthal_orders
    scaled_u_thetas = torch.einsum("i, jk -> ijk", azimuthal_orders, u_theta)
    cosines_of_scaled_u_thetas = torch.cos(scaled_u_thetas)
    sines_of_scaled_u_thetas = torch.sin(scaled_u_thetas[1:])

    local_obj_subset = locals()
    
    coord_transform_input_key_subset_1 = \
        _generate_coord_transform_input_key_subset_1()

    for key in coord_transform_input_key_subset_1:
        elem = local_obj_subset[key]
        _set_coord_transform_inputs_elem(coord_transform_inputs, key, elem)

    return None



def _generate_coord_transform_input_key_subset_1():
    coord_transform_input_key_subset_1 = \
        ("u_x",
         "u_y",
         "delta_u_x",
         "delta_u_y",
         "powers_of_u_r",
         "cosines_of_scaled_u_thetas",
         "sines_of_scaled_u_thetas")

    return coord_transform_input_key_subset_1



def _update_coord_transform_input_subset_2(coord_transform_inputs,
                                           coord_transform):
    exponents = coord_transform.exponents
    M = coord_transform.M
    powers_of_u_r = coord_transform_inputs["powers_of_u_r"]
    derivative_of_powers_of_u_r_wrt_u_r = torch.einsum("i, ijk -> ijk",
                                                       exponents[1:M+1],
                                                       powers_of_u_r[0:M])

    azimuthal_orders = \
        coord_transform.azimuthal_orders
    sines_of_scaled_u_thetas = \
        coord_transform_inputs["sines_of_scaled_u_thetas"]
    cosines_of_scaled_u_thetas = \
        coord_transform_inputs["cosines_of_scaled_u_thetas"]

    derivative_of_cosines_of_scaled_u_thetas_wrt_u_theta = \
        torch.einsum("i, ijk -> ijk",
                     azimuthal_orders[1:],
                     -sines_of_scaled_u_thetas)
    derivative_of_sines_of_scaled_u_thetas_wrt_u_theta = \
        torch.einsum("i, ijk -> ijk",
                     azimuthal_orders[1:],
                     cosines_of_scaled_u_thetas[1:])

    bool_mat_1 = (powers_of_u_r[1] == 0)
    bool_mat_2 = ~bool_mat_1
    divisor = powers_of_u_r[2] + bool_mat_1
    delta_u_x = coord_transform_inputs["delta_u_x"]
    delta_u_y = coord_transform_inputs["delta_u_y"]
    derivative_of_u_theta_wrt_u_x = (-delta_u_y/divisor) * bool_mat_2
    derivative_of_u_theta_wrt_u_y = (delta_u_x/divisor) * bool_mat_2

    local_obj_subset = locals()

    coord_transform_input_key_subset_2 = \
        _generate_coord_transform_input_key_subset_2()

    for key in coord_transform_input_key_subset_2:
        elem = local_obj_subset[key]
        _set_coord_transform_inputs_elem(coord_transform_inputs, key, elem)

    return None



def _generate_coord_transform_input_key_subset_2():
    coord_transform_input_key_subset_2 = \
        ("derivative_of_powers_of_u_r_wrt_u_r",
         "derivative_of_cosines_of_scaled_u_thetas_wrt_u_theta",
         "derivative_of_sines_of_scaled_u_thetas_wrt_u_theta",
         "derivative_of_u_theta_wrt_u_x",
         "derivative_of_u_theta_wrt_u_y")

    return coord_transform_input_key_subset_2



def _set_coord_transform_inputs_elem(coord_transform_inputs, key, elem):
    if key in coord_transform_inputs:
        coord_transform_inputs[key][:] = elem[:]
    else:
        coord_transform_inputs[key] = elem

    return None



def _check_and_convert_center(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    center = czekitout.convert.to_pair_of_floats(**kwargs)

    return center



def _pre_serialize_center(center):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_center(serializable_rep):
    center = serializable_rep

    return center



def _check_and_convert_radial_cosine_coefficient_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    params["coefficient_matrix"] = obj
    params["name_of_alias_of_coefficient_matrix"] = obj_name

    radial_cosine_coefficient_matrix = \
        _check_and_convert_coefficient_matrix(params)

    del params["coefficient_matrix"]
    del params["name_of_alias_of_coefficient_matrix"]

    return radial_cosine_coefficient_matrix



def _check_and_convert_coefficient_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    if obj is None:
        coefficient_matrix = ((0.,),)
    else:
        try:
            kwargs = \
                {"obj": obj, "obj_name": obj_name}
            coefficient_matrix = \
                czekitout.convert.to_real_numpy_matrix(**kwargs)
        except:
            name_of_alias_of_coefficient_matrix = \
                params["name_of_alias_of_coefficient_matrix"]
            unformatted_err_msg = \
                globals()[current_func_name+"_err_msg_1"]
            err_msg = \
                unformatted_err_msg.format(name_of_alias_of_coefficient_matrix)

            raise TypeError(err_msg)                

        if coefficient_matrix.size == 0:
            coefficient_matrix = ((0.,),)
        else:
            coefficient_matrix = tuple(tuple(row) for row in coefficient_matrix)

    return coefficient_matrix



def _pre_serialize_radial_cosine_coefficient_matrix(
        radial_cosine_coefficient_matrix):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_cosine_coefficient_matrix(serializable_rep):
    radial_cosine_coefficient_matrix = serializable_rep

    return radial_cosine_coefficient_matrix



def _check_and_convert_radial_sine_coefficient_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    params["coefficient_matrix"] = obj
    params["name_of_alias_of_coefficient_matrix"] = obj_name

    radial_sine_coefficient_matrix = \
        _check_and_convert_coefficient_matrix(params)

    del params["coefficient_matrix"]
    del params["name_of_alias_of_coefficient_matrix"]

    return radial_sine_coefficient_matrix



def _pre_serialize_radial_sine_coefficient_matrix(
        radial_sine_coefficient_matrix):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_sine_coefficient_matrix(serializable_rep):
    radial_sine_coefficient_matrix = serializable_rep

    return radial_sine_coefficient_matrix



def _check_and_convert_tangential_cosine_coefficient_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    params["coefficient_matrix"] = obj
    params["name_of_alias_of_coefficient_matrix"] = obj_name

    tangential_cosine_coefficient_matrix = \
        _check_and_convert_coefficient_matrix(params)

    del params["coefficient_matrix"]
    del params["name_of_alias_of_coefficient_matrix"]

    return tangential_cosine_coefficient_matrix



def _pre_serialize_tangential_cosine_coefficient_matrix(
        tangential_cosine_coefficient_matrix):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_tangential_cosine_coefficient_matrix(serializable_rep):
    tangential_cosine_coefficient_matrix = serializable_rep

    return tangential_cosine_coefficient_matrix



def _check_and_convert_tangential_sine_coefficient_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    params["coefficient_matrix"] = obj
    params["name_of_alias_of_coefficient_matrix"] = obj_name

    tangential_sine_coefficient_matrix = \
        _check_and_convert_coefficient_matrix(params)

    del params["coefficient_matrix"]
    del params["name_of_alias_of_coefficient_matrix"]

    return tangential_sine_coefficient_matrix



def _pre_serialize_tangential_sine_coefficient_matrix(
        tangential_sine_coefficient_matrix):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_tangential_sine_coefficient_matrix(serializable_rep):
    tangential_sine_coefficient_matrix = serializable_rep

    return tangential_sine_coefficient_matrix



_default_center = (0.5, 0.5)
_default_radial_cosine_coefficient_matrix = ((0.,),)
_default_radial_sine_coefficient_matrix = ((0.,),)
_default_tangential_cosine_coefficient_matrix = ((0.,),)
_default_tangential_sine_coefficient_matrix = ((0.,),)
_default_skip_validation_and_conversion = False



_cls_alias = fancytypes.PreSerializableAndUpdatable
class CoordTransformParams(_cls_alias):
    r"""The parameters of a generic trigonometric series defining a coordinate 
    transformation.

    Users are encouraged to read the summary documentation of the class
    :class:`distoptica.DistortionModel` before reading the documentation for the
    current class as it provides essential context to what is discussed below.

    As discussed in the summary documentation of the class
    :class:`distoptica.DistortionModel`, optical distortions introduced in an
    imaging experiment can be described by a coordinate transformation,
    comprising of two components:
    :math:`T_{\wasylozenge;x}\left(u_{x},u_{y}\right)` and
    :math:`T_{\wasylozenge;y}\left(u_{x},u_{y}\right)`. In :mod:`distoptica`,
    these two functions are assumed to be of the following respective
    mathematical forms:

    .. math ::
        T_{\wasylozenge;x}\left(u_{x},u_{y}\right) & =
        \left\{ u_{r}\cos\left(u_{\theta}\right)+x_{c;D}\right\} \nonumber \\
        & \quad+\left\{ T_{\wasylozenge;r}\left(u_{r},u_{\theta}\right)
        \cos\left(u_{\theta}\right)
        -T_{\wasylozenge;t}\left(u_{r},u_{\theta}\right)
        \sin\left(u_{\theta}\right)\right\} ,
        :label: T_distsq_x__1

    .. math ::
        T_{\wasylozenge;y}\left(u_{x},u_{y}\right) & =
        \left\{ u_{r}\sin\left(u_{\theta}\right)+y_{c;D}\right\} \nonumber \\
        & \quad+\left\{ T_{\wasylozenge;r}\left(u_{r},u_{\theta}\right)
        \sin\left(u_{\theta}\right)
        +T_{\wasylozenge;t}\left(u_{r},u_{\theta}\right)
        \cos\left(u_{\theta}\right)\right\} ,
        :label: T_distsq_y__1

    where

    .. math ::
        \left(x_{c;D},y_{c;D}\right) & \in\mathbb{R}^{2},
        :label: center_of_distortion__1

    .. math ::
        u_{r} & =\sqrt{\left(u_{x}-x_{c;D}\right)^{2}+
        \left(u_{y}-y_{c;D}\right)^{2}},
        :label: u_r__1

    .. math ::
        u_{\theta} & =\tan^{-1}\left(\frac{y-y_{c;D}}{x-x_{c;D}}\right),
        :label: u_theta__1

    .. math ::
        T_{\wasylozenge;r}\left(u_{r},u_{\theta}\right) & =
        \sum_{v_{1}=0}^{N_{r;\cos}}\rho_{\cos;r;v_{1}}\left(u_{r}\right)
        \cos\left(v_{1}u_{\theta}\right)\nonumber \\
        & \quad+\sum_{v_{1}=0}^{N_{r;\sin}-1}
        \rho_{\sin;r;v_{1}}\left(u_{r}\right)\sin\left(\left\{ v_{1}+1\right\} 
        u_{\theta}\right),
        :label: T_distsq_r__1

    .. math ::
        T_{\wasylozenge;t}\left(u_{r},u_{\theta}\right) & =
        \sum_{v_{1}=0}^{N_{t;\cos}}\rho_{\cos;t;v_{1}}\left(u_{r}\right)
        \cos\left(v_{1}u_{\theta}\right)\nonumber \\
        & \quad+\sum_{v_{1}=0}^{N_{t;\sin}-1}
        \rho_{\sin;t;v_{1}}\left(u_{r}\right)\sin\left(\left\{ v_{1}+1\right\} 
        u_{\theta}\right),
        :label: T_distsq_t__1

    with
    
    .. math ::
        \rho_{r;\cos;v_{1}}\left(u_{r}\right) & =\sum_{v_{2}=0}^{M_{r;\cos}-1}
        A_{r;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_r_cos_v_1__1

    .. math ::
        \rho_{r;\sin;v_{1}}\left(u_{r}\right) & =\sum_{v_{2}=0}^{M_{r;\sin}-1}
        B_{r;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_r_sin_v_1__1

    .. math ::
        \rho_{t;\cos;v_{1}}\left(u_{r}\right) & =\sum_{v_{2}=0}^{M_{r;\cos}-1}
        A_{t;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_t_cos_v_1__1

    .. math ::
        \rho_{t;\sin;v_{1}}\left(u_{r}\right) & =\sum_{v_{2}=0}^{M_{r;\sin}-1}
        B_{t;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_t_sin_v_1__1

    :math:`A_{r;v_{1},v_{2}}` being a real-valued
    :math:`\left(N_{r;\cos}+1\right)\times M_{r;\cos}` matrix,
    :math:`B_{r;v_{1},v_{2}}` being a real-valued :math:`N_{r;\sin}\times
    M_{r;\sin}` matrix, :math:`A_{t;v_{1},v_{2}}` being a real-valued
    :math:`\left(N_{t;\cos}+1\right)\times M_{t;\cos}` matrix, and
    :math:`B_{t;v_{1},v_{2}}` being a real-valued :math:`N_{t;\sin}\times
    M_{t;\sin}` matrix. We refer to :math:`\left(x_{c;D},y_{c;D}\right)`,
    :math:`A_{r;v_{1},v_{2}}`, :math:`B_{r;v_{1},v_{2}}`,
    :math:`A_{t;v_{1},v_{2}}`, and :math:`B_{t;v_{1},v_{2}}` as the distortion
    center, the radial cosine coefficient matrix, the radial sine coefficient
    matrix, the tangential cosine coefficient matrix, and the tangential sine
    coefficient matrix respectively.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)), optional
        The distortion center :math:`\left(x_{c;D},y_{c;D}\right)`, where
        ``center[0]`` and ``center[1]`` are :math:`x_{c;D}` and :math:`y_{c;D}`
        respectively.
    radial_cosine_coefficient_matrix : `array_like` (`float`, ndim=2), optional
        The radial cosine coefficient matrix. For every pair of nonnegative
        integers ``(v_1, v_2)`` that does not raise an ``IndexError`` exception
        upon calling ``radial_cosine_coefficient_matrix[v_1, v_2]``,
        ``radial_cosine_coefficient_matrix[v_1, v_2]`` is equal to
        :math:`A_{r;v_{1},v_{2}}`, with the integers :math:`v_{1}`, and
        :math:`v_{2}` being equal to the values of ``v_1``, and ``v_2``
        respectively.
    radial_sine_coefficient_matrix : `array_like` (`float`, ndim=2), optional
        The radial sine coefficient matrix. For every pair of nonnegative
        integers ``(v_1, v_2)`` that does not raise an ``IndexError`` exception
        upon calling ``radial_sine_coefficient_matrix[v_1, v_2]``,
        ``radial_sine_coefficient_matrix[v_1, v_2]`` is equal to
        :math:`B_{r;v_{1},v_{2}}`, with the integers :math:`v_{1}`, and
        :math:`v_{2}` being equal to the values of ``v_1``, and ``v_2``
        respectively.
    tangential_cosine_coefficient_matrix : `array_like` (`float`, ndim=2), optional
        The tangential cosine coefficient matrix. For every pair of nonnegative
        integers ``(v_1, v_2)`` that does not raise an ``IndexError`` exception
        upon calling ``tangential_cosine_coefficient_matrix[v_1, v_2]``,
        ``tangential_cosine_coefficient_matrix[v_1, v_2]`` is equal to
        :math:`A_{t;v_{1},v_{2}}`, with the integers :math:`v_{1}`, and
        :math:`v_{2}` being equal to the values of ``v_1``, and ``v_2``
        respectively.
    tangential_sine_coefficient_matrix : `array_like` (`float`, ndim=2), optional
        The tangential sine coefficient matrix. For every pair of nonnegative
        integers ``(v_1, v_2)`` that does not raise an ``IndexError`` exception
        upon calling ``tangential_sine_coefficient_matrix[v_1, v_2]``,
        ``tangential_sine_coefficient_matrix[v_1, v_2]`` is equal to
        :math:`B_{t;v_{1},v_{2}}`, with the integers :math:`v_{1}`, and
        :math:`v_{2}` being equal to the values of ``v_1``, and ``v_2``
        respectively.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "radial_cosine_coefficient_matrix",
                        "radial_sine_coefficient_matrix",
                        "tangential_cosine_coefficient_matrix",
                        "tangential_sine_coefficient_matrix")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}
    

    
    def __init__(self,
                 center=\
                 _default_center,
                 radial_cosine_coefficient_matrix=\
                 _default_radial_cosine_coefficient_matrix,
                 radial_sine_coefficient_matrix=\
                 _default_radial_sine_coefficient_matrix,
                 tangential_cosine_coefficient_matrix=\
                 _default_tangential_cosine_coefficient_matrix,
                 tangential_sine_coefficient_matrix=\
                 _default_tangential_sine_coefficient_matrix,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        self.execute_post_core_attrs_update_actions()

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def execute_post_core_attrs_update_actions(self):
        r"""Execute the sequence of actions that follows immediately after 
        updating the core attributes.

        """
        partial_sum_of_abs_vals_of_coefficients = 0.0
        total_sum_of_abs_vals_of_coefficients = 0.0

        self_core_attrs = self.get_core_attrs(deep_copy=False)

        for attr_name in self_core_attrs:
            if "coefficient_matrix" in attr_name:
                coefficient_matrix = \
                    np.array(self_core_attrs[attr_name])

                starting_row = \
                    1 if "radial_cosine" in attr_name else 0
                partial_sum_of_abs_vals_of_coefficients += \
                    np.sum(np.abs(coefficient_matrix[starting_row:]))
                
                starting_row = \
                    0
                total_sum_of_abs_vals_of_coefficients += \
                    np.sum(np.abs(coefficient_matrix[starting_row:]))

        if partial_sum_of_abs_vals_of_coefficients == 0.0:
            self._is_corresponding_model_azimuthally_symmetric = True
        else:
            self._is_corresponding_model_azimuthally_symmetric = False

        if total_sum_of_abs_vals_of_coefficients == 0.0:
            self._is_corresponding_model_trivial = True
        else:
            self._is_corresponding_model_trivial = False

        total_sum_of_abs_vals_of_modified_coefficients = \
            self._calc_total_sum_of_abs_vals_of_modified_coefficients()

        if total_sum_of_abs_vals_of_modified_coefficients == 0.0:
            self._is_corresponding_model_standard = True
        else:
            self._is_corresponding_model_standard = False

        return None



    def _calc_total_sum_of_abs_vals_of_modified_coefficients(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        A_r = self_core_attrs["radial_cosine_coefficient_matrix"]
        B_r = self_core_attrs["radial_sine_coefficient_matrix"]
        A_t = self_core_attrs["tangential_cosine_coefficient_matrix"]
        B_t = self_core_attrs["tangential_sine_coefficient_matrix"]

        modified_A_r = np.array(A_r)
        if modified_A_r.shape[1] >= 3:
            modified_A_r[0, 2] = 0
        if (modified_A_r.shape[0] >= 2) and (modified_A_r.shape[1] >= 2):
            modified_A_r[1, 1] = 0
        if modified_A_r.shape[0] >= 3:
            modified_A_r[2, 0] = 0

        modified_B_r = np.array(B_r)
        if modified_B_r.shape[0] >= 2:
            modified_B_r[1, 0] = 0
        if modified_B_r.shape[1] >= 2:
            modified_B_r[0, 1] = 0

        modified_A_t = np.array(A_t)
        if modified_A_t.shape[1] >= 3:
            modified_A_t[0, 2] = 0
        if (modified_A_t.shape[0] >= 2) and (modified_A_t.shape[1] >= 2):
            modified_A_t[1, 1] = 0
        if modified_A_t.shape[0] >= 3:
            modified_A_t[2, 0] = 0

        modified_B_t = np.array(B_t)
        if modified_B_t.shape[0] >= 2:
            modified_B_t[1, 0] = 0
        if modified_B_t.shape[1] >= 2:
            modified_B_t[0, 1] = 0

        total_sum_of_abs_vals_of_modified_coefficients = \
            (np.sum(np.abs(modified_A_r))
             + np.sum(np.abs(modified_B_r))
             + np.sum(np.abs(modified_A_t))
             + np.sum(np.abs(modified_B_t)))

        return total_sum_of_abs_vals_of_modified_coefficients



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def is_corresponding_model_azimuthally_symmetric(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is azimuthally symmetric.

        If ``is_corresponding_model_azimuthally_symmetric`` is set to ``True``,
        then the distortion model corresponding to the coordinate transform
        parameters is azimuthally symmetric. Otherwise, the distortion model is
        not azimuthally symmetric.

        Note that ``is_corresponding_model_azimuthally_symmetric`` should be
        considered **read-only**.

        """
        return self._is_corresponding_model_azimuthally_symmetric



    @property
    def is_corresponding_model_trivial(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is trivial.

        We define a trivial distortion model to be one with a corresponding
        coordinate transformation that is equivalent to the identity
        transformation.

        If ``is_corresponding_model_trivial`` is set to ``True``, then the
        distortion model corresponding to the coordinate transformation
        parameters is trivial. Otherwise, the distortion model is not trivial.

        Note that ``is_corresponding_model_trivial`` should be considered
        **read-only**.

        """
        return self._is_corresponding_model_trivial



    @property
    def is_corresponding_model_standard(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is standard.

        See the documentation for the class
        :class:`distoptica.StandardCoordTransformParams` for a definition of a
        standard distortion model.

        If ``is_corresponding_model_standard`` is set to ``True``, then the
        distortion model corresponding to the coordinate transformation
        parameters is standard. Otherwise, the distortion model is not standard.

        Note that ``is_corresponding_model_standard`` should be considered
        **read-only**.

        """
        return self._is_corresponding_model_standard



def _check_and_convert_coord_transform_params(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    accepted_types = (CoordTransformParams, type(None))

    if isinstance(obj, accepted_types[1]):
        coord_transform_params = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        coord_transform_params = copy.deepcopy(obj)

    return coord_transform_params



def _pre_serialize_coord_transform_params(coord_transform_params):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_coord_transform_params(serializable_rep):
    coord_transform_params = \
        CoordTransformParams.de_pre_serialize(serializable_rep)

    return coord_transform_params



def _check_and_convert_max_num_iterations(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    max_num_iterations = czekitout.convert.to_positive_int(**kwargs)

    return max_num_iterations



def _pre_serialize_max_num_iterations(max_num_iterations):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_max_num_iterations(serializable_rep):
    max_num_iterations = serializable_rep

    return max_num_iterations



def _check_and_convert_initial_damping(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    initial_damping = czekitout.convert.to_positive_float(**kwargs)

    return initial_damping



def _pre_serialize_initial_damping(initial_damping):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_initial_damping(serializable_rep):
    initial_damping = serializable_rep

    return initial_damping



def _check_and_convert_factor_for_decreasing_damping(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = \
        {"obj": obj, "obj_name": obj_name}
    factor_for_decreasing_damping = \
        czekitout.convert.to_positive_float(**kwargs)

    return factor_for_decreasing_damping



def _pre_serialize_factor_for_decreasing_damping(factor_for_decreasing_damping):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_factor_for_decreasing_damping(serializable_rep):
    factor_for_decreasing_damping = serializable_rep

    return factor_for_decreasing_damping



def _check_and_convert_factor_for_increasing_damping(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = \
        {"obj": obj, "obj_name": obj_name}
    factor_for_increasing_damping = \
        czekitout.convert.to_positive_float(**kwargs)

    return factor_for_increasing_damping



def _pre_serialize_factor_for_increasing_damping(factor_for_increasing_damping):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_factor_for_increasing_damping(serializable_rep):
    factor_for_increasing_damping = serializable_rep

    return factor_for_increasing_damping



def _check_and_convert_improvement_tol(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    improvement_tol = czekitout.convert.to_positive_float(**kwargs)

    return improvement_tol



def _pre_serialize_improvement_tol(improvement_tol):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_improvement_tol(serializable_rep):
    improvement_tol = serializable_rep

    return improvement_tol



def _check_and_convert_rel_err_tol(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    rel_err_tol = czekitout.convert.to_positive_float(**kwargs)

    return rel_err_tol



def _pre_serialize_rel_err_tol(rel_err_tol):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_rel_err_tol(serializable_rep):
    rel_err_tol = serializable_rep

    return rel_err_tol



def _check_and_convert_plateau_tol(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    plateau_tol = czekitout.convert.to_nonnegative_float(**kwargs)

    return plateau_tol



def _pre_serialize_plateau_tol(plateau_tol):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_plateau_tol(serializable_rep):
    plateau_tol = serializable_rep

    return plateau_tol



def _check_and_convert_plateau_patience(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    plateau_patience = czekitout.convert.to_positive_int(**kwargs)

    return plateau_patience



def _pre_serialize_plateau_patience(plateau_patience):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_plateau_patience(serializable_rep):
    plateau_patience = serializable_rep

    return plateau_patience



_default_max_num_iterations = 10
_default_initial_damping = 1e-3
_default_factor_for_decreasing_damping = 9
_default_factor_for_increasing_damping = 11
_default_improvement_tol = 0.1
_default_rel_err_tol = 1e-2
_default_plateau_tol = 1e-3
_default_plateau_patience = 2



_cls_alias = fancytypes.PreSerializableAndUpdatable
class LeastSquaresAlgParams(_cls_alias):
    r"""Insert description here.

    Parameters
    ----------
    max_num_iterations : `int`, optional
        Insert description here.
    initial_damping : `float`, optional
        Insert description here.
    factor_for_decreasing_damping : `float`, optional
        Insert description here.
    factor_for_increase_damping : `float`, optional
        Insert description here.
    improvement_tol : `float`, optional
        Insert description here.
    rel_err_tol : `float`, optional
        Insert description here.
    plateau_tol : `float`, optional
        Insert description here.
    plateau_patience : `int`, optional
        Insert description here.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("max_num_iterations",
                        "initial_damping",
                        "factor_for_decreasing_damping",
                        "factor_for_increase_damping",
                        "improvement_tol",
                        "rel_err_tol",
                        "plateau_tol",
                        "plateau_patience")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}
    
    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 max_num_iterations=\
                 _default_max_num_iterations,
                 initial_damping=\
                 _default_initial_damping,
                 factor_for_decreasing_damping=\
                 _default_factor_for_decreasing_damping,
                 factor_for_increasing_damping=\
                 _default_factor_for_increasing_damping,
                 improvement_tol=\
                 _default_improvement_tol,
                 rel_err_tol=\
                 _default_rel_err_tol,
                 plateau_tol=\
                 _default_plateau_tol,
                 plateau_patience=\
                 _default_plateau_patience,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



def _check_and_convert_least_squares_alg_params(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    accepted_types = (LeastSquaresAlgParams, type(None))

    if isinstance(obj, accepted_types[1]):
        least_squares_alg_params = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        least_squares_alg_params = copy.deepcopy(obj)

    return least_squares_alg_params



def _pre_serialize_least_squares_alg_params(least_squares_alg_params):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_least_squares_alg_params(serializable_rep):
    least_squares_alg_params = \
        LeastSquaresAlgParams.de_pre_serialize(serializable_rep)

    return least_squares_alg_params



class _CoordTransformRightInverse(torch.nn.Module):
    def __init__(self, coord_transform, least_squares_alg_params):
        super().__init__()
        
        self.coord_transform_1 = coord_transform
        self.coord_transform_2 = self.copy_coord_transform(coord_transform)

        self.least_squares_alg_params_core_attrs = \
            least_squares_alg_params.get_core_attrs(deep_copy=False)

        self.forward_output = None

        return None



    def copy_coord_transform(self, coord_transform):
        coord_transform_center = coord_transform.center.numpy(force=True)
        radial_fourier_series = coord_transform.radial_fourier_series
        tangential_fourier_series = coord_transform.tangential_fourier_series

        radial_cosine_amplitudes = \
            radial_fourier_series.cosine_amplitudes
        coefficient_matrix = \
            radial_cosine_amplitudes.coefficient_matrix.numpy(force=True)
        radial_cosine_amplitudes = \
            _Polynomials(coefficient_matrix)

        radial_sine_amplitudes = \
            radial_fourier_series.sine_amplitudes
        coefficient_matrix = \
            radial_sine_amplitudes.coefficient_matrix.numpy(force=True)
        radial_sine_amplitudes = \
            _Polynomials(coefficient_matrix)

        tangential_cosine_amplitudes = \
            tangential_fourier_series.cosine_amplitudes
        coefficient_matrix = \
            tangential_cosine_amplitudes.coefficient_matrix.numpy(force=True)
        tangential_cosine_amplitudes = \
            _Polynomials(coefficient_matrix)

        tangential_sine_amplitudes = \
            tangential_fourier_series.sine_amplitudes
        coefficient_matrix = \
            tangential_sine_amplitudes.coefficient_matrix.numpy(force=True)
        tangential_sine_amplitudes = \
            _Polynomials(coefficient_matrix)

        kwargs = {"cosine_amplitudes": radial_cosine_amplitudes,
                  "sine_amplitudes": radial_sine_amplitudes}
        radial_fourier_series = _FourierSeries(**kwargs)

        kwargs = {"cosine_amplitudes": tangential_cosine_amplitudes,
                  "sine_amplitudes": tangential_sine_amplitudes}
        tangential_fourier_series = _FourierSeries(**kwargs)

        kwargs = {"center": coord_transform_center,
                  "radial_fourier_series": radial_fourier_series,
                  "tangential_fourier_series": tangential_fourier_series}
        coord_transform_copy = _CoordTransform(**kwargs)

        return coord_transform_copy



    def initialize_levenberg_marquardt_alg_variables(self, inputs):
        q_x = inputs["q_x"]
        q_y = inputs["q_y"]
        self.q = torch.zeros((2,)+q_x.shape, dtype=q_x.dtype, device=q_x.device)
        self.q[0] = q_x
        self.q[1] = q_y
        self.q_sq = q_x*q_x + q_y*q_y

        initial_damping = \
            self.least_squares_alg_params_core_attrs["initial_damping"]
        
        self.damping = initial_damping * torch.ones_like(q_x)
        self.mask_1 = torch.zeros_like(q_x)
        self.mask_2 = torch.zeros_like(q_x)
        
        self.p_1 = torch.zeros_like(self.q)
        self.p_1[0] = q_x
        self.p_1[1] = q_y

        self.p_2 = torch.zeros_like(self.p_1)

        self.coord_transform_inputs_1 = dict()
        self.coord_transform_inputs_2 = dict()

        kwargs = {"coord_transform_inputs": self.coord_transform_inputs_1,
                  "p": self.p_1}
        self.update_coord_transform_inputs(**kwargs)

        kwargs = {"coord_transform_inputs": self.coord_transform_inputs_1,
                  "coord_transform": self.coord_transform_1}
        self.q_hat_1 = self.eval_q_hat(**kwargs)        
        self.chi_1 = self.eval_chi(q_hat=self.q_hat_1)
        self.chi_sq_1 = self.eval_chi_sq(chi=self.chi_1)

        self.q_hat_2 = torch.zeros_like(self.q_hat_1)
        self.chi_2 = torch.zeros_like(self.chi_1)
        self.chi_sq_2 = torch.zeros_like(self.chi_sq_1)

        kwargs = {"coord_transform_inputs": self.coord_transform_inputs_1}
        self.J = self.eval_J(**kwargs)
        self.H = self.eval_H()
        self.g = self.eval_g()
        self.D = torch.zeros_like(self.H)
        self.h = torch.zeros_like(self.g)

        self.convergence_map = torch.zeros_like(self.q_sq, dtype=bool)
        self.best_rel_err_sum = float("inf")
        self.num_iterations_of_plateauing = 0

        return None



    def update_coord_transform_inputs(self, coord_transform_inputs, p):
        kwargs = {"coord_transform_inputs": coord_transform_inputs,
                  "coord_transform": self.coord_transform_1,
                  "u_x": p[0],
                  "u_y": p[1]}
        _update_coord_transform_input_subset_1(**kwargs)

        kwargs = {"coord_transform_inputs": coord_transform_inputs,
                  "coord_transform": self.coord_transform_1}
        _update_coord_transform_input_subset_2(**kwargs)

        return None



    def eval_q_hat(self, coord_transform_inputs, coord_transform):
        kwargs = {"inputs": coord_transform_inputs}

        obj_set = (coord_transform.radial_fourier_series.cosine_amplitudes,
                   coord_transform.radial_fourier_series.sine_amplitudes,
                   coord_transform.tangential_fourier_series.cosine_amplitudes,
                   coord_transform.tangential_fourier_series.sine_amplitudes,
                   coord_transform.radial_fourier_series,
                   coord_transform.tangential_fourier_series)
        for obj in obj_set:
            obj.forward_output = obj.eval_forward_output(**kwargs)
                
        q_hat = coord_transform.eval_forward_output(**kwargs)

        return q_hat



    def eval_chi(self, q_hat):
        chi = self.q - q_hat

        return chi



    def eval_chi_sq(self, chi):
        chi_sq = torch.einsum("nij, nij -> ij", chi, chi)

        return chi_sq



    def eval_J(self, coord_transform_inputs):
        coord_transform = self.coord_transform_1

        kwargs = {"inputs": coord_transform_inputs}

        obj_set = \
            (coord_transform.radial_fourier_series.cosine_amplitudes,
             coord_transform.radial_fourier_series.sine_amplitudes,
             coord_transform.tangential_fourier_series.cosine_amplitudes,
             coord_transform.tangential_fourier_series.sine_amplitudes)
        for obj in obj_set:
            obj.derivative_wrt_u_r = \
                obj.eval_derivative_wrt_u_r(**kwargs)

        obj_set = \
            (coord_transform.radial_fourier_series,
             coord_transform.tangential_fourier_series)
        for obj in obj_set:
            obj.derivative_wrt_u_r = \
                obj.eval_derivative_wrt_u_r(**kwargs)
            obj.derivative_wrt_u_theta = \
                obj.eval_derivative_wrt_u_theta(**kwargs)
        
        J = coord_transform.eval_jacobian(**kwargs)

        return J



    def eval_H(self):
        H = torch.einsum("lnij, lmij -> nmij", self.J, self.J)

        return H



    def eval_g(self):
        g = torch.einsum("mnij, mij -> nij", self.J, self.chi_1)

        return g



    def eval_forward_output(self, inputs):
        if len(inputs) != 0:
            self.initialize_levenberg_marquardt_alg_variables(inputs)

        u_x, u_y = self.calc_u_x_and_u_y_via_levenberg_marquardt_alg()

        output_tensor_shape = (2,) + u_x.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=u_x.dtype,
                                    device=u_x.device)
        output_tensor[0] = u_x
        output_tensor[1] = u_y

        return output_tensor



    def calc_u_x_and_u_y_via_levenberg_marquardt_alg(self, inputs):
        iteration_idx = 0

        max_num_iterations = \
            self.least_squares_alg_params_core_attrs["max_num_iterations"]

        try:
            for iteration_idx in range(max_num_iterations):
                alg_has_converged = self.perform_levenberg_marquardt_alg_step()
                if alg_has_converged:
                    break

            alg_did_not_converged = (not alg_has_converged)
            if alg_did_not_converged:
                raise

        except:
            unformatted_err_msg = _coord_transform_right_inverse_err_msg_1
            err_msg = unformatted_err_msg.format(max_num_iterations)
            raise RuntimeError(err_msg)

        u_x = self.p_1[0]
        u_y = self.p_1[1]

        self.del_subset_of_levenberg_marquardt_alg_variables()

        return u_x, u_y



    def perform_levenberg_marquardt_alg_step(self):
        self.update_D()
        self.h[:] = self.eval_h()[:]
            
        self.p_2[:] = self.p_1[:] + self.h[:]

        kwargs = {"coord_transform_inputs": self.coord_transform_inputs_2,
                  "p": self.p_2}
        self.update_coord_transform_inputs(**kwargs)

        kwargs = {"coord_transform_inputs": self.coord_transform_inputs_2,
                  "coord_transform": self.coord_transform_2}
        self.q_hat_2[:] = self.eval_q_hat(**kwargs)[:]
        self.chi_2[:] = self.eval_chi(q_hat=self.q_hat_2)[:]
        self.chi_sq_2[:] = self.eval_chi_sq(chi=self.chi_2)[:]

        self.update_masks_1_and_2()
        self.apply_masks_1_and_2()

        current_rel_err = self.eval_current_rel_err()
        current_rel_err_sum = torch.sum(current_rel_err).item()

        kwargs = {"current_rel_err": current_rel_err,
                  "current_rel_err_sum": current_rel_err_sum}
        alg_has_converged = self.levenberg_marquardt_alg_has_converged(**kwargs)
        alg_has_not_converged = (not alg_has_converged)

        if alg_has_not_converged:
            kwargs = {"coord_transform_inputs": self.coord_transform_inputs_1}
            self.J[:] = self.eval_J(**kwargs)[:]
            self.H[:] = self.eval_H()[:]
            self.g[:] = self.eval_g()[:]

        self.best_rel_err_sum = min(self.best_rel_err_sum, current_rel_err_sum)

        rel_err_tol = self.least_squares_alg_params_core_attrs["rel_err_tol"]
        self.convergence_map[:] = (current_rel_err < rel_err_tol)[:]

        return alg_has_converged



    def update_D(self):
        self.D[0, 0] = self.damping*self.H[0, 0]
        self.D[1, 1] = self.damping*self.H[1, 1]

        return None



    def eval_h(self):
        A = self.H + self.D

        V, pseudo_inv_of_Sigma, U = self.eval_V_and_pseudo_inv_of_Sigma_and_U(A)

        b = torch.einsum("mnij, mij -> nij", U, self.g)
        z = torch.einsum("nmij, mij -> nij", pseudo_inv_of_Sigma, b)
        h = torch.einsum("nmij, mij -> nij", V, z)
        
        return h



    def eval_V_and_pseudo_inv_of_Sigma_and_U(self, A):
        method_alias = \
            self.eval_mask_subset_1_and_denom_set_and_abs_denom_set_and_lambdas
        mask_subset_1, denom_set, abs_denom_set, lambdas = \
            method_alias(A)

        mask_subset_2 = self.eval_mask_subset_2(mask_subset_1, abs_denom_set)

        V = self.eval_V(A, mask_subset_1, mask_subset_2, denom_set)

        pseudo_inv_of_Sigma = self.eval_pseudo_inv_of_Sigma(A, lambdas)

        U = self.eval_U(A, V, pseudo_inv_of_Sigma)

        return V, pseudo_inv_of_Sigma, U



    def eval_mask_subset_1_and_denom_set_and_abs_denom_set_and_lambdas(self, A):
        a, b, c = self.eval_a_b_and_c(A)

        a_minus_c = a-c
        b_sq = b*b

        lambda_sum_over_2 = (a+c)/2
        lambda_diff_over_2 = torch.sqrt(a_minus_c*a_minus_c + 4*b_sq)/2

        lambda_0 = lambda_sum_over_2+lambda_diff_over_2
        lambda_1 = torch.clamp(lambda_sum_over_2-lambda_diff_over_2, min=0)
        lambdas = (lambda_0, lambda_1)

        abs_b = torch.abs(b)
        lambda_0_minus_a = lambda_0-a
        lambda_0_minus_c = lambda_0-c
        lambda_1_minus_a = lambda_1-a
        lambda_1_minus_c = lambda_1-c
        abs_lambda_0_minus_a = torch.abs(lambda_0_minus_a)
        abs_lambda_0_minus_c = torch.abs(lambda_0_minus_c)
        abs_lambda_1_minus_a = torch.abs(lambda_1_minus_a)
        abs_lambda_1_minus_c = torch.abs(lambda_1_minus_c)

        mask_subset_1 = self.eval_mask_subset_1(abs_b,
                                                abs_lambda_0_minus_a,
                                                abs_lambda_0_minus_c,
                                                abs_lambda_1_minus_a,
                                                abs_lambda_1_minus_c,
                                                lambda_sum_over_2,
                                                lambda_diff_over_2)
        mask_9, mask_10, _, _ = mask_subset_1

        denom_0 = b
        denom_1 = mask_9*lambda_0_minus_a + mask_10*lambda_1_minus_a
        denom_2 = mask_9*lambda_0_minus_c + mask_10*lambda_1_minus_c
        
        abs_denom_0 = abs_b
        abs_denom_1 = (mask_9*abs_lambda_0_minus_a
                       + mask_10*abs_lambda_1_minus_a)
        abs_denom_2 = (mask_9*abs_lambda_0_minus_c
                       + mask_10*abs_lambda_1_minus_c)

        denom_set = (denom_0, denom_1, denom_2)
        abs_denom_set = (abs_denom_0, abs_denom_1, abs_denom_2)

        return mask_subset_1, denom_set, abs_denom_set, lambdas



    def eval_a_b_and_c(self, A):
        a = A[0, 0]*A[0, 0] + A[1, 0]*A[1, 0]
        b = A[0, 0]*A[0, 1] + A[1, 0]*A[1, 1]
        c = A[0, 1]*A[0, 1] + A[1, 1]*A[1, 1]

        return a, b, c



    def eval_mask_subset_1(self,
                           abs_b,
                           abs_lambda_0_minus_a,
                           abs_lambda_0_minus_c,
                           abs_lambda_1_minus_a,
                           abs_lambda_1_minus_c,
                           lambda_sum_over_2,
                           lambda_diff_over_2):
        M_0 = abs_b
        M_1 = abs_lambda_0_minus_a
        M_2 = abs_lambda_0_minus_c
        M_3 = abs_lambda_1_minus_a
        M_4 = abs_lambda_1_minus_c

        mask_3 = (M_1 >= M_3)
        mask_4 = (M_2 >= M_4)

        M_5 = mask_3*M_1 + (~mask_3)*M_3
        M_6 = mask_4*M_2 + (~mask_4)*M_4

        mask_5 = (M_5 >= M_6)
        mask_6 = ~mask_5

        M_7 = mask_5*M_5 + mask_6*M_6

        mask_7 = (M_0 >= M_7)
        mask_8 = ~mask_7

        mask_9 = mask_7*mask_3 + mask_8*(mask_5*mask_3 + mask_6*mask_4)
        mask_10 = ~mask_9

        tol = 2*np.finfo(np.float32).eps
        mask_11 = (lambda_diff_over_2/lambda_sum_over_2 > tol)
        mask_12 = ~mask_11

        mask_subset_1 = (mask_9, mask_10, mask_11, mask_12)

        return mask_subset_1



    def eval_mask_subset_2(self, mask_subset_1, abs_denom_set):
        mask_11 = mask_subset_1[2]
        abs_denom_0, abs_denom_1, abs_denom_2 = abs_denom_set

        mask_13 = (abs_denom_0 >= abs_denom_1)
        mask_14 = ~mask_13
        
        M_8 = mask_13*abs_denom_0 + mask_14*abs_denom_1
        
        mask_15 = (abs_denom_2 >= M_8)
        mask_16 = ~mask_15

        mask_13[:] = (mask_13*mask_16)[:]
        mask_14[:] = (mask_14*mask_16)[:]

        mask_16[:] = (mask_13+mask_15 > 0)[:]
        mask_17 = ~mask_16
        
        mask_13[:] = (mask_11*mask_13)[:]
        mask_14[:] = (mask_11*mask_14)[:]
        mask_15[:] = (mask_11*mask_15)[:]

        mask_subset_2 = (mask_13, mask_14, mask_15, mask_16, mask_17)
        
        return mask_subset_2



    def eval_V(self, A, mask_subset_1, mask_subset_2, denom_set):
        V = torch.zeros_like(A)

        mask_9, mask_10, _, mask_12 = mask_subset_1
        mask_13, mask_14, mask_15, mask_16, mask_17 = mask_subset_2
        denom_0, denom_1, denom_2 = denom_set

        M_9 = ((mask_13
                + (mask_14*denom_0
                   / (mask_14*denom_1 + mask_12 + mask_13 + mask_15))
                + mask_15)
               + mask_12*mask_16)
        M_10 = (((mask_13*denom_1
                  / (mask_13*denom_0 + mask_12 + mask_14 + mask_15))
                 + mask_14
                 + (mask_15*denom_0
                    / (mask_15*denom_2 + mask_12 + mask_13 + mask_14)))
                + mask_12*mask_17)
        M_11 = torch.sqrt(M_9*M_9 + M_10*M_10)

        M_9[:] = (M_9/M_11)[:]
        M_10[:] = (M_10/M_11)[:]

        V[0, 0] = mask_9*M_9 + mask_10*M_10
        V[1, 0] = mask_9*M_10 - mask_10*M_9
        V[0, 1] = mask_10*M_9 + mask_9*M_10
        V[1, 1] = mask_10*M_10 - mask_9*M_9

        return V



    def eval_pseudo_inv_of_Sigma(self, A, lambdas):
        pseudo_inv_of_Sigma = torch.zeros_like(A)

        lambda_0, lambda_1 = lambdas
        
        sigma_0 = torch.sqrt(lambda_0)
        sigma_1 = torch.sqrt(lambda_1)
        
        tol = 2*np.finfo(np.float32).eps
        mask_18 = (sigma_1/sigma_0 > tol)
        mask_19 = ~mask_18

        pseudo_inv_of_Sigma[0, 0] = 1 / sigma_0
        pseudo_inv_of_Sigma[1, 1] = mask_18 / (sigma_1 + mask_19)

        return pseudo_inv_of_Sigma



    def eval_U(self, A, V, pseudo_inv_of_Sigma):
        A_V = torch.einsum("nlij, lmij -> nmij", A, V)
        U = torch.einsum("nlij, lmij -> nmij", A_V, pseudo_inv_of_Sigma)

        return U



    def eval_P(self, A):
        A_0_norm_sq = A[0, 0]*A[0, 0] + A[1, 0]*A[1, 0]
        A_1_norm_sq = A[0, 1]*A[0, 1] + A[1, 1]*A[1, 1]

        P = torch.zeros_like(A)
        P[0, 0] = (A_0_norm_sq >= A_1_norm_sq)
        P[0, 1] = 1-P[0, 0]
        P[1, 0] = P[0, 1]
        P[1, 1] = P[0, 0]

        return P



    def eval_alpha(self, M):
        alpha = -(torch.sign(M[0, 0])
                  * torch.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0]))

        return alpha



    def eval_Q(self, M, alpha):
        u = torch.zeros_like(M[:, 0])
        u[:] = M[:, 0]
        u[0] -= alpha

        reciprocal_of_u_norm = 1.0 / torch.sqrt(u[0]*u[0] + u[1]*u[1])

        v = torch.einsum("ij, nij -> nij", reciprocal_of_u_norm, u)

        Q = torch.zeros_like(M)
        Q[0, 0] = 1 - 2*v[0]*v[0]
        Q[0, 1] = -2*v[0]*v[1]
        Q[1, 0] = Q[0, 1]
        Q[1, 1] = 1 - 2*v[1]*v[1]

        return Q



    def eval_R(self, M, alpha, Q):
        R = torch.zeros_like(M)
        R[0, 0] = alpha
        R[0, 1] = Q[0, 0]*M[0, 1] + Q[0, 1]*M[1, 1]
        R[1, 1] = Q[1, 0]*M[0, 1] + Q[1, 1]*M[1, 1]
        
        return R



    def eval_z(self, R, b):
        z = torch.zeros_like(b)
        z[1] = b[1] / R[1, 1]
        z[0] = (b[0] - R[0, 1]*z[1]) / R[0, 0]

        return z



    def update_masks_1_and_2(self):
        D_h_plus_g = torch.einsum("nmij, mij -> nij", self.D, self.h) + self.g
        
        rho_numerator = self.chi_sq_1 - self.chi_sq_2
        rho_denominator = torch.abs(torch.einsum("nij, nij -> ij",
                                                 self.h,
                                                 D_h_plus_g))
        rho = rho_numerator/rho_denominator

        rol_tol = self.least_squares_alg_params_core_attrs["improvement_tol"]

        self.mask_1[:] = (rho > rol_tol)[:]
        self.mask_2[:] = (1-self.mask_1)[:]

        return None



    def apply_masks_1_and_2(self):
        self.update_damping()

        attr_subset_1 = (self.p_1, self.q_hat_1, self.chi_1)
        attr_subset_2 = (self.p_2, self.q_hat_2, self.chi_2)

        for attr_1, attr_2 in zip(attr_subset_1, attr_subset_2):
            attr_1[:] = (torch.einsum("ij, nij -> nij",
                                      self.mask_1,
                                      attr_2)
                         + torch.einsum("ij, nij -> nij",
                                        self.mask_2,
                                        attr_1))[:]

        self.chi_sq_1[:] = (self.mask_1*self.chi_sq_2
                            + self.mask_2*self.chi_sq_1)[:]

        for key in self.coord_transform_inputs_1:
            dict_elem_1 = self.coord_transform_inputs_1[key]
            dict_elem_2 = self.coord_transform_inputs_2[key]
            
            if (("powers" in key) or ("thetas" in key)):
                dict_elem_1[:] = (torch.einsum("ij, nij -> nij",
                                               self.mask_1,
                                               dict_elem_2)
                                  + torch.einsum("ij, nij -> nij",
                                                 self.mask_2,
                                                 dict_elem_1))[:]
            else:
                dict_elem_1[:] = (self.mask_1*dict_elem_2
                                  + self.mask_2*dict_elem_1)[:]

        self.update_coord_transform_1_forward_output_cmpnts()

        return None



    def update_damping(self):
        attr_name = "factor_for_decreasing_damping"
        L_down = self.least_squares_alg_params_core_attrs[attr_name]

        attr_name = "factor_for_increasing_damping"
        L_up = self.least_squares_alg_params_core_attrs[attr_name]

        min_possible_vals = torch.maximum(self.damping/L_down,
                                          (1e-7)*torch.ones_like(self.damping))
        max_possible_vals = torch.minimum(self.damping*L_up,
                                          (1e7)*torch.ones_like(self.damping))
        
        self.damping[:] = (self.mask_1*min_possible_vals
                           + self.mask_2*max_possible_vals)[:]

        return None



    def update_coord_transform_1_forward_output_cmpnts(self):
        obj_set_1 = \
            (self.coord_transform_1.radial_fourier_series.cosine_amplitudes,
             self.coord_transform_1.radial_fourier_series.sine_amplitudes,
             self.coord_transform_1.tangential_fourier_series.cosine_amplitudes,
             self.coord_transform_1.tangential_fourier_series.sine_amplitudes,
             self.coord_transform_1.radial_fourier_series,
             self.coord_transform_1.tangential_fourier_series)
        obj_set_2 = \
            (self.coord_transform_2.radial_fourier_series.cosine_amplitudes,
             self.coord_transform_2.radial_fourier_series.sine_amplitudes,
             self.coord_transform_2.tangential_fourier_series.cosine_amplitudes,
             self.coord_transform_2.tangential_fourier_series.sine_amplitudes,
             self.coord_transform_2.radial_fourier_series,
             self.coord_transform_2.tangential_fourier_series)

        for obj_1, obj_2 in zip(obj_set_1, obj_set_2):
            forward_output_cmpnt_1 = obj_1.forward_output
            forward_output_cmpnt_2 = obj_2.forward_output
            
            if isinstance(obj_1, _Polynomials):
                forward_output_cmpnt_1[:] = \
                    (torch.einsum("ij, nij -> nij",
                                  self.mask_1,
                                  forward_output_cmpnt_2)
                     + torch.einsum("ij, nij -> nij",
                                    self.mask_2,
                                    forward_output_cmpnt_1))[:]
            else:
                forward_output_cmpnt_1[:] = \
                    (self.mask_1*forward_output_cmpnt_2
                     + self.mask_2*forward_output_cmpnt_1)[:]

        return None



    def eval_current_rel_err(self):
        current_rel_err = torch.sqrt(self.chi_sq_1/self.q_sq)

        return current_rel_err



    def levenberg_marquardt_alg_has_converged(self,
                                              current_rel_err,
                                              current_rel_err_sum):
        rel_err_tol = \
            self.least_squares_alg_params_core_attrs["rel_err_tol"]
        plateau_tol = \
            self.least_squares_alg_params_core_attrs["plateau_tol"]
        plateau_patience = \
            self.least_squares_alg_params_core_attrs["plateau_patience"]
        
        plateau_metric = current_rel_err_sum/self.best_rel_err_sum
        if plateau_metric < 1-plateau_tol:
            self.num_iterations_of_plateauing = 0
        else:
            self.num_iterations_of_plateauing += 1

        if self.num_iterations_of_plateauing >= plateau_patience:
            mask_20 = ((0 <= self.p_1[0]) * (self.p_1[0] <= 1)
                       * (0 <= self.p_1[1]) * (self.p_1[1] <= 1))
        else:
            mask_20 = torch.ones_like(current_rel_err)

        alg_has_converged = torch.all(mask_20*current_rel_err < rel_err_tol)

        return alg_has_converged



    def del_subset_of_levenberg_marquardt_alg_variables(self):
        del self.q
        del self.q_sq
        del self.damping
        del self.mask_1
        del self.mask_2
        del self.p_1
        del self.p_2
        del self.coord_transform_inputs_1
        del self.coord_transform_inputs_2
        del self.q_hat_1
        del self.chi_1
        del self.chi_sq_1
        del self.q_hat_2
        del self.chi_2
        del self.chi_sq_2
        del self.J
        del self.H
        del self.g
        del self.D
        del self.h
        del self.num_iterations_of_plateauing

        return None



    def forward(self, inputs):
        output_tensor = self.eval_forward_output(inputs)

        return output_tensor



    def eval_abs_det_J(self):
        abs_det_J = torch.abs(self.J[0, 0]*self.J[1, 1]
                              - self.J[1, 0]*self.J[0, 1])

        return abs_det_J



def _get_device(device_name):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)

    return device



def _check_and_convert_u_x_and_u_y(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    u_x, u_y = obj

    params["real_torch_matrix"] = u_x
    params["name_of_alias_of_real_torch_matrix"] = "u_x"
    u_x = _check_and_convert_real_torch_matrix(params)

    params["real_torch_matrix"] = u_y
    params["name_of_alias_of_real_torch_matrix"] = "u_y"
    u_y = _check_and_convert_real_torch_matrix(params)

    del params["real_torch_matrix"]
    del params["name_of_alias_of_real_torch_matrix"]

    if u_x.shape != u_y.shape:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format("u_x", "u_y")
        raise ValueError(err_msg)

    u_x_and_u_y = (u_x, u_y)

    return u_x_and_u_y



def _check_and_convert_real_torch_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    name_of_alias_of_real_torch_matrix = \
        params["name_of_alias_of_real_torch_matrix"]
    
    if isinstance(obj, torch.Tensor):
        try:
            if len(obj.shape) != 2:
                raise
            if obj.dtype != torch.float32:
                obj = obj.float()
        except:
            unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
            unformatted_err_msg_1.format(name_of_alias_of_real_torch_matrix)
            raise ValueError(err_msg)

        if obj.device != params["device"]:
            obj = obj.to(params["device"])

        real_torch_matrix = obj

    else:
        kwargs = {"obj": obj, "obj_name": name_of_alias_of_real_torch_matrix}
        real_numpy_matrix = czekitout.convert.to_real_numpy_matrix(**kwargs)

        real_torch_matrix = torch.tensor(real_numpy_matrix,
                                         dtype=torch.float32,
                                         device=params["device"])

    return real_torch_matrix



_default_u_x = ((0.5,),)
_default_u_y = _default_u_x



def _generate_coord_meshgrid(sampling_grid_dims_in_pixels, device):
    j_range = torch.arange(sampling_grid_dims_in_pixels[0], device=device)
    i_range = torch.arange(sampling_grid_dims_in_pixels[1], device=device)

    horizontal_coords_of_grid = (j_range + 0.5) / j_range.numel()
    vertical_coords_of_grid = 1 - (i_range + 0.5) / i_range.numel()
    pair_of_1d_coord_arrays = (horizontal_coords_of_grid,
                               vertical_coords_of_grid)

    coord_meshgrid = torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")

    return coord_meshgrid



def _check_and_convert_q_x_and_q_y(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    q_x, q_y = obj

    params["real_torch_matrix"] = q_x
    params["name_of_alias_of_real_torch_matrix"] = "q_x"
    q_x = _check_and_convert_real_torch_matrix(params)

    params["real_torch_matrix"] = q_y
    params["name_of_alias_of_real_torch_matrix"] = "q_y"
    q_y = _check_and_convert_real_torch_matrix(params)

    del params["real_torch_matrix"]
    del params["name_of_alias_of_real_torch_matrix"]

    if q_x.shape != q_y.shape:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format("q_x", "q_y")
        raise ValueError(err_msg)

    q_x_and_q_y = (q_x, q_y)

    return q_x_and_q_y



_default_q_x = ((0.5,),)
_default_q_y = _default_q_x



def _calc_minimum_frame_to_mask_all_zero_valued_elems(mat):
    if mat.sum().item() != mat.numel():
        area_of_largest_rectangle_in_mat = 0
        minimum_frame_to_mask_all_zero_valued_elems = np.zeros((4,), dtype=int)
        num_rows = mat.shape[0]
        mat = mat.cpu().detach().numpy()
        current_histogram = np.zeros_like(mat[0])

        for row_idx, row in enumerate(mat):
            current_histogram = row*(current_histogram+1)

            func_alias = \
                _calc_mask_frame_and_area_of_largest_rectangle_in_histogram
            kwargs = \
                {"histogram": current_histogram,
                 "max_possible_rectangle_height": row_idx+1}
            mask_frame_and_area = \
                func_alias(**kwargs)
            mask_frame_of_largest_rectangle_in_current_histogram = \
                mask_frame_and_area[0]
            area_of_largest_rectangle_in_current_histogram = \
                mask_frame_and_area[1]

            if (area_of_largest_rectangle_in_current_histogram
                > area_of_largest_rectangle_in_mat):
                area_of_largest_rectangle_in_mat = \
                    area_of_largest_rectangle_in_current_histogram
                minimum_frame_to_mask_all_zero_valued_elems = \
                    mask_frame_of_largest_rectangle_in_current_histogram
                minimum_frame_to_mask_all_zero_valued_elems[2] = \
                    (num_rows-1)-row_idx
            
        minimum_frame_to_mask_all_zero_valued_elems = \
            tuple(minimum_frame_to_mask_all_zero_valued_elems)
    else:
        minimum_frame_to_mask_all_zero_valued_elems = (0, 0, 0, 0)

    return minimum_frame_to_mask_all_zero_valued_elems



def _calc_mask_frame_and_area_of_largest_rectangle_in_histogram(
        histogram, max_possible_rectangle_height):
    stack = []
    area_of_largest_rectangle = 0
    mask_frame_of_largest_rectangle = np.zeros((4,), dtype=int)
    idx_1 = 0

    num_bins = len(histogram)

    while not ((idx_1 == num_bins) and (len(stack) == 0)):
        if (((len(stack) == 0)
             or (histogram[stack[-1]] <= histogram[idx_1%num_bins]))
            and (idx_1 < num_bins)):
            stack.append(idx_1)
            idx_1 += 1
        else:
            top_of_stack = stack.pop()
            
            idx_2 = top_of_stack
            idx_3 = (stack[-1] if (len(stack) > 0) else -1)
            
            height_of_current_rectangle = histogram[idx_2].item()
            width_of_current_rectangle = idx_1-idx_3-1
            area_of_current_rectangle = (height_of_current_rectangle
                                         * width_of_current_rectangle)

            if area_of_current_rectangle > area_of_largest_rectangle:
                area_of_largest_rectangle = \
                    area_of_current_rectangle
                mask_frame_of_largest_rectangle[0] = \
                    idx_3+1
                mask_frame_of_largest_rectangle[1] = \
                    num_bins-idx_1
                mask_frame_of_largest_rectangle[2] = \
                    0
                mask_frame_of_largest_rectangle[3] = \
                    max_possible_rectangle_height-height_of_current_rectangle

    return mask_frame_of_largest_rectangle, area_of_largest_rectangle



def _check_and_convert_images(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]
    
    name_of_alias_of_images = params["name_of_alias_of_images"]

    try:
        if not isinstance(obj, torch.Tensor):
            kwargs = {"obj": obj, "obj_name": name_of_alias_of_images}
            obj = czekitout.convert.to_real_numpy_array(**kwargs)

            obj = torch.tensor(obj,
                               dtype=torch.float32,
                               device=params["device"])
    
        if (len(obj.shape) >= 2) and (len(obj.shape) <= 4):
            if len(obj.shape) == 2:
                obj = torch.unsqueeze(obj, dim=0)
            if len(obj.shape) == 3:
                obj = torch.unsqueeze(obj, dim=0)
        else:
            raise
            
        if obj.dtype != torch.float32:
            obj = obj.float()

        if obj.device != params["device"]:
            obj = obj.to(params["device"])

        images = obj

    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg_1.format(name_of_alias_of_images)
        raise ValueError(err_msg)

    return images



_default_undistorted_images = ((0.0,),)
_default_distorted_images = ((0.0,),)



def _check_and_convert_sampling_grid_dims_in_pixels(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = \
        {"obj": obj, "obj_name": obj_name}
    sampling_grid_dims_in_pixels = \
        czekitout.convert.to_pair_of_positive_ints(**kwargs)

    return sampling_grid_dims_in_pixels



def _pre_serialize_sampling_grid_dims_in_pixels(sampling_grid_dims_in_pixels):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_sampling_grid_dims_in_pixels(serializable_rep):
    sampling_grid_dims_in_pixels = serializable_rep

    return sampling_grid_dims_in_pixels



def _check_and_convert_device_name(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    if obj is None:
        device_name = obj
    else:
        try:
            kwargs = {"obj": obj, "obj_name": obj_name}
            device_name = czekitout.convert.to_str_from_str_like(**kwargs)
            
            torch.device(device_name)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise ValueError(err_msg)

    return device_name



def _pre_serialize_device_name(device_name):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_device_name(serializable_rep):
    device_name = serializable_rep

    return device_name



_default_coord_transform_params = None
_default_sampling_grid_dims_in_pixels = (512, 512)
_default_device_name = None
_default_least_squares_alg_params = None



_cls_alias = fancytypes.PreSerializableAndUpdatable
class DistortionModel(_cls_alias):
    r"""Insert description here.

    Parameters
    ----------
    coord_transform_params : :class:`distoptica.CoordTransformParams` | `None`, optional
        Insert description here.
    sampling_grid_dims_in_pixels : `array_like` (`int`, shape=(2,)), optional
        Insert description here.
    device_name : `str` | `None`, optional
        Insert description here.
    least_squares_alg_params : :class:`distoptica.LeastSquaresAlgParams` | `None`, optional
        Insert description here.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("coord_transform_params",
                        "sampling_grid_dims_in_pixels",
                        "device_name",
                        "least_squares_alg_params")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}
    
    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 coord_transform_params=\
                 _default_coord_transform_params,
                 sampling_grid_dims_in_pixels=\
                 _default_sampling_grid_dims_in_pixels,
                 device_name=\
                 _default_device_name,
                 least_squares_alg_params=\
                 _default_least_squares_alg_params,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def execute_post_core_attrs_update_actions(self):
        r"""Execute the sequence of actions that follows immediately after 
        updating the core attributes.

        """
        self._device = self._get_device()

        self._coord_transform_right_inverse = \
            self._generate_coord_transform_right_inverse()
        self._coord_transform_right_inverse = \
            self._coord_transform_right_inverse.to(self._device)
        
        self._coord_transform_right_inverse.eval()

        self_core_attrs = self.get_core_attrs(deep_copy=False)

        coord_transform_params = \
            self_core_attrs["coord_transform_params"]
        self._is_azimuthally_symmetric = \
            coord_transform_params._is_corresponding_model_azimuthally_symmetric
        self._is_trivial = \
            coord_transform_params._is_corresponding_model_trivial

        self._cached_u_x = None
        self._cached_u_y = None
        self._cached_q_x = None
        self._cached_q_y = None

        self._cached_convergence_map = None

        self._cached_abs_det_J = None
        self._cached_abs_det_tilde_J = None

        return None



    def _get_device(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        device_name = self_core_attrs["device_name"]
        device = _get_device(device_name)

        return device



    def _generate_coord_transform_right_inverse(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        least_squares_alg_params = self_core_attrs["least_squares_alg_params"]
        
        coord_transform = self._generate_coord_transform()

        kwargs = {"coord_transform": coord_transform,
                  "least_squares_alg_params": least_squares_alg_params}
        coord_transform_right_inverse = _CoordTransformRightInverse(**kwargs)

        return coord_transform_right_inverse



    def _generate_coord_transform(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        coord_transform_params = self_core_attrs["coord_transform_params"]

        coord_transform_params_core_attrs = \
            coord_transform_params.get_core_attrs(deep_copy=False)

        coord_transform_center = coord_transform_params_core_attrs["center"]
        coefficient_matrices = {key: coord_transform_params_core_attrs[key]
                                for key in coord_transform_params_core_attrs
                                if "coefficient_matrix" in key}

        coefficient_matrix = \
            coefficient_matrices["radial_cosine_coefficient_matrix"]
        radial_cosine_amplitudes = \
            _Polynomials(coefficient_matrix)

        coefficient_matrix = \
            coefficient_matrices["radial_sine_coefficient_matrix"]
        radial_sine_amplitudes = \
            _Polynomials(coefficient_matrix)

        coefficient_matrix = \
            coefficient_matrices["tangential_cosine_coefficient_matrix"]
        tangential_cosine_amplitudes = \
            _Polynomials(coefficient_matrix)

        coefficient_matrix = \
            coefficient_matrices["tangential_sine_coefficient_matrix"]
        tangential_sine_amplitudes = \
            _Polynomials(coefficient_matrix)

        kwargs = {"cosine_amplitudes": radial_cosine_amplitudes,
                  "sine_amplitudes": radial_sine_amplitudes}
        radial_fourier_series = _FourierSeries(**kwargs)

        kwargs = {"cosine_amplitudes": tangential_cosine_amplitudes,
                  "sine_amplitudes": tangential_sine_amplitudes}
        tangential_fourier_series = _FourierSeries(**kwargs)

        kwargs = {"center": coord_transform_center,
                  "radial_fourier_series": radial_fourier_series,
                  "tangential_fourier_series": tangential_fourier_series}
        coord_transform = _CoordTransform(**kwargs)

        return coord_transform



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self.execute_post_core_attrs_update_actions()

        return None



    def map_to_fractional_cartesian_coords_of_distorted_image(
            self, u_x=_default_u_x, u_y=_default_u_y):
        r"""Construct the discretized :math:`k`-space fractional intensity 
        corresponding to a given discretized :math:`k`-space wavefunction of 
        some coherent probe.

        See the documentation for the class
        :class:`embeam.stem.probe.discretized.kspace.Wavefunction` for a
        discussion on discretized :math:`k`-space wavefunctions of coherent
        probes.

        Parameters
        ----------
        discretized_wavefunction : :class:`embeam.stem.probe.discretized.kspace.Wavefunction`
            The discretized :math:`k`-space wavefunction of the coherent probe
            of interest, from which to construct the discretized :math:`k`-space
            fractional intensity.

        Returns
        -------
        discretized_intensity : :class:`embeam.stem.probe.discretized.kspace.Intensity`
            The discretized :math:`k`-space fractional intensity corresponding
            to the given discretized :math:`k`-space wavefunction of the
            coherent probe of interest.

        """
        params = {"u_x_and_u_y": (u_x, u_y), "device": self._device}
        u_x, u_y = _check_and_convert_u_x_and_u_y(params)

        method_alias = \
            self._map_to_fractional_cartesian_coords_of_distorted_image
        q_x, q_y = \
            method_alias(u_x, u_y)

        return q_x, q_y



    def _map_to_fractional_cartesian_coords_of_distorted_image(self, u_x, u_y):
        if (self._cached_q_x is None) or (self._cached_q_y is None):
            self._cached_q_x, self._cached_q_y = self._calc_cached_q_x_and_q_y()

        grid_shape = (1,) + u_x.shape + (2,)
        grid = torch.zeros(grid_shape, dtype=u_x.dtype, device=u_x.device)
        grid[0, :, :, 0] = 2*(u_x-0.5)
        grid[0, :, :, 1] = -2*(u_y-0.5)

        kwargs = {"input": self._cached_q_x,
                  "grid": grid,
                  "mode": "bilinear",
                  "padding_mode": "zeros",
                  "align_corners": False}
        q_x = torch.nn.functional.grid_sample(**kwargs)[0, 0, :, :]

        kwargs["input"] = self._cached_q_y
        q_y = torch.nn.functional.grid_sample(**kwargs)[0, 0, :, :]

        return q_x, q_y



    def _calc_cached_q_x_and_q_y(self):
        u_x, u_y = self._generate_coord_meshgrid()

        with torch.no_grad():
            distortion_model_is_trivial = self._is_trivial
            
            if distortion_model_is_trivial:
                cached_q_x = u_x
                cached_q_y = u_y
            else:
                obj_alias = self._coord_transform_right_inverse

                coord_transform_inputs = dict()
                kwargs = {"coord_transform_inputs": coord_transform_inputs,
                          "p": (u_x, u_y)}
                obj_alias.update_coord_transform_inputs(**kwargs)

                kwargs = {"coord_transform_inputs": coord_transform_inputs,
                          "coord_transform": obj_alias.coord_transform_1}
                q_hat = obj_alias.eval_q_hat(**kwargs)

                cached_q_x, cached_q_y = q_hat

        for _ in range(2):
            cached_q_x = torch.unsqueeze(cached_q_x, dim=0)
            cached_q_y = torch.unsqueeze(cached_q_y, dim=0)
        
        return cached_q_x, cached_q_y



    def _generate_coord_meshgrid(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        
        kwargs = {"sampling_grid_dims_in_pixels": \
                  self_core_attrs["sampling_grid_dims_in_pixels"],
                  "device": \
                  self._device}
        coord_meshgrid = _generate_coord_meshgrid(**kwargs)

        return coord_meshgrid



    def map_to_fractional_cartesian_coords_of_undistorted_image(
            self, q_x=_default_q_x, q_y=_default_q_y):
        params = {"q_x_and_q_y": (q_x, q_y), "device": self._device}
        q_x, q_y = _check_and_convert_q_x_and_q_y(params)

        method_alias = \
            self._map_to_fractional_cartesian_coords_of_undistorted_image
        u_x, u_y, convergence_map, mask_frame = \
            method_alias(q_x, q_y)

        return u_x, u_y, convergence_map, mask_frame



    def _map_to_fractional_cartesian_coords_of_undistorted_image(self,
                                                                 q_x,
                                                                 q_y):
        if (self._cached_u_x is None) or (self._cached_u_y is None):
            cached_objs = self._calc_cached_u_x_u_y_q_x_q_y_and_abs_det_J()
            self._cached_u_x = cached_objs[0]
            self._cached_u_y = cached_objs[1]
            self._cached_q_x = cached_objs[2]
            self._cached_q_y = cached_objs[3]
            self._cached_abs_det_J = cached_objs[4]
            self._cached_convergence_map = self._calc_cached_convergence_map()

        grid_shape = (1,) + q_x.shape + (2,)
        grid = torch.zeros(grid_shape, dtype=q_x.dtype, device=q_x.device)
        grid[0, :, :, 0] = 2*(q_x-0.5)
        grid[0, :, :, 1] = -2*(q_y-0.5)

        kwargs = {"input": self._cached_u_x,
                  "grid": grid,
                  "mode": "bilinear",
                  "padding_mode": "zeros",
                  "align_corners": False}
        u_x = torch.nn.functional.grid_sample(**kwargs)[0, 0, :, :]

        kwargs["input"] = self._cached_u_y
        u_y = torch.nn.functional.grid_sample(**kwargs)[0, 0, :, :]

        kwargs["input"] = self._cached_convergence_map
        convergence_map = torch.nn.functional.grid_sample(**kwargs)[0, 0, :, :]
        convergence_map = (convergence_map >= 1)

        kwargs = {"mat": convergence_map}
        mask_frame = _calc_minimum_frame_to_mask_all_zero_valued_elems(**kwargs)

        return u_x, u_y, convergence_map, mask_frame



    def _calc_cached_u_x_u_y_q_x_q_y_and_abs_det_J(self):
        q_x, q_y = self._generate_coord_meshgrid()
        
        with torch.no_grad():
            distortion_model_is_trivial = self._is_trivial
            
            if distortion_model_is_trivial:
                cached_u_x = q_x
                cached_u_y = q_y
                cached_q_x = q_x
                cached_q_y = q_y
                cached_abs_det_J = torch.ones_like(q_x)
            else:
                inputs = \
                    {"q_x": q_x, "q_y": q_y}
                obj_alias = \
                    self._coord_transform_right_inverse
                method_alias = \
                    obj_alias.initialize_levenberg_marquardt_alg_variables
                _ = \
                    method_alias(inputs)

                cached_q_x = obj_alias.q_hat_1[0] + 0
                cached_q_y = obj_alias.q_hat_1[1] + 0

                J = obj_alias.J
                cached_abs_det_J = torch.abs(J[0, 0]*J[1, 1] - J[1, 0]*J[0, 1])

                inputs = dict()
                cached_u_x, cached_u_y = obj_alias.eval_forward_output(inputs)

        for _ in range(2):
            cached_u_x = torch.unsqueeze(cached_u_x, dim=0)
            cached_u_y = torch.unsqueeze(cached_u_y, dim=0)
            cached_q_x = torch.unsqueeze(cached_q_x, dim=0)
            cached_q_y = torch.unsqueeze(cached_q_y, dim=0)

        return cached_u_x, cached_u_y, cached_q_x, cached_q_y, cached_abs_det_J



    def _calc_cached_convergence_map(self):
        distortion_model_is_trivial = self._is_trivial

        if distortion_model_is_trivial:
            cached_convergence_map = \
                1.0*torch.ones_like(self._cached_u_x[0, 0], dtype=bool)
        else:
            coord_transform_right_inverse = \
                self._coord_transform_right_inverse
            cached_convergence_map = \
                1.0*coord_transform_right_inverse.convergence_map

        for _ in range(2):
            cached_convergence_map = torch.unsqueeze(cached_convergence_map,
                                                     dim=0)

        return cached_convergence_map



    def distort_images(self, undistorted_images=_default_undistorted_images):
        params = {"images": undistorted_images,
                  "name_of_alias_of_images": "undistorted_images",
                  "device": self._device}
        undistorted_images = _check_and_convert_images(params)

        distorted_images, convergence_map, mask_frame = \
            self._distort_images(undistorted_images)

        return distorted_images, convergence_map, mask_frame



    def _distort_images(self, undistorted_images):
        q_x, q_y = self._generate_coord_meshgrid()

        method_alias = \
            self._map_to_fractional_cartesian_coords_of_undistorted_image
        u_x, u_y, convergence_map, mask_frame = \
            method_alias(q_x, q_y)

        num_images = undistorted_images.shape[0]

        grid_shape = (num_images,) + u_x.shape + (2,)
        grid = torch.zeros(grid_shape, dtype=u_x.dtype, device=u_x.device)
        for image_idx in range(num_images):
            grid[image_idx, :, :, 0] = 2*(u_x-0.5)
            grid[image_idx, :, :, 1] = -2*(u_y-0.5)

        kwargs = {"input": undistorted_images,
                  "grid": grid,
                  "mode": "bilinear",
                  "padding_mode": "zeros",
                  "align_corners": False}
        distorted_images = (torch.nn.functional.grid_sample(**kwargs)
                            * self._cached_abs_det_tilde_J)

        return distorted_images, convergence_map, mask_frame



    def undistort_images(self, distorted_images=_default_distorted_images):
        params = {"images": distorted_images,
                  "name_of_alias_of_images": "distorted_images",
                  "device": self._device}
        distorted_images = _check_and_convert_images(params)

        undistorted_images = self._undistort_images(distorted_images)

        return undistorted_images



    def _undistort_images(self, distorted_images):
        u_x, u_y = self._generate_coord_meshgrid()

        if self._cached_abs_det_J is None:
            cached_objs = self._calc_cached_q_x_q_y_and_abs_det_J(u_x, u_y)
            self._cached_q_x = cached_objs[0]
            self._cached_q_y = cached_objs[1]
            self._cached_abs_det_J = cached_objs[2]

        q_x = self._cached_q_x[0, 0]
        q_y = self._cached_q_y[0, 0]

        num_images = distorted_images.shape[0]

        grid_shape = (num_images,) + q_x.shape + (2,)
        grid = torch.zeros(grid_shape, dtype=q_x.dtype, device=q_x.device)
        for image_idx in range(num_images):
            grid[image_idx, :, :, 0] = 2*(q_x-0.5)
            grid[image_idx, :, :, 1] = -2*(q_y-0.5)

        kwargs = {"input": distorted_images,
                  "grid": grid,
                  "mode": "bilinear",
                  "padding_mode": "zeros",
                  "align_corners": False}
        undistorted_images = (torch.nn.functional.grid_sample(**kwargs)
                              * self._cached_abs_det_J[None, None, :, :])

        return undistorted_images



    def _calc_cached_q_x_q_y_and_abs_det_J(self, u_x, u_y):
        with torch.no_grad():
            distortion_model_is_trivial = self._is_trivial
            
            if distortion_model_is_trivial:
                if (self._cached_q_x is None) or (self._cached_q_y is None):
                    cached_q_x = u_x
                    cached_q_y = u_y
                else:
                    cached_q_x = self._cached_q_x
                    cached_q_y = self._cached_q_y
                    
                cached_abs_det_J = torch.ones_like(u_x)
            else:
                obj_alias = self._coord_transform_right_inverse

                coord_transform_inputs = dict()
                kwargs = {"coord_transform_inputs": coord_transform_inputs,
                          "p": (u_x, u_y)}
                obj_alias.update_coord_transform_inputs(**kwargs)

                kwargs = {"coord_transform_inputs": coord_transform_inputs,
                          "coord_transform": obj_alias.coord_transform_1}
                q_hat = obj_alias.eval_q_hat(**kwargs)

                if (self._cached_q_x is None) or (self._cached_q_y is None):
                    cached_q_x, cached_q_y = q_hat
                else:
                    cached_q_x = self._cached_q_x
                    cached_q_y = self._cached_q_y

                kwargs = {"coord_transform_inputs": coord_transform_inputs}
                J = obj_alias.eval_q_hat(**kwargs)
                cached_abs_det_J = torch.abs(J[0, 0]*J[1, 1] - J[1, 0]*J[0, 1])

        for _ in range(2):
            if (self._cached_q_x is None) or (self._cached_q_y is None):
                cached_q_x = torch.unsqueeze(cached_q_x, dim=0)
                cached_q_y = torch.unsqueeze(cached_q_y, dim=0)

        return cached_q_x, cached_q_y, cached_abs_det_J



    @property
    def is_azimuthally_symmetric(self):
        r"""`bool`: A boolean variable indicating whether the distortion model 
        is azimuthally symmetric.

        If ``is_azimuthally_symmetric`` is set to ``True``, then the distortion
        model is azimuthally symmetric. Otherwise, the distortion model is not
        azimuthally symmetric.

        Note that ``is_azimuthally_symmetric`` should be considered
        **read-only**.

        """
        return self._is_azimuthally_symmetric



    @property
    def is_trivial(self):
        r"""`bool`: A boolean variable indicating whether the distortion model 
        is trivial.

        We define a trivial distortion model to be one with a corresponding
        coordinate transformation that is equivalent to the identity
        transformation.

        If ``is_trivial`` is set to ``True``, then the distortion model is
        trivial. Otherwise, the distortion model is not trivial.

        Note that ``is_trivial`` should be considered **read-only**.

        """
        return self._is_trivial



def _check_and_convert_quadratic_radial_distortion_amplitude(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = \
        {"obj": obj, "obj_name": obj_name}
    quadratic_radial_distortion_amplitude = \
        czekitout.convert.to_float(**kwargs)

    return quadratic_radial_distortion_amplitude



def _pre_serialize_quadratic_radial_distortion_amplitude(
        quadratic_radial_distortion_amplitude):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_quadratic_radial_distortion_amplitude(serializable_rep):
    quadratic_radial_distortion_amplitude = serializable_rep

    return quadratic_radial_distortion_amplitude



def _check_and_convert_elliptical_distortion_vector(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    elliptical_distortion_vector = czekitout.convert.to_pair_of_floats(**kwargs)

    return elliptical_distortion_vector



def _pre_serialize_elliptical_distortion_vector(elliptical_distortion_vector):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_elliptical_distortion_vector(serializable_rep):
    elliptical_distortion_vector = serializable_rep

    return elliptical_distortion_vector



def _check_and_convert_spiral_distortion_amplitude(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = \
        {"obj": obj, "obj_name": obj_name}
    spiral_distortion_amplitude = \
        czekitout.convert.to_float(**kwargs)

    return spiral_distortion_amplitude



def _pre_serialize_spiral_distortion_amplitude(spiral_distortion_amplitude):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_spiral_distortion_amplitude(serializable_rep):
    spiral_distortion_amplitude = serializable_rep

    return spiral_distortion_amplitude



def _check_and_convert_parabolic_distortion_vector(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    kwargs = {"obj": obj, "obj_name": obj_name}
    parabolic_distortion_vector = czekitout.convert.to_pair_of_floats(**kwargs)

    return parabolic_distortion_vector



def _pre_serialize_parabolic_distortion_vector(parabolic_distortion_vector):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_parabolic_distortion_vector(serializable_rep):
    parabolic_distortion_vector = serializable_rep

    return parabolic_distortion_vector



_default_quadratic_radial_distortion_amplitude = 0
_default_elliptical_distortion_vector = (0, 0)
_default_spiral_distortion_amplitude = 0
_default_parabolic_distortion_vector = (0, 0)



_cls_alias = fancytypes.PreSerializableAndUpdatable
class StandardCoordTransformParams(_cls_alias):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)), optional
        Insert description here.
    quadratic_radial_distortion_amplitude : `float`, optional
        Insert description here.
    elliptical_distortion_vector : `array_like` (`float`, shape=(2,)), optional
        Insert description here.
    spiral_distortion_amplitude : `float`, optional
        Insert description here.
    parabolic_distortion_vector : `array_like` (`float`, shape=(2,)), optional
        Insert description here.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "quadratic_radial_distortion_amplitude",
                        "elliptical_distortion_vector",
                        "spiral_distortion_amplitude",
                        "parabolic_distortion_vector")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}
    
    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 center=\
                 _default_center,
                 quadratic_radial_distortion_amplitude=\
                 _default_quadratic_radial_distortion_amplitude,
                 elliptical_distortion_vector=\
                 _default_elliptical_distortion_vector,
                 spiral_distortion_amplitude=\
                 _default_spiral_distortion_amplitude,
                 parabolic_distortion_vector=\
                 _default_parabolic_distortion_vector,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def execute_post_core_attrs_update_actions(self):
        r"""Execute the sequence of actions that follows immediately after 
        updating the core attributes.

        """
        self_core_attrs = self.get_core_attrs(deep_copy=False)

        elliptical_distortion_vector = \
            self_core_attrs["elliptical_distortion_vector"]
        spiral_distortion_amplitude = \
            self_core_attrs["spiral_distortion_amplitude"]
        parabolic_distortion_vector = \
            self_core_attrs["parabolic_distortion_vector"]

        if ((np.linalg.norm(elliptical_distortion_vector) == 0)
            and (np.linalg.norm(parabolic_distortion_vector) == 0)
            and np.abs(spiral_distortion_amplitude) == 0):
            self._is_azimuthally_symmetric = True
        else:
            self._is_azimuthally_symmetric = False

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def is_azimuthally_symmetric(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is azimuthally symmetric.

        If ``is_azimuthally_symmetric`` is set to ``True``, then the distortion
        model corresponding to the coordinate transformation parameters is
        azimuthally symmetric. Otherwise, the distortion model is not
        azimuthally symmetric.

        Note that ``is_azimuthally_symmetric`` should be considered
        **read-only**.

        """
        return self._is_azimuthally_symmetric



def _check_and_convert_standard_coord_transform_params(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    accepted_types = (StandardCoordTransformParams, type(None))

    if isinstance(obj, accepted_types[1]):
        standard_coord_transform_params = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        standard_coord_transform_params = copy.deepcopy(obj)

    return standard_coord_transform_params



def _pre_serialize_standard_coord_transform_params(
        standard_coord_transform_params):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_standard_coord_transform_params(serializable_rep):
    standard_coord_transform_params = \
        StandardCoordTransformParams.de_pre_serialize(serializable_rep)

    return standard_coord_transform_params



_default_standard_coord_transform_params = None



def generate_standard_distortion_model(standard_coord_transform_params=\
                                       _default_standard_coord_transform_params,
                                       sampling_grid_dims_in_pixels=\
                                       _default_sampling_grid_dims_in_pixels,
                                       device_name=\
                                       _default_device_name,
                                       least_squares_alg_params=\
                                       _default_least_squares_alg_params):
    params = locals()
    for param_name in params:
        func_name = "_check_and_convert_" + param_name
        func_alias = globals()[func_name]
        params[param_name] = func_alias(params)

    func_name = "_" + inspect.stack()[0][3]
    func_alias = globals()[func_name]
    kwargs = params
    distortion_model = func_alias(**kwargs)

    return distortion_model



def _generate_standard_distortion_model(standard_coord_transform_params,
                                        sampling_grid_dims_in_pixels,
                                        device_name,
                                        least_squares_alg_params):
    standard_coord_transform_params_core_attrs = \
        standard_coord_transform_params.get_core_attrs(deep_copy=False)
    center = \
        standard_coord_transform_params_core_attrs["center"]

    kwargs = \
        {"standard_coord_transform_params": standard_coord_transform_params}
    radial_cosine_coefficient_matrix = \
        _generate_standard_radial_cosine_coefficient_matrix(**kwargs)
    radial_sine_coefficient_matrix = \
        _generate_standard_radial_sine_coefficient_matrix(**kwargs)
    tangential_cosine_coefficient_matrix = \
        _generate_standard_tangential_cosine_coefficient_matrix(**kwargs)
    tangential_sine_coefficient_matrix = \
        _generate_standard_tangential_sine_coefficient_matrix(**kwargs)

    kwargs = {"center": \
              center,
              "radial_cosine_coefficient_matrix": \
              radial_cosine_coefficient_matrix,
              "radial_sine_coefficient_matrix": \
              radial_sine_coefficient_matrix,
              "tangential_cosine_coefficient_matrix": \
              tangential_cosine_coefficient_matrix,
              "tangential_sine_coefficient_matrix": \
              tangential_sine_coefficient_matrix}
    coord_transform_params = CoordTransformParams(**kwargs)

    kwargs = {"coord_transform_params": coord_transform_params,
              "sampling_grid_dims_in_pixels": sampling_grid_dims_in_pixels,
              "device_name": device_name,
              "least_squares_alg_params": least_squares_alg_params}
    distortion_model = DistortionModel(**kwargs)

    return distortion_model



def _generate_standard_radial_cosine_coefficient_matrix(
        standard_coord_transform_params):
    standard_coord_transform_params_core_attrs = \
        standard_coord_transform_params.get_core_attrs(deep_copy=False)

    attr_name = \
        "quadratic_radial_distortion_amplitude"
    quadratic_radial_distortion_amplitude = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "elliptical_distortion_vector"
    elliptical_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "parabolic_distortion_vector"
    parabolic_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    A_r_0_3 = quadratic_radial_distortion_amplitude
    A_r_2_1 = elliptical_distortion_vector[0]
    A_r_1_2 = parabolic_distortion_vector[0]

    radial_cosine_coefficient_matrix = ((0, 0, A_r_0_3), 
                                        (0, A_r_1_2, 0), 
                                        (A_r_2_1, 0, 0))

    return radial_cosine_coefficient_matrix



def _generate_standard_radial_sine_coefficient_matrix(
        standard_coord_transform_params):
    standard_coord_transform_params_core_attrs = \
        standard_coord_transform_params.get_core_attrs(deep_copy=False)

    attr_name = \
        "elliptical_distortion_vector"
    elliptical_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "parabolic_distortion_vector"
    parabolic_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    B_r_2_1 = elliptical_distortion_vector[1]
    B_r_1_2 = parabolic_distortion_vector[1]

    radial_sine_coefficient_matrix = ((0, B_r_1_2), 
                                      (B_r_2_1, 0))

    return radial_sine_coefficient_matrix



def _generate_standard_tangential_cosine_coefficient_matrix(
        standard_coord_transform_params):
    standard_coord_transform_params_core_attrs = \
        standard_coord_transform_params.get_core_attrs(deep_copy=False)
    
    attr_name = \
        "spiral_distortion_amplitude"
    spiral_distortion_amplitude = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "elliptical_distortion_vector"
    elliptical_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "parabolic_distortion_vector"
    parabolic_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    A_t_0_3 = spiral_distortion_amplitude
    A_t_2_1 = elliptical_distortion_vector[1]
    A_t_1_2 = parabolic_distortion_vector[1]/3

    tangential_cosine_coefficient_matrix = ((0, 0, A_t_0_3), 
                                            (0, A_t_1_2, 0), 
                                            (A_t_2_1, 0, 0))

    return tangential_cosine_coefficient_matrix



def _generate_standard_tangential_sine_coefficient_matrix(
        standard_coord_transform_params):
    standard_coord_transform_params_core_attrs = \
        standard_coord_transform_params.get_core_attrs(deep_copy=False)
    
    attr_name = \
        "elliptical_distortion_vector"
    elliptical_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    attr_name = \
        "parabolic_distortion_vector"
    parabolic_distortion_vector = \
        standard_coord_transform_params_core_attrs[attr_name]

    B_t_2_1 = -elliptical_distortion_vector[0]
    B_t_1_2 = -parabolic_distortion_vector[0]/3

    tangential_sine_coefficient_matrix = ((0, B_t_1_2), 
                                          (B_t_2_1, 0))

    return tangential_sine_coefficient_matrix



###########################
## Define error messages ##
###########################

_check_and_convert_coefficient_matrix_err_msg_1 = \
    ("The object ``{}`` must be a 2D array of real numbers or "
     "of the type `NoneType`.")

_coord_transform_right_inverse_err_msg_1 = \
    ("Failed to calculate iteratively the right-inverse to the specified "
     "coordinate transformation in ``max_num_iterations`` steps or less, where "
     "the object ``max_num_iterations`` is the maximum number of iteration "
     "steps allowed, which in this case was set to {}.")

_check_and_convert_u_x_and_u_y_err_msg_1 = \
    ("The object ``{}`` must have the same shape as the object ``{}``.")

_check_and_convert_real_torch_matrix_err_msg_1 = \
    ("The object ``{}`` must be a real-valued matrix.")

_check_and_convert_q_x_and_q_y_err_msg_1 = \
    _check_and_convert_u_x_and_u_y_err_msg_1

_check_and_convert_images_err_msg_1 = \
    ("The object ``{}`` must be a real-valued 2D, 3D, or 4D array.")

_check_and_convert_device_name_err_msg_1 = \
    ("The object ``device_name`` must be either of the type `NoneType` or "
     "`str`, wherein the latter case, ``device_name`` must be a valid device "
     "name.")
