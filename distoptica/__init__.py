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
        coefficient_matrix = tuple(tuple(row.tolist())
                                   for row
                                   in coefficient_matrix)

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
    comprising of two components: :math:`T_{⌑;x}\left(u_{x},u_{y}\right)` and
    :math:`T_{⌑;y}\left(u_{x},u_{y}\right)`. Following Ref. [Brázda1]_, we
    assume that these two functions are of the following respective mathematical
    forms:

    .. math ::
        T_{⌑;x}\left(u_{x},u_{y}\right) & =
        \left\{ u_{r}\cos\left(u_{\theta}\right)+x_{c;D}\right\} \nonumber \\
        & \quad+\left\{ T_{⌑;r}\left(u_{r},u_{\theta}\right)
        \cos\left(u_{\theta}\right)
        -T_{⌑;t}\left(u_{r},u_{\theta}\right)
        \sin\left(u_{\theta}\right)\right\} ,
        :label: T_distsq_x__1

    .. math ::
        T_{⌑;y}\left(u_{x},u_{y}\right) & =
        \left\{ u_{r}\sin\left(u_{\theta}\right)+y_{c;D}\right\} \nonumber \\
        & \quad+\left\{ T_{⌑;r}\left(u_{r},u_{\theta}\right)
        \sin\left(u_{\theta}\right)
        +T_{⌑;t}\left(u_{r},u_{\theta}\right)
        \cos\left(u_{\theta}\right)\right\} ,
        :label: T_distsq_y__1

    where

    .. math ::
        \left(x_{c;D},y_{c;D}\right) \in\mathbb{R}^{2},
        :label: center_of_distortion__1

    .. math ::
        u_{r} =\sqrt{\left(u_{x}-x_{c;D}\right)^{2}+
        \left(u_{y}-y_{c;D}\right)^{2}},
        :label: u_r__1

    .. math ::
        u_{\theta} =\tan^{-1}\left(\frac{y-y_{c;D}}{x-x_{c;D}}\right),
        :label: u_theta__1

    .. math ::
        T_{⌑;r}\left(u_{r},u_{\theta}\right) & =
        \sum_{v_{1}=0}^{N_{r;\cos}}\rho_{\cos;r;v_{1}}\left(u_{r}\right)
        \cos\left(v_{1}u_{\theta}\right)\nonumber \\
        & \quad+\sum_{v_{1}=0}^{N_{r;\sin}-1}
        \rho_{\sin;r;v_{1}}\left(u_{r}\right)\sin\left(\left\{ v_{1}+1\right\} 
        u_{\theta}\right),
        :label: T_distsq_r__1

    .. math ::
        T_{⌑;t}\left(u_{r},u_{\theta}\right) & =
        \sum_{v_{1}=0}^{N_{t;\cos}}\rho_{\cos;t;v_{1}}\left(u_{r}\right)
        \cos\left(v_{1}u_{\theta}\right)\nonumber \\
        & \quad+\sum_{v_{1}=0}^{N_{t;\sin}-1}
        \rho_{\sin;t;v_{1}}\left(u_{r}\right)\sin\left(\left\{ v_{1}+1\right\} 
        u_{\theta}\right),
        :label: T_distsq_t__1

    with
    
    .. math ::
        \rho_{r;\cos;v_{1}}\left(u_{r}\right) =\sum_{v_{2}=0}^{M_{r;\cos}-1}
        A_{r;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_r_cos_v_1__1

    .. math ::
        \rho_{r;\sin;v_{1}}\left(u_{r}\right) =\sum_{v_{2}=0}^{M_{r;\sin}-1}
        B_{r;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_r_sin_v_1__1

    .. math ::
        \rho_{t;\cos;v_{1}}\left(u_{r}\right) =\sum_{v_{2}=0}^{M_{r;\cos}-1}
        A_{t;v_{1},v_{2}}u_{r}^{v_{2}+1},
        :label: rho_t_cos_v_1__1

    .. math ::
        \rho_{t;\sin;v_{1}}\left(u_{r}\right) =\sum_{v_{2}=0}^{M_{r;\sin}-1}
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



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def is_corresponding_model_azimuthally_symmetric(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is azimuthally symmetric.

        If ``is_corresponding_model_azimuthally_symmetric`` is set to ``True``,
        then the distortion model corresponding to the coordinate transformation
        parameters is azimuthally symmetric. Otherwise, the distortion model is
        not azimuthally symmetric.

        Note that ``is_corresponding_model_azimuthally_symmetric`` should be
        considered **read-only**.

        """
        result = self._is_corresponding_model_azimuthally_symmetric
        
        return result



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
        result = self._is_corresponding_model_trivial
    
        return result



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
        result = self._is_corresponding_model_standard
        
        return result



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
    r"""The parameters of the least-squares algorithm to be used to calculate 
    the mapping of fractional Cartesian coordinates of distorted images to those
    of the corresponding undistorted images.

    Users are encouraged to read the summary documentation of the class
    :class:`distoptica.DistortionModel` before reading the documentation for the
    current class as it provides essential context to what is discussed below.

    As discussed in the summary documentation of the class
    :class:`distoptica.DistortionModel`, optical distortions introduced in an
    imaging experiment can be described by a coordinate transformation,
    comprising of two components: :math:`T_{⌑;x}\left(u_{x},u_{y}\right)` and
    :math:`T_{⌑;y}\left(u_{x},u_{y}\right)`. Moreover, we assume that there
    exists a right inverse to the coordinate transformation, i.e. that there are
    functions :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` satisfying
    Eqs. :eq:`defining_T_sq_x__1` and :eq:`defining_T_sq_y__1`.

    In order to undistort images, one needs to calculate
    :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` for multiple values of both
    :math:`q_{x}` and :math:`q_{y}`, where :math:`q_{x}` and :math:`q_{y}` are
    fractional horizontal and vertical coordinates respectively of a distorted
    image. Let :math:`q_{x;w_{1},w_{2}}` and :math:`q_{y;w_{1},w_{2}}` be
    matrices of the same dimensions storing the values of :math:`q_{x}` and
    :math:`q_{y}`, respectively, for which to calculate
    :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)`, where :math:`w_{1}` and
    :math:`w_{2}` are row and column indices respectively. In :mod:`distoptica`,
    :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` are calculated using the
    Levenberg-Marquardt (LM) algorithm. Specifically, we apply the LM algorithm
    to try to find a solution iteratively to the problem:

    .. math ::
        0=q_{x;w_{1},w_{2}}-
        T_{⌑;x}\left(u_{x}=u_{x;w_{1},w_{2}},u_{y}=u_{y;w_{1},w_{2}}\right),
        :label: LM_problem__1

    .. math ::
        0=q_{y;w_{1},w_{2}}-
        T_{⌑;y}\left(u_{x}=u_{x;w_{1},w_{2}},u_{y}=u_{y;w_{1},w_{2}}\right),
        :label: LM_problem__2

    where :math:`q_{x;w_{1},w_{2}}`, :math:`q_{y;w_{1},w_{2}}`,
    :math:`T_{⌑;x}\left(u_{x},u_{y}\right)`, and
    :math:`T_{⌑;y}\left(u_{x},u_{y}\right)` are given, and
    :math:`u_{x;w_{1},w_{2}}` and :math:`u_{y;w_{1},w_{2}}` are unknowns to be
    determined, if possible.

    Before describing the LM algorithm, we introduce a few quantities and
    notation. First, let :math:`\lambda_{0}` be a given positive real number,
    which we refer to as the initial damping of the LM algorithm. Next, let
    :math:`\lambda_{\uparrow}` and :math:`\lambda_{\downarrow}` be given
    positive real numbers, which we refer to as the factors for increasing and
    decreasing the damping respectively. Next, let :math:`\epsilon_{\rho}`,
    :math:`\epsilon_{\chi}`, and :math:`\epsilon_{-}` be given positive real
    numbers, which we refer to as the improvement tolerance, the error
    tolerance, and the plateau tolerance. Next, let :math:`N_{-;\max}` and
    :math:`N_{\nu}` be given positive integers, which we refer to as the plateau
    patience and max number of iterations respectively. Next, let
    :math:`N_{w;x}` and :math:`N_{w;y}` be the number of columns and rows of the
    matrix :math:`q_{x;w_{1},w_{2}}`. Next, let :math:`\nu` be the iteration
    index. Next, we introduce the following boldface notation:

    .. math ::
        \mathbf{O}_{\nu;w_{1},w_{2}}=
        \begin{pmatrix}O_{\nu;w_{1},w_{2};0,0} & O_{\nu;w_{1},w_{2};0,1}\\
        O_{\nu;w_{1},w_{2};1,0} & O_{\nu;w_{1},w_{2};1,1}
        \end{pmatrix},
        :label: boldface_matrix_notation__1

    .. math ::
        \mathbf{o}_{\nu;w_{1},w_{2}}=\begin{pmatrix}o_{\nu;w_{1},w_{2};0}\\
        o_{\nu;w_{1},w_{2};1}
        \end{pmatrix},
        :label: boldface_matrix_notation__2

    where the letters :math:`\mathbf{O}` and :math:`\mathbf{o}` are placeholders for any
    boldface uppercase and lowercase symbols respectively.

    With the quantities and notation that we have introduced in the previous
    paragraph, we describe the LM algorithm below:

    1. :math:`\nu\leftarrow 0`.

    2. :math:`N_{-}\leftarrow 0`.

    3. :math:`\Delta_{\text{best}}\leftarrow\infty`.

    4. :math:`\lambda_{\nu;w_{1},w_{2}}\leftarrow\lambda_{0}`.

    5. :math:`\mathbf{p}_{\nu;w_{1},w_{2}}\leftarrow\begin{pmatrix}q_{x;w_{1},w_{2}}\\
    q_{y;w_{1},w_{2}} \end{pmatrix}`.

    6. :math:`\hat{\mathbf{q}}_{\nu;w_{1},w_{2}}\leftarrow
    \begin{pmatrix}T_{⌑;x}\left(u_{x}=p_{\nu;w_{1},w_{2};0},
    u_{y}=p_{\nu;w_{1},w_{2};1}\right)\\
    T_{⌑;y}\left(u_{x}=p_{\nu;w_{1},w_{2};0},u_{y}=p_{\nu;w_{1},w_{2};1}\right)
    \end{pmatrix}`.

    7. :math:`\boldsymbol{\chi}_{\nu;w_{1},w_{2}}\leftarrow
    \begin{pmatrix}q_{x;w_{1},w_{2}}-\hat{q}_{\nu;w_{1},w_{2};0}\\
    q_{y;w_{1},w_{2}}-\hat{q}_{\nu;w_{1},w_{2};1}
    \end{pmatrix}`.

    8. :math:`\mathbf{J}_{\nu;w_{1},w_{2}}\leftarrow
    \mathbf{J}_{⌑}\left(u_{x}=p_{\nu;w_{1},w_{2};0},
    u_{y}=p_{\nu;w_{1},w_{2};1}\right)`.

    9. :math:`\mathbf{H}_{\nu;w_{1},w_{2}}\leftarrow
    \mathbf{J}_{\nu;w_{1},w_{2}}\mathbf{J}_{\nu;w_{1},w_{2}}`.

    10. :math:`\mathbf{D}_{\nu;w_{1},w_{2}}\leftarrow
    \lambda_{\nu;w_{1},w_{2}}\begin{pmatrix}H_{\nu;w_{1},w_{2};0,0} & 0\\
    0 & H_{\nu;w_{1},w_{2};1,1}
    \end{pmatrix}`.

    11. :math:`\mathbf{A}_{\nu;w_{1},w_{2}}\leftarrow
    \mathbf{H}_{\nu;w_{1},w_{2}}+\mathbf{D}_{\nu;w_{1},w_{2}}`.

    12. :math:`\mathbf{g}_{\nu;w_{1},w_{2}}\leftarrow
    \mathbf{J}_{\nu;w_{1},w_{2}}\boldsymbol{\chi}_{\nu;w_{1},w_{2}}`.

    13. Solve :math:`\mathbf{A}_{\nu;w_{1},w_{2}}
    \mathbf{h}_{\nu;w_{1},w_{2}}=\mathbf{g}_{\nu;w_{1},w_{2}}`, for 
    :math:`\mathbf{h}_{\nu;w_{1},w_{2}}` via singular value decomposition.

    14. :math:`\mathbf{p}_{\nu+1;w_{1},w_{2}}\leftarrow
    \mathbf{p}_{\nu;w_{1},w_{2}}+\mathbf{h}_{\nu;w_{1},w_{2}}`.

    15. :math:`\hat{\mathbf{q}}_{\nu+1;w_{1},w_{2}}\leftarrow
    \begin{pmatrix}T_{⌑;x}\left(u_{x}=p_{\nu+1;w_{1},w_{2};0},
    u_{y}=p_{\nu+1;w_{1},w_{2};1}\right)\\
    T_{⌑;y}\left(u_{x}=p_{\nu+1;w_{1},w_{2};0},
    u_{y}=p_{\nu+1;w_{1},w_{2};1}\right)
    \end{pmatrix}`.

    16. :math:`\boldsymbol{\chi}_{\nu+1;w_{1},w_{2}}\leftarrow
    \begin{pmatrix}q_{x;w_{1},w_{2}}-\hat{q}_{\nu+1;w_{1},w_{2};0}\\
    q_{y;w_{1},w_{2}}-\hat{q}_{\nu+1;w_{1},w_{2};1}
    \end{pmatrix}`.

    17. :math:`\delta_{\nu+1;w_{1},w_{2}}\leftarrow
    \max\left(N_{w;x},N_{w;y}\right)
    \left|\boldsymbol{\chi}_{\nu+1;w_{1},w_{2}}\right|`.

    18. :math:`\Delta_{\nu+1}\leftarrow
    \sum_{w_{1},w_{2}}\delta_{\nu+1;w_{1},w_{2}}`.

    19. :math:`\boldsymbol{\rho}_{\nu+1;w_{1},w_{2}}\leftarrow
    \frac{\left\Vert \boldsymbol{\chi}_{\nu;w_{1},w_{2}}\right\Vert ^{2}
    -\left\Vert \boldsymbol{\chi}_{\nu+1;w_{1},
    w_{2}}\right\Vert ^{2}}{\left|\mathbf{h}_{\nu;w_{1},
    w_{2}}^{\text{T}}\left(\mathbf{D}_{\nu;w_{1},
    w_{2}}\mathbf{h}_{\nu;w_{1},w_{2}}+\mathbf{g}_{\nu;w_{1},
    w_{2}}\right)\right|}`.

    20. :math:`\lambda_{\nu+1;w_{1},w_{2}}\leftarrow\begin{cases}
    \max\left(\frac{1}{\lambda_{\downarrow}}\lambda_{\nu+1;w_{1},w_{2}},
    10^{-7}\right),
    & \text{if }\boldsymbol{\rho}_{\nu+1;w_{1},w_{2}}>\epsilon_{\rho},\\
    \min\left(\lambda_{\uparrow}\lambda_{\nu+1;w_{1},w_{2}},10^{7}\right), &
    \text{otherwise}.  \end{cases}`

    21. :math:`N_{-}\leftarrow\begin{cases} 0, & \text{if
    }\Delta_{\nu+1}<\left(1-\epsilon_{-}\right)\Delta_{\text{best}},\\ N_{-}+1,
    & \text{otherwise}.  \end{cases}`

    22. If :math:`N_{-}\ge N_{-;\max}`, then go to step 23. Otherwise, go to
    step 24.

    23. If :math:`\delta_{\nu+1;w_{1},w_{2}}<\epsilon_{\chi}`, for all
    :math:`w_{1}` and :math:`w_{2}` in which
    :math:`\mathbf{p}_{\nu+1;w_{1},w_{2}}\in[0,1]\times\left[0,1\right]`, then
    go to step 28. Otherwise, go to step 25.

    24. If :math:`\delta_{\nu+1;w_{1},w_{2}}<\epsilon_{\chi}`, for all
    :math:`w_{1}` and :math:`w_{2}`, then go to step 28. Otherwise, go to step
    25.

    25. If :math:`\nu=N_{\nu}-1`, then go to step 31. Otherwise, go to step 26.

    26. :math:`\nu\leftarrow \nu+1`.

    27. Go to step 8.

    28. :math:`u_{x;w_{1},w_{2}}\leftarrow p_{\nu+1;w_{1},w_{2};0}`.

    29. :math:`u_{y;w_{1},w_{2}}\leftarrow p_{\nu+1;w_{1},w_{2};1}`.

    30. Stop algorithm without raising an exception.

    31. Stop algorithm with an exception raised.

    Parameters
    ----------
    max_num_iterations : `int`, optional
        The max number of iterations, :math:`N_{\nu}`, introduced in the summary
        documentation above.
    initial_damping : `float`, optional
        The initial damping, :math:`\lambda_{0}`, introduced in the summary 
        documentation above.
    factor_for_decreasing_damping : `float`, optional
        The factor for decreasing damping, :math:`\lambda_{\downarrow}`, 
        introduced in the summary documentation above.
    factor_for_increasing_damping : `float`, optional
        The factor for increase damping, :math:`\lambda_{\uparrow}`, introduced
        in the summary documentation above.
    improvement_tol : `float`, optional
        The improvement tolerance, :math:`\epsilon_{\rho}`, introduced in the 
        summary documentation above.
    rel_err_tol : `float`, optional
        The error tolerance, :math:`\epsilon_{\chi}`, introduced in the summary
        documentation above.
    plateau_tol : `float`, optional
        The plateau tolerance, :math:`\epsilon_{-}`, introduced in the summary 
        documentation above.
    plateau_patience : `int`, optional
        The plateau patience, :math:`N_{-;\max}`, introduced in the summary 
        documentation above.
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
                        "factor_for_increasing_damping",
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
        u_x, u_y = self.calc_u_x_and_u_y_via_levenberg_marquardt_alg()

        output_tensor_shape = (2,) + u_x.shape
        output_tensor = torch.zeros(output_tensor_shape,
                                    dtype=u_x.dtype,
                                    device=u_x.device)
        output_tensor[0] = u_x
        output_tensor[1] = u_y

        return output_tensor



    def calc_u_x_and_u_y_via_levenberg_marquardt_alg(self):
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



def _calc_minimum_frame_to_mask_all_zero_valued_elems(mat):
    if mat.sum().item() == 0:
        minimum_frame_to_mask_all_zero_valued_elems = (mat.shape[1],
                                                       0,
                                                       0,
                                                       mat.shape[0])
    elif mat.sum().item() != mat.numel():
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
            tuple(minimum_frame_to_mask_all_zero_valued_elems.tolist())
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
            
        images = obj.to(device=params["device"], dtype=torch.float32)

    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(name_of_alias_of_images)
        raise TypeError(err_msg)

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
            raise TypeError(err_msg)

    return device_name



def _pre_serialize_device_name(device_name):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_device_name(serializable_rep):
    device_name = serializable_rep

    return device_name



def _check_and_convert_deep_copy(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    deep_copy = czekitout.convert.to_bool(**kwargs)

    return deep_copy



_default_coord_transform_params = None
_default_sampling_grid_dims_in_pixels = (512, 512)
_default_device_name = None
_default_least_squares_alg_params = None
_default_deep_copy = True



_cls_alias = fancytypes.PreSerializableAndUpdatable
class DistortionModel(_cls_alias):
    r"""An optical distortion model.

    To begin our discussion on optical distortion models, let us consider two
    thought experiments. First, let :math:`E_{\square}` denote an imaging
    experiment of a sample wherein the imaging aparatus is operating at a fixed
    set of target parameters, and all the optical elements used in the imaging
    aparatus are idealized in the sense that they do not introduce any optical
    distortions. Secondly, let :math:`E_{⌑}` denote an imaging experiment that
    is identical to :math:`E_{\square}` except that the optical elements used in
    the imaging aparatus possess imperfections in the sense that they introduce
    a particular set of optical distortions. We will refer to the set of images
    resulting from the imaging experiment :math:`E_{\square}` as the set of
    undistorted images, and the set of images resulting from the imaging
    experiment :math:`E_{⌑}` as the set of distorted images.

    We assume that the images resulting from both imaging experiments are formed
    at a common image plane, that each experiment yields the same number of
    images :math:`N_{\mathcal{I}}`, that all images have the same number of
    channels :math:`N_{\mathcal{I};C}`, and that all images are of the same
    spatial dimensions, i.e. they have the same dimensions in pixels with the
    same pixel sizes. Let :math:`N_{\mathcal{I};x}` and
    :math:`N_{\mathcal{I};y}` be the number of pixels in either image from left
    to right and top to bottom respectively.

    For simplicity, we describe positions within images using fractional
    coordinates. First, let :math:`u_{x}` and :math:`u_{y}` be the fractional
    horizontal and vertical coordinates, respectively, of a point in an
    undistorted image, where :math:`\left(u_{x},u_{y}\right)=\left(0,0\right)`
    is the bottom left corner of the image. Secondly, let :math:`q_{x}` and
    :math:`q_{y}` be the fractional horizontal and vertical coordinates,
    respectively, of a point in a distorted image, where
    :math:`\left(q_{x},q_{y}\right)=\left(0,0\right)` is the bottom left corner
    of the image.

    The optical distortions introduced by experiment :math:`E_{⌑}` can be
    described by a coordinate transformation, which maps a given coordinate pair
    :math:`\left(u_{x},u_{y}\right)` to a corresponding coordinate pair
    :math:`\left(q_{x},q_{y}\right)`. Let
    :math:`T_{⌑;x}\left(u_{x},u_{y}\right)` be the component of the coordinate
    transformation that maps :math:`\left(u_{x},u_{y}\right)` to its
    corresponding :math:`q_{x}`, and let :math:`T_{⌑;y}\left(u_{x},u_{y}\right)`
    be the component of the coordinate transformation that maps
    :math:`\left(u_{x},u_{y}\right)` to its corresponding :math:`q_{y}`. See the
    documentation for the class :class:`distoptica.CoordTransformParams` for a
    mathematical description of :math:`T_{⌑;x}\left(u_{x},u_{y}\right)` and
    :math:`T_{⌑;y}\left(u_{x},u_{y}\right)`.

    We assume that there exists a right inverse to the coordinate
    transformation, i.e. that there are functions
    :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` satisfying:

    .. math ::
        T_{⌑;x}\left(u_{x}=T_{\square;x}\left(q_{x},q_{y}\right),u_{y}=
        T_{\square;y}\left(q_{x},q_{y}\right)\right)\equiv q_{x},
        :label: defining_T_sq_x__1

    .. math ::
        T_{⌑;y}\left(u_{x}=T_{\square;x}\left(q_{x},q_{y}\right),u_{y}=
        T_{\square;y}\left(q_{x},q_{y}\right)\right)\equiv q_{y},
        :label: defining_T_sq_y__1

    In other words, :math:`T_{\square;x}\left(q_{x},q_{y}\right)` maps
    :math:`\left(q_{x},q_{y}\right)` to its corresponding :math:`u_{x}`, and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` maps
    :math:`\left(q_{x},q_{y}\right)` to its corresponding :math:`u_{y}`, when
    :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
    T_{\square;y}\left(q_{x},q_{y}\right)\right)` is well-defined at
    :math:`\left(q_{x},q_{y}\right)`. See the documentation for the class
    :class:`distoptica.LeastSquaresAlgParams` for a discussion on how
    :math:`T_{\square;x}\left(q_{x},q_{y}\right)` and
    :math:`T_{\square;y}\left(q_{x},q_{y}\right)` are calculated.

    One of the primary purposes of the class :class:`distoptica.DistortionModel`
    is to distort undistorted images or to undistort distorted images, given a
    coordinate transformation :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
    T_{⌑;x}\left(u_{x},u_{y}\right)\right)`, and then subsequently resample the
    transformed images. To describe how :class:`distoptica.DistortionModel`
    approximates the aforementioned images transformations and resampling, it is
    worth introducing several mathematical objects. First, let
    :math:`\mathcal{I}_{\square;l,k,n,m}` be the value of the pixel centered at
    :math:`\left(u_{x},u_{y}\right)=
    \left(u_{\mathcal{I};x;m},u_{\mathcal{I};y;n}\right)` in the
    :math:`k^{\text{th}}` channel of the :math:`l^{\text{th}}` undistorted
    image, where

    .. math ::
        l\in\left\{ l^{\prime}\right\} _{l^{\prime}=0}^{N_{\mathcal{I}}-1},
        :label: l_range__1

    .. math ::
        k\in\left\{ k^{\prime}\right\} _{k^{\prime}=0}^{N_{\mathcal{I};C}-1},
        :label: k_range__1
    
    .. math ::
        m\in\left\{ m^{\prime}\right\} _{m^{\prime}=0}^{N_{\mathcal{I};x}-1},
        :label: m_range__1

    .. math ::
        n\in\left\{ n^{\prime}\right\} _{n^{\prime}=0}^{N_{\mathcal{I};y}-1},
        :label: n_range__1

    .. math ::
        u_{\mathcal{I};x;m}=\left(m+\frac{1}{2}\right)\Delta u_{\mathcal{I};x},
        :label: u_I_x_m__1

    .. math ::
        u_{\mathcal{I};y;n}=1-\left(n+\frac{1}{2}\right)
        \Delta u_{\mathcal{I};y},
        :label: u_I_y_n__1
    
    .. math ::
        \Delta u_{\mathcal{I};x}=\frac{1}{N_{\mathcal{I};x}},
        :label: Delta_u_I_x__1
    
    .. math ::
        \Delta u_{\mathcal{I};y}=\frac{1}{N_{\mathcal{I};y}},
        :label: Delta_u_I_y__1

    Next, let :math:`\mathcal{I}_{⌑;l,k,n,m}` be the value of the pixel centered
    at :math:`\left(q_{x},q_{y}\right)=
    \left(q_{\mathcal{I};x;m},q_{\mathcal{I};y;n}\right)` in the
    :math:`k^{\text{th}}` channel of the :math:`l^{\text{th}}` distorted image,
    where

    .. math ::
        q_{\mathcal{I};x;m}=\left(m+\frac{1}{2}\right)\Delta q_{\mathcal{I};x},
        :label: q_I_x_m__1

    .. math ::
        q_{\mathcal{I};y;n}=
        1-\left(n+\frac{1}{2}\right)\Delta q_{\mathcal{I};y},
        :label: q_I_y_n__1

    .. math ::
        \Delta q_{\mathcal{I};x}=\frac{1}{N_{\mathcal{I};x}},
        :label: Delta_q_I_x__1

    .. math ::
        \Delta q_{\mathcal{I};y}=\frac{1}{N_{\mathcal{I};y}}.
        :label: Delta_q_I_y__1

    Next, let :math:`\check{\mathcal{I}}_{\square;l,k}\left(u_{x},u_{y}\right)`
    be the interpolation of the :math:`k^{\text{th}}` channel of the
    :math:`l^{\text{th}}` undistorted image at
    :math:`\left(u_{x},u_{y}\right)`. Next, let
    :math:`\check{\mathcal{I}}_{⌑;l,k}\left(q_{x},q_{y}\right)` be the
    interpolation of the :math:`k^{\text{th}}` channel of the
    :math:`l^{\text{th}}` distorted image at
    :math:`\left(q_{x},q_{y}\right)`. Next, let
    :math:`\mathring{\mathcal{I}}_{\square;l,k,i,j}` be the value of the pixel
    centered at :math:`\left(u_{x},u_{y}\right)=
    \left(u_{\mathring{\mathcal{I}};x;j},u_{\mathring{\mathcal{I}};y;i}\right)`
    in the :math:`k^{\text{th}}` channel of the :math:`l^{\text{th}}` resampled
    undistorted image, where

    .. math ::
        j\in\left\{ j^{\prime}\right\}_{j^{\prime}=0}^{
        N_{\mathcal{\mathring{I}};x}-1},
        :label: j_range__1

    .. math ::
        i\in\left\{ i^{\prime}\right\}_{i^{\prime}=0}^{
        N_{\mathcal{\mathring{I}};y}-1},
        :label: i_range__1

    .. math ::
        u_{\mathcal{\mathring{I}};x;j}=\left(j+\frac{1}{2}\right)
        \Delta u_{\mathcal{\mathring{I}};x},
        :label: u_I_circ_x_j__1

    .. math ::
        u_{\mathcal{\mathring{I}};y;i}=1-\left(i+\frac{1}{2}\right)
        \Delta u_{\mathcal{\mathring{I}};y},
        :label: u_I_circ_y_i__1

    .. math ::
        \Delta u_{\mathcal{\mathring{I}};x}=
        \frac{1}{N_{\mathcal{\mathring{I}};x}},
        :label: Delta_u_I_circ_x__1

    .. math ::
        \Delta u_{\mathcal{\mathring{I}};y}=
        \frac{1}{N_{\mathcal{\mathring{I}};y}},
        :label: Delta_u_I_circ_y__1

    and :math:`N_{\mathcal{\mathring{I}};x}` and
    :math:`N_{\mathcal{\mathring{I}};y}` are the number of pixels in the
    sampling grid from left to right and top to bottom respectively. Next, let
    :math:`\mathring{\mathcal{I}}_{⌑;l,k,i,j}` be the value of the pixel
    centered at :math:`\left(q_{x},q_{y}\right)=
    \left(q_{\mathring{\mathcal{I}};x;j},q_{\mathring{\mathcal{I}};y;i}\right)`
    in the :math:`k^{\text{th}}` channel of the :math:`l^{\text{th}}` resampled
    distorted image, where

    .. math ::
        q_{\mathcal{\mathring{I}};x;j}=\left(j+\frac{1}{2}\right)
        \Delta q_{\mathcal{\mathring{I}};x},
        :label: q_I_circ_x_j__1

    .. math ::
        q_{\mathcal{\mathring{I}};y;i}=1-\left(i+\frac{1}{2}\right)
        \Delta q_{\mathcal{\mathring{I}};y},
        :label: q_I_circ_y_i__1

    .. math ::
        \Delta q_{\mathcal{\mathring{I}};x}=
        \frac{1}{N_{\mathcal{\mathring{I}};x}},
        :label: Delta_q_I_circ_x__1

    .. math ::
        \Delta q_{\mathcal{\mathring{I}};y}=
        \frac{1}{N_{\mathcal{\mathring{I}};y}}.
        :label: Delta_q_I_circ_y__1

    Next, let :math:`\mathbf{J}_{⌑}\left(u_{x},u_{y}\right)` be the Jacobian of
    :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
    T_{⌑;x}\left(u_{x},u_{y}\right)\right)`:

    .. math ::
        \mathbf{J}_{⌑}\left(u_{x},u_{y}\right)=
        \begin{pmatrix}\frac{\partial T_{⌑;x}}{\partial u_{x}} 
        & \frac{\partial T_{⌑;x}}{\partial u_{y}}\\
        \frac{\partial T_{⌑;y}}{\partial u_{x}} 
        & \frac{\partial T_{⌑;y}}{\partial u_{y}}
        \end{pmatrix}.
        :label: J_distsq__1

    Lastly, let :math:`\mathbf{J}_{\square}\left(q_{x},q_{y}\right)` be the
    Jacobian of :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
    T_{\square;x}\left(q_{x},q_{y}\right)\right)`:

    .. math ::
        \mathbf{J}_{\square}\left(q_{x},q_{y}\right)=
        \begin{pmatrix}\frac{\partial T_{\square;x}}{\partial q_{x}} 
        & \frac{\partial T_{\square;x}}{\partial q_{y}}\\
        \frac{\partial T_{\square;y}}{\partial q_{x}} 
        & \frac{\partial T_{\square;y}}{\partial q_{y}}
        \end{pmatrix}.
        :label: J_sq__1

    The same class, via the method
    :meth:`~distoptica.DistortionModel.undistort_then_resample_images`,
    approximates undistorting then resampling images by:

    .. math ::
        \mathring{\mathcal{I}}_{\square;l,k,i,j}&\approx
        \frac{N_{\mathcal{I};x}N_{\mathcal{I};y}}{N_{\mathring{\mathcal{I}};x}
        N_{\mathring{\mathcal{I}};y}}
        \left|\text{det}\left(\mathbf{J}_{⌑}\left(u_{\mathring{\mathcal{I}};
        x;j},u_{\mathring{\mathcal{I}};y;i}\right)\right)\right|\\
        &\hphantom{\approx}\quad\times\check{\mathcal{I}}_{⌑;l,k}\left(
        T_{⌑;x}\left(u_{\mathring{\mathcal{I}};x;j},
        u_{\mathring{\mathcal{I}};y;i}\right),T_{⌑;y}\left(
        u_{\mathring{\mathcal{I}};x;j},
        u_{\mathring{\mathcal{I}};y;i}\right)\right),
        :label: undistorting_then_resampling_images__1

    where :math:`\mathbf{J}_{⌑}\left(u_{\mathring{\mathcal{I}};x;j},
    u_{\mathring{\mathcal{I}};y;i}\right)` is calculated via
    Eq. :eq:`J_distsq__1`, with the derivatives being calculated analytically.

    The class :class:`distoptica.DistortionModel`, via the method
    :meth:`~distoptica.DistortionModel.distort_then_resample_images`,
    approximates distorting then resampling images by:

    .. math ::
        \mathring{\mathcal{I}}_{⌑;l,k,i,j}&\approx
        \frac{N_{\mathcal{I};x}N_{\mathcal{I};y}}{
        N_{\mathring{\mathcal{I}};x}N_{\mathring{\mathcal{I}};y}}\left|
        \text{det}\left(\mathbf{J}_{\square}\left(
        q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right)\right|\\
        &\hphantom{\approx}\quad\times\check{\mathcal{I}}_{\square;l,k}\left(
        T_{\square;x}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right),T_{\square;y}\left(
        q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right),
        :label: distorting_then_resampling_images__1

    where :math:`\mathbf{J}_{\square}\left(q_{\mathring{\mathcal{I}};x;j},
    q_{\mathring{\mathcal{I}};y;i}\right)` is calculated via Eq. :eq:`J_sq__1`,
    with the derivatives being calculated numerically using the second-order
    accurate central differences method.

    Parameters
    ----------
    coord_transform_params : :class:`distoptica.CoordTransformParams` | `None`, optional
        If ``coord_transform_params`` is set to ``None``, then the coordinate 
        transformation :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
        T_{⌑;y}\left(u_{x},u_{y}\right)\right)` to be used is the identity 
        transformation. Otherwise, ``coord_transform_params`` specifies the 
        parameters of the coordinate transformation to be used.
    sampling_grid_dims_in_pixels : `array_like` (`int`, shape=(2,)), optional
        The dimensions of the sampling grid, in units of pixels:
        ``sampling_grid_dims_in_pixels[0]`` and
        ``sampling_grid_dims_in_pixels[1]`` are 
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` respectively.
    device_name : `str` | `None`, optional
        This parameter specifies the device to be used to perform
        computationally intensive calls to PyTorch functions and where to store
        attributes of the type :class:`torch.Tensor`. If ``device_name`` is a
        string, then it is the name of the device to be used, e.g. ``”cuda”`` or
        ``”cpu”``. If ``device_name`` is set to ``None`` and a GPU device is
        available, then a GPU device is to be used. Otherwise, the CPU is used.
    least_squares_alg_params : :class:`distoptica.LeastSquaresAlgParams` | `None`, optional
        If ``least_squares_alg_params`` is set to ``None``, then the parameters
        of the least-squares algorithm to be used to calculate
        :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
        T_{\square;y}\left(q_{x},q_{y}\right)\right)` are those specified by
        ``distoptica.LeastSquaresAlgParams()``. Otherwise,
        ``least_squares_alg_params`` specifies the parameters of the
        least-squares algorithm to be used.
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
        self._is_standard = \
            coord_transform_params._is_corresponding_model_standard

        self._sampling_grid = self._calc_sampling_grid()

        self._flow_field_of_coord_transform = None
        self._renormalized_flow_field_of_coord_transform = None
        self._flow_field_of_coord_transform_right_inverse = None
        self._renormalized_flow_field_of_coord_transform_right_inverse = None

        self._convergence_map_of_distorted_then_resampled_images = None
        self._mask_frame_of_distorted_then_resampled_images = None

        self._jacobian_weights_for_distorting_then_resampling = None
        self._jacobian_weights_for_undistorting_then_resampling = None

        self._out_of_bounds_map_of_undistorted_then_resampled_images = None
        self._out_of_bounds_map_of_distorted_then_resampled_images = None

        return None



    def _get_device(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        device_name = self_core_attrs["device_name"]

        if device_name is None:
            cuda_is_available = torch.cuda.is_available()
            device_name = "cuda"*cuda_is_available + "cpu"*(1-cuda_is_available)

        device = torch.device(device_name)

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



    def _calc_sampling_grid(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)

        sampling_grid_dims_in_pixels = \
            self_core_attrs["sampling_grid_dims_in_pixels"]
        j_range = \
            torch.arange(sampling_grid_dims_in_pixels[0], device=self._device)
        i_range = \
            torch.arange(sampling_grid_dims_in_pixels[1], device=self._device)
        
        pair_of_1d_coord_arrays = ((j_range + 0.5) / j_range.numel(),
                                   1 - (i_range + 0.5) / i_range.numel())
        sampling_grid = torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")

        return sampling_grid



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    def distort_then_resample_images(self,
                                     undistorted_images=\
                                     _default_undistorted_images):
        r"""Distort then resample a 1D stack of undistorted images.

        See the summary documentation of the class
        :class:`distoptica.DistortionModel` for additional context.

        Each undistorted image is distorted and subsequently resampled according
        to Eq. :eq:`distorting_then_resampling_images__1`.

        Parameters
        ----------
        undistorted_images : `array_like` (`float`, ndim=2) | `array_like` (`float`, ndim=3) | `array_like` (`float`, ndim=4), optional
            The undistorted images to be distorted and resampled. If
            ``len(undistorted_images.shape)==4``, then for every quadruplet of
            nonnegative integers ``(l, k, n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``undistorted_images[l, k, n,
            m]``, ``undistorted_images[l, k, n, m]`` is interpreted to be the
            quantity :math:`\mathcal{I}_{\square;l,k,n,m}`, introduced in the
            summary documentation of the class
            :class:`distoptica.DistortionModel`, with the integers :math:`l`,
            :math:`k`, :math:`n`, and :math:`m` being equal to the values of
            ``l``, ``k``, ``n``, and ``m`` respectively, and the quadruplet of
            integers
            :math:`\left(N_{\mathcal{I}},N_{\mathcal{I};C},N_{\mathcal{I};y},
            N_{\mathcal{I};x}\right)`, introduced in the summary documentation
            of the class :class:`distoptica.DistortionModel`, being equal to the
            shape of ``undistorted_images``. If
            ``len(undistorted_images.shape)==3``, then for every quadruplet of
            nonnegative integers ``(k, n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``undistorted_images[k, n,
            m]``, ``undistorted_images[k, n, m]`` is interpreted to be the
            quantity :math:`\mathcal{I}_{\square;0,k,n,m}`, with the integers
            :math:`k`, :math:`n`, and :math:`m` being equal to the values of
            ``k``, ``n``, and ``m`` respectively, the triplet of integers
            :math:`\left(N_{\mathcal{I};C},N_{\mathcal{I};y},
            N_{\mathcal{I};x}\right)` being equal to the shape of
            ``undistorted_images``, and :math:`N_{\mathcal{I}}` being equal to
            unity. Otherwise, if ``len(undistorted_images.shape)==2``, then for
            every pair of nonnegative integers ``(n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``undistorted_images[n, m]``,
            ``undistorted_images[n, m]`` is interpreted to be the quantity
            :math:`\mathcal{I}_{\square;0,0,n,m}`, with the integers :math:`n`
            and :math:`m` being equal to the values of ``n``, and ``m``
            respectively, the pair of integers
            :math:`\left(N_{\mathcal{I};y},N_{\mathcal{I};x}\right)` being equal
            to the shape of ``undistorted_images``, and both
            :math:`N_{\mathcal{I}}` and :math:`N_{\mathcal{I};C}` being equal to
            unity.

        Returns
        -------
        distorted_then_resampled_images : `torch.Tensor` (`float`, ndim=4)
            The images resulting from distorting then resampling the input image
            set. For every quadruplet of nonnegative integers ``(l, k, i, j)``
            that does not raise an ``IndexError`` exception upon calling
            ``distorted_then_resampled_images[l, k, i, j]``,
            ``distorted_then_resampled_images[l, k, i, j]`` is interpreted to be
            the quantity :math:`\mathring{\mathcal{I}}_{⌑;l,k,i,j}`, introduced
            in the summary documentation of the class
            :class:`distoptica.DistortionModel`, with the integers :math:`l`,
            :math:`k`, :math:`i`, and :math:`j` being equal to the values of
            ``l``, ``k``, ``i``, and ``j`` respectively, with the quadruplet of
            integers :math:`\left(N_{\mathcal{I}},N_{\mathcal{I};C},
            N_{\mathring{\mathcal{I}};y},N_{\mathring{\mathcal{I}};x}\right)`,
            introduced in the summary documentation of the class
            :class:`distoptica.DistortionModel`, being equal to the shape of
            ``distorted_then_resampled_images``.

        """
        params = {"images": undistorted_images,
                  "name_of_alias_of_images": "undistorted_images",
                  "device": self._device}
        undistorted_images = _check_and_convert_images(params)

        distorted_then_resampled_images = \
            self._distort_then_resample_images(undistorted_images)

        return distorted_then_resampled_images



    def _distort_then_resample_images(self, undistorted_images):
        if self._jacobian_weights_for_distorting_then_resampling is None:
            self._update_attr_subset_1()

        num_images = undistorted_images.shape[0]
        
        sampling_grid = \
            self._sampling_grid
        flow_field = \
            self._renormalized_flow_field_of_coord_transform_right_inverse
        jacobian_weights = \
            self._jacobian_weights_for_distorting_then_resampling

        grid_shape = (num_images,) + jacobian_weights.shape + (2,)
        grid = torch.zeros(grid_shape,
                           dtype=jacobian_weights.dtype,
                           device=jacobian_weights.device)
        grid[:, :, :, 0] = flow_field[0][None, :, :]
        grid[:, :, :, 1] = flow_field[1][None, :, :]

        resampling_normalization_weight = (undistorted_images.shape[-2]
                                           * undistorted_images.shape[-1]
                                           / jacobian_weights.shape[0]
                                           / jacobian_weights.shape[1])

        kwargs = \
            {"input": undistorted_images,
             "grid": grid,
             "mode": "bilinear",
             "padding_mode": "zeros",
             "align_corners": False}
        distorted_then_resampled_images = \
            (torch.nn.functional.grid_sample(**kwargs)
             * jacobian_weights[None, None, :, :]
             * resampling_normalization_weight)

        return distorted_then_resampled_images



    def _update_attr_subset_1(self):
        sampling_grid = self._sampling_grid
        inputs = {"q_x": sampling_grid[0], "q_y": sampling_grid[1]}
            
        method_name = "initialize_levenberg_marquardt_alg_variables"
        method_alias = getattr(self._coord_transform_right_inverse, method_name)
        method_alias(inputs)

        q_x = self._coord_transform_right_inverse.q_hat_1[0]
        q_y = self._coord_transform_right_inverse.q_hat_1[1]

        self._flow_field_of_coord_transform = (q_x-sampling_grid[0],
                                               q_y-sampling_grid[1])
        self._renormalized_flow_field_of_coord_transform = (2*(q_x-0.5),
                                                            -2*(q_y-0.5))

        J = \
            self._coord_transform_right_inverse.J
        self._jacobian_weights_for_undistorting_then_resampling = \
                torch.abs(J[0, 0]*J[1, 1] - J[1, 0]*J[0, 1])

        self._out_of_bounds_map_of_undistorted_then_resampled_images = \
            ((q_x*q_x.shape[1]<0.5) | (q_x.shape[1]-0.5<q_x*q_x.shape[1])
             | (q_y*q_y.shape[0]<0.5) | (q_y.shape[0]-0.5<q_y*q_y.shape[0]))

        method_name = "eval_forward_output"
        method_alias = getattr(self._coord_transform_right_inverse, method_name)
        u_x, u_y = method_alias(inputs=dict())

        self._flow_field_of_coord_transform_right_inverse = \
            (u_x-sampling_grid[0], u_y-sampling_grid[1])
        self._renormalized_flow_field_of_coord_transform_right_inverse = \
            (2*(u_x-0.5), -2*(u_y-0.5))

        self._jacobian_weights_for_distorting_then_resampling = \
            self._calc_jacobian_weights_for_distorting_then_resampling(u_x, u_y)

        self._convergence_map_of_distorted_then_resampled_images = \
            self._coord_transform_right_inverse.convergence_map

        kwargs = \
            {"mat": self._coord_transform_right_inverse.convergence_map}
        self._mask_frame_of_distorted_then_resampled_images = \
            _calc_minimum_frame_to_mask_all_zero_valued_elems(**kwargs)

        self._out_of_bounds_map_of_distorted_then_resampled_images = \
            ((u_x*u_x.shape[1]<0.5) | (u_x.shape[1]-0.5<u_x*u_x.shape[1])
             | (u_y*u_y.shape[0]<0.5) | (u_y.shape[0]-0.5<u_y*u_y.shape[0]))

        return None



    def _calc_jacobian_weights_for_distorting_then_resampling(self, u_x, u_y):
        spacing = (self._sampling_grid[1][:, 0], self._sampling_grid[0][0, :])

        kwargs = {"input": u_x,
                  "spacing": spacing,
                  "dim": None,
                  "edge_order": 2}
        d_u_x_over_d_q_y, d_u_x_over_d_q_x = torch.gradient(**kwargs)

        kwargs["input"] = u_y
        d_u_y_over_d_q_y, d_u_y_over_d_q_x = torch.gradient(**kwargs)

        jacobian_weights_for_distorting_then_resampling = \
            torch.abs(d_u_x_over_d_q_x*d_u_y_over_d_q_y
                      - d_u_x_over_d_q_y*d_u_y_over_d_q_x)

        return jacobian_weights_for_distorting_then_resampling



    def undistort_then_resample_images(self,
                                       distorted_images=\
                                       _default_distorted_images):
        r"""Undistort and resample a 1D stack of distorted images.

        Each distorted image is undistorted and subsequently resampled according
        to Eq. :eq:`undistorting_then_resampling_images__1`.

        Parameters
        ----------
        distorted_images : `array_like` (`float`, ndim=2) | `array_like` (`float`, ndim=3) | `array_like` (`float`, ndim=4), optional
            The distorted images to be undistorted and resampled. If
            ``len(distorted_images.shape)==4``, then for every quadruplet of
            nonnegative integers ``(l, k, n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``distorted_images[l, k, n,
            m]``, ``distorted_images[l, k, n, m]`` is interpreted to be the
            quantity :math:`\mathcal{I}_{⌑;l,k,n,m}`, introduced in the summary
            documentation of the class :class:`distoptica.DistortionModel`, with
            the integers :math:`l`, :math:`k`, :math:`n`, and :math:`m` being
            equal to the values of ``l``, ``k``, ``n``, and ``m`` respectively,
            and the quadruplet of integers
            :math:`\left(N_{\mathcal{I}},N_{\mathcal{I};C},N_{\mathcal{I};y},
            N_{\mathcal{I};x}\right)`, introduced in the summary documentation
            of the class :class:`distoptica.DistortionModel`, being equal to the
            shape of ``distorted_images``. If
            ``len(distorted_images.shape)==3``, then for every quadruplet of
            nonnegative integers ``(k, n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``distorted_images[k, n, m]``,
            ``distorted_images[k, n, m]`` is interpreted to be the quantity
            :math:`\mathcal{I}_{⌑;0,k,n,m}`, with the integers :math:`k`,
            :math:`n`, and :math:`m` being equal to the values of ``c``, ``n``,
            and ``m`` respectively, the triplet of integers
            :math:`\left(N_{\mathcal{I};C},N_{\mathcal{I};y},
            N_{\mathcal{I};x}\right)` being equal to the shape of
            ``distorted_images``, and :math:`N_{\mathcal{I}}` being equal to
            unity. Otherwise, if ``len(distorted_images.shape)==2``, then for
            every pair of nonnegative integers ``(n, m)`` that does not raise an
            ``IndexError`` exception upon calling ``distorted_images[n, m]``,
            ``distorted_images[n, m]`` is interpreted to be the quantity
            :math:`\mathcal{I}_{⌑;0,0,n,m}`, with the integers :math:`n` and
            :math:`m` being equal to the values of ``n``, and ``m``
            respectively, the pair of integers
            :math:`\left(N_{\mathcal{I};y},N_{\mathcal{I};x}\right)` being equal
            to the shape of ``distorted_images``, and both
            :math:`N_{\mathcal{I}}` and :math:`N_{\mathcal{I};C}` being equal to
            unity.

        Returns
        -------
        undistorted_then_resampled_images : `torch.Tensor` (`float`, ndim=4)
            The images resulting from undistorting then resampling the input
            image set. For every quadruplet of nonnegative integers ``(l, k, i,
            j)`` that does not raise an ``IndexError`` exception upon calling
            ``undistorted_then_resampled_images[l, k, i, j]``,
            ``undistorted_then_resampled_images[l, k, i, j]`` is interpreted to
            be the quantity :math:`\mathring{\mathcal{I}}_{\square;l,k,i,j}`,
            introduced in the summary documentation of the class
            :class:`distoptica.DistortionModel`, with the integers :math:`l`,
            :math:`k`, :math:`i`, and :math:`j` being equal to the values of
            ``l``, ``k``, ``i``, and ``j`` respectively, with the quadruplet of
            integers :math:`\left(N_{\mathcal{I}},N_{\mathcal{I};C},
            N_{\mathring{\mathcal{I}};y},N_{\mathring{\mathcal{I}};x}\right)`,
            introduced in the summary documentation of the class
            :class:`distoptica.DistortionModel`, being equal to the shape of
            ``undistorted_then_resampled_images``.

        """
        params = {"images": distorted_images,
                  "name_of_alias_of_images": "distorted_images",
                  "device": self._device}
        distorted_images = _check_and_convert_images(params)

        undistorted_then_resampled_images = \
            self._undistort_then_resample_images(distorted_images)

        return undistorted_then_resampled_images



    def _undistort_then_resample_images(self, distorted_images):
        if self._jacobian_weights_for_undistorting_then_resampling is None:
            self._update_attr_subset_2()

        num_images = distorted_images.shape[0]

        sampling_grid = \
            self._sampling_grid
        flow_field = \
            self._renormalized_flow_field_of_coord_transform
        jacobian_weights = \
            self._jacobian_weights_for_undistorting_then_resampling

        grid_shape = (num_images,) + jacobian_weights.shape + (2,)
        grid = torch.zeros(grid_shape,
                           dtype=jacobian_weights.dtype,
                           device=jacobian_weights.device)
        grid[:, :, :, 0] = flow_field[0][None, :, :]
        grid[:, :, :, 1] = flow_field[1][None, :, :]

        resampling_normalization_weight = (distorted_images.shape[-2]
                                           * distorted_images.shape[-1]
                                           / jacobian_weights.shape[0]
                                           / jacobian_weights.shape[1])

        kwargs = \
            {"input": distorted_images,
             "grid": grid,
             "mode": "bilinear",
             "padding_mode": "zeros",
             "align_corners": False}
        undistorted_then_resampled_images = \
            (torch.nn.functional.grid_sample(**kwargs)
             * jacobian_weights[None, None, :, :]
             * resampling_normalization_weight)

        return undistorted_then_resampled_images



    def _update_attr_subset_2(self):
        sampling_grid = self._sampling_grid

        coord_transform_inputs = dict()

        method_name = "update_coord_transform_inputs"
        method_alias = getattr(self._coord_transform_right_inverse, method_name)
        kwargs = {"coord_transform_inputs": coord_transform_inputs,
                  "p": sampling_grid}
        method_alias(**kwargs)

        kwargs = {"coord_transform_inputs": \
                  coord_transform_inputs,
                  "coord_transform": \
                  self._coord_transform_right_inverse.coord_transform_1}
        q_x, q_y = self._coord_transform_right_inverse.eval_q_hat(**kwargs)

        self._flow_field_of_coord_transform = (q_x-sampling_grid[0],
                                               q_y-sampling_grid[1])
        self._renormalized_flow_field_of_coord_transform = (2*(q_x-0.5),
                                                            -2*(q_y-0.5))

        kwargs = \
            {"coord_transform_inputs": coord_transform_inputs}
        J = \
            self._coord_transform_right_inverse.eval_J(**kwargs)
        self._jacobian_weights_for_undistorting_then_resampling = \
            torch.abs(J[0, 0]*J[1, 1] - J[1, 0]*J[0, 1])

        self._out_of_bounds_map_of_undistorted_then_resampled_images = \
            ((q_x*q_x.shape[1]<0.5) | (q_x.shape[1]-0.5<q_x*q_x.shape[1])
             | (q_y*q_y.shape[0]<0.5) | (q_y.shape[0]-0.5<q_y*q_y.shape[0]))

        return None



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
        result = self._is_azimuthally_symmetric
        
        return result



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
        result = self._is_trivial
        
        return result



    @property
    def is_standard(self):
        r"""`bool`: A boolean variable indicating whether the distortion model 
        is standard.

        See the documentation for the class
        :class:`distoptica.StandardCoordTransformParams` for a definition of a
        standard distortion model.

        If ``is_standard`` is set to ``True``, then the distortion model is
         standard. Otherwise, the distortion model is not standard.

        Note that ``is_standard`` should be considered **read-only**.

        """
        result = self._is_standard
        
        return result



    def get_sampling_grid(self, deep_copy=_default_deep_copy):
        r"""Return the fractional coordinates of the sampling grid.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``sampling_grid`` denote the attribute
            :attr:`distoptica.DistortionModel.sampling_grid`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``sampling_grid`` is returned.  Otherwise, a reference to
            ``sampling_grid`` is returned.

        Returns
        -------
        sampling_grid : `array_like` (`torch.Tensor` (`float`, ndim=2), shape=(2,))
            The attribute :attr:`distoptica.DistortionModel.sampling_grid`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            sampling_grid = (self._sampling_grid[0].detach().clone(),
                             self._sampling_grid[1].detach().clone())
        else:
            sampling_grid = self._sampling_grid

        return sampling_grid



    @property
    def sampling_grid(self):
        r"""`array_like`: The fractional coordinates of the sampling grid.

        See the summary documentation of the class
        :class:`distoptica.DistortionModel` for additional context.

        ``sampling_grid`` is a 2-element tuple, where ``sampling_grid[0]`` and
        ``sampling_grid[1]`` are PyTorch tensors, each having a shape equal to
        :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, with
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        For every pair of nonnegative integers ``(i, j)`` that does not raise an
        ``IndexError`` exception upon calling ``sampling_grid[0, i, j]``,
        ``sampling_grid[0, i, j]`` and ``sampling_grid[1, i, j]`` are equal to
        the quantities :math:`u_{\mathcal{\mathring{I}};x;j}` and
        :math:`u_{\mathcal{\mathring{I}};y;i}` respectively, with the integers
        :math:`i` and :math:`j` being equal to the values of ``i`` and ``j``
        respectively, and :math:`u_{\mathcal{\mathring{I}};x;j}` and
        :math:`u_{\mathcal{\mathring{I}};y;i}` being given by
        Eqs. :eq:`u_I_circ_x_j__1` and :eq:`u_I_circ_y_i__1` respectively.

        Note that ``sampling_grid`` should be considered **read-only**.

        """
        result = self.get_sampling_grid(deep_copy=True)

        return result



    def get_convergence_map_of_distorted_then_resampled_images(
            self, deep_copy=_default_deep_copy):
        r"""Return the convergence map of the iterative algorithm used to 
        distort then resample images.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``convergence_map_of_distorted_then_resampled_images`` denote
            the attribute
            :attr:`distoptica.DistortionModel.convergence_map_of_distorted_then_resampled_images`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``convergence_map_of_distorted_then_resampled_images`` is returned.
            Otherwise, a reference to
            ``convergence_map_of_distorted_then_resampled_images`` is returned.

        Returns
        -------
        convergence_map_of_distorted_then_resampled_images : `torch.Tensor` (`bool`, ndim=2)
            The attribute
            :attr:`distoptica.DistortionModel.convergence_map_of_distorted_then_resampled_images`.

        """
        if self._convergence_map_of_distorted_then_resampled_images is None:
            self._update_attr_subset_1()

        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            convergence_map = \
                self._convergence_map_of_distorted_then_resampled_images
            convergence_map_of_distorted_then_resampled_images = \
                convergence_map.detach().clone()
        else:
            convergence_map_of_distorted_then_resampled_images = \
                self._convergence_map_of_distorted_then_resampled_images

        return convergence_map_of_distorted_then_resampled_images



    @property
    def convergence_map_of_distorted_then_resampled_images(self):
        r"""`torch.Tensor`: The convergence map of the iterative algorithm used 
        to distort then resample images.

        See the documentation for the method
        :meth:`distoptica.DistortionModel.distort_then_resample_images` for
        additional context.

        ``convergence_map_of_distorted_then_resampled_images`` is a PyTorch
        tensor having a shape equal to
        :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, where
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        Let ``distorted_then_resampled_images`` denote the output of a call to
        the method
        :meth:`distoptica.DistortionModel.distort_then_resample_images`.

        For every row index ``i`` and column index ``j``,
        ``convergence_map_of_distorted_then_resampled_images[i, j]`` evaluates
        to ``False`` if the iterative algorithm used to calculate
        ``distorted_images`` does not converge within the error tolerance for
        elements ``distorted_images[:, :, i, j]``, and evaluates to ``True``
        otherwise.

        Note that ``convergence_map_of_distorted_then_resampled_images`` should
        be considered **read-only**.

        """
        method_alias = \
            self.get_convergence_map_of_distorted_then_resampled_images
        result = \
            method_alias(deep_copy=True)

        return result



    @property
    def mask_frame_of_distorted_then_resampled_images(self):
        r"""`array_like`: The minimum frame to mask all boolean values of 
        ``False`` in the attribute 
        :attr:`distoptica.DistortionModel.convergence_map_of_distorted_then_resampled_images`.

        See the documentation for the attribute
        :attr:`distoptica.DistortionModel.convergence_map_of_distorted_then_resampled_images`
        for additional context.

        ``mask_frame_of_distorted_then_resampled_images`` is a 4-element tuple
        where ``mask_frame_of_distorted_then_resampled_images[0]``,
        ``mask_frame_of_distorted_then_resampled_images[1]``,
        ``mask_frame_of_distorted_then_resampled_images[2]``, and
        ``mask_frame_of_distorted_then_resampled_images[3]`` are the widths, in
        units of pixels, of the left, right, bottom, and top sides of the mask
        frame respectively. If all elements of
        ``mask_frame_of_distorted_then_resampled_images`` are equal to zero,
        then every element of
        :attr:`distoptica.DistortionModel.convergence_map_of_distorted_then_resampled_images`
        evaluates to ``True``.

        Note that ``mask_frame_of_distorted_then_resampled_images`` should be
        considered **read-only**.

        """
        if self._mask_frame_of_distorted_then_resampled_images is None:
            self._update_attr_subset_1()

        result = self._mask_frame_of_distorted_then_resampled_images

        return result



    def get_flow_field_of_coord_transform(self, deep_copy=_default_deep_copy):
        r"""Return the flow field of the coordinate transformation corresponding
        to the distortion model.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``flow_field_of_coord_transform`` denote the attribute
            :attr:`distoptica.DistortionModel.flow_field_of_coord_transform`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``flow_field_of_coord_transform`` is returned.  Otherwise, a
            reference to ``flow_field_of_coord_transform`` is returned.

        Returns
        -------
        flow_field_of_coord_transform : `array_like` (`torch.Tensor` (`float`, ndim=2), shape=(2,))
            The attribute
            :attr:`distoptica.DistortionModel.flow_field_of_coord_transform`.

        """
        if self._flow_field_of_coord_transform is None:
            self._update_attr_subset_2()

        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            flow_field = self._flow_field_of_coord_transform
            flow_field_of_coord_transform = (flow_field[0].detach().clone(),
                                             flow_field[1].detach().clone())
        else:
            flow_field_of_coord_transform = self._flow_field_of_coord_transform

        return flow_field_of_coord_transform



    @property
    def flow_field_of_coord_transform(self):
        r"""`array_like`: The flow field of the coordinate transformation 
        corresponding to the distortion model.

        See the summary documentation of the class
        :class:`distoptica.DistortionModel` for additional context.

        ``flow_field_of_coord_transform`` is a 2-element tuple, where
        ``flow_field_of_coord_transform[0]`` and
        ``flow_field_of_coord_transform[1]`` are PyTorch tensors, each having a
        shape equal to :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, with
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        For every pair of nonnegative integers ``(i, j)`` that does not raise an
        ``IndexError`` exception upon calling ``flow_field_of_coord_transform[0,
        i, j]``, ``flow_field_of_coord_transform[0, i, j]`` and
        ``flow_field_of_coord_transform[1, i, j]`` are equal to the quantities

        .. math ::
            \Delta T_{⌑;x;i,j}=
            T_{⌑;x}\left(u_{\mathring{\mathcal{I}};x;j},
            u_{\mathring{\mathcal{I}};y;i}\right)
            -u_{\mathring{\mathcal{I}};x;j},
            :label: flow_field_of_coord_transform__1

        and

        .. math ::
            \Delta T_{⌑;y;i,j}=
            T_{⌑;y}\left(u_{\mathring{\mathcal{I}};x;j},
            u_{\mathring{\mathcal{I}};y;i}\right)
            -u_{\mathring{\mathcal{I}};y;j},
            :label: flow_field_of_coord_transform__2

        respectively, with the integers :math:`i` and :math:`j` being equal to
        the values of ``i`` and ``j`` respectively,
        :math:`u_{\mathcal{\mathring{I}};x;j}` and
        :math:`u_{\mathcal{\mathring{I}};y;i}` being given by
        Eqs. :eq:`u_I_circ_x_j__1` and :eq:`u_I_circ_y_i__1` respectively, and
        :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
        T_{⌑;x}\left(u_{x},u_{y}\right)\right)` being the coordinate
        transformation corresponding to the distortion model.

        Note that ``flow_field_of_coord_transform`` should be considered
        **read-only**.

        """
        method_alias = \
            self.get_flow_field_of_coord_transform
        result = \
            method_alias(deep_copy=True)

        return result



    def get_flow_field_of_coord_transform_right_inverse(
            self, deep_copy=_default_deep_copy):
        r"""Return the flow field of the right-inverse of the coordinate 
        transformation corresponding to the distortion model.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``flow_field_of_coord_transform_right_inverse`` denote the 
            attribute
            :attr:`distoptica.DistortionModel.flow_field_of_coord_transform_right_inverse`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``flow_field_of_coord_transform_right_inverse`` is returned.
            Otherwise, a reference to
            ``flow_field_of_coord_transform_right_inverse`` is returned.

        Returns
        -------
        flow_field_of_coord_transform_right_inverse : `array_like` (`torch.Tensor` (`float`, ndim=2), shape=(2,))
            The attribute
            :attr:`distoptica.DistortionModel.flow_field_of_coord_transform_right_inverse`.

        """
        if self._flow_field_of_coord_transform_right_inverse is None:
            self._update_attr_subset_1()

        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            flow_field = \
                self._flow_field_of_coord_transform_right_inverse
            flow_field_of_coord_transform_right_inverse = \
                (flow_field[0].detach().clone(), flow_field[1].detach().clone())
        else:
            flow_field_of_coord_transform_right_inverse = \
                self._flow_field_of_coord_transform_right_inverse

        return flow_field_of_coord_transform_right_inverse



    @property
    def flow_field_of_coord_transform_right_inverse(self):
        r"""`array_like`: The flow field of the right-inverse of the coordinate 
        transformation corresponding to the distortion model.

        ``flow_field_of_coord_transform_right_inverse`` is a 2-element tuple,
        where ``flow_field_of_coord_transform_right_inverse[0]`` and
        ``flow_field_of_coord_transform_right_inverse[1]`` are PyTorch tensors,
        each having a shape equal to :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, with
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        For every pair of nonnegative integers ``(i, j)`` that does not raise an
        ``IndexError`` exception upon calling
        ``flow_field_of_coord_transform_right_inverse[0, i, j]``,
        ``flow_field_of_coord_transform_right_inverse[0, i, j]`` and
        ``flow_field_of_coord_transform_right_inverse[1, i, j]`` are equal to
        the quantities

        .. math ::
            \Delta T_{\square;x;i,j}=
            T_{\square;x}\left(q_{\mathring{\mathcal{I}};x;j},
            q_{\mathring{\mathcal{I}};y;i}\right)
            -q_{\mathring{\mathcal{I}};x;j},
            :label: flow_field_of_coord_transform_right_inverse__1

        and

        .. math ::
            \Delta T_{\square;y;i,j}=
            T_{\square;y}\left(q_{\mathring{\mathcal{I}};x;j},
            q_{\mathring{\mathcal{I}};y;i}\right)
            -q_{\mathring{\mathcal{I}};y;j},
            :label: flow_field_of_coord_transform_right_inverse__2

        respectively, with the integers :math:`i` and :math:`j` being equal to
        the values of ``i`` and ``j`` respectively,
        :math:`q_{\mathcal{\mathring{I}};x;j}` and
        :math:`q_{\mathcal{\mathring{I}};y;i}` being given by
        Eqs. :eq:`q_I_circ_x_j__1` and :eq:`q_I_circ_y_i__1` respectively, and
        :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
        T_{\square;x}\left(q_{x},q_{y}\right)\right)` being the right inverse of
        the coordinate transformation corresponding to the distortion model.

        Note that ``flow_field_of_coord_transform_right_inverse`` should be
        considered **read-only**.

        """
        method_alias = \
            self.get_flow_field_of_coord_transform_right_inverse
        result = \
            method_alias(deep_copy=True)

        return result



    def get_out_of_bounds_map_of_undistorted_then_resampled_images(
            self, deep_copy=_default_deep_copy):
        r"""Return the out-of-bounds map of undistorted then resampled images.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``out_of_bounds_map_of_undistorted_then_resampled_images``
            denote the attribute
            :attr:`distoptica.DistortionModel.out_of_bounds_map_of_undistorted_then_resampled_images`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``out_of_bounds_map_of_undistorted_then_resampled_images`` is
            returned.  Otherwise, a reference to
            ``out_of_bounds_map_of_undistorted_then_resampled_images`` is
            returned.

        Returns
        -------
        out_of_bounds_map_of_undistorted_then_resampled_images : `torch.Tensor` (`bool`, ndim=2)
            The attribute
            :attr:`distoptica.DistortionModel.out_of_bounds_map_of_undistorted_then_resampled_images`.

        """
        if self._out_of_bounds_map_of_undistorted_then_resampled_images is None:
            self._update_attr_subset_2()

        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            out_of_bounds_map = \
                self._out_of_bounds_map_of_undistorted_then_resampled_images
            out_of_bounds_map_of_undistorted_then_resampled_images = \
                out_of_bounds_map.detach().clone()
        else:
            out_of_bounds_map_of_undistorted_then_resampled_images = \
                self._out_of_bounds_map_of_undistorted_then_resampled_images

        return out_of_bounds_map_of_undistorted_then_resampled_images



    @property
    def out_of_bounds_map_of_undistorted_then_resampled_images(self):
        r"""`torch.Tensor`: The out-of-bounds map of undistorted then resampled 
        images.

        See the summary documentation of the class
        :class:`distoptica.DistortionModel` for additional context.

        ``out_of_bounds_map_of_undistorted_then_resampled_images`` is a PyTorch
        tensor having a shape equal to
        :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, where
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        Let ``sampling_grid`` and ``flow_field_of_coord_transform`` denote the
        attributes :attr:`distoptica.DistortionModel.sampling_grid` and
        :attr:`distoptica.DistortionModel.flow_field_of_coord_transform`
        respectively. Furthermore, let ``q_x =
        sampling_grid[0]+flow_field_of_coord_transform[0]`` and ``q_y =
        sampling_grid[1]+flow_field_of_coord_transform[1]``.

        ``out_of_bounds_map_of_undistorted_then_resampled_images`` is equal to
        ``(q_x*q_x.shape[1]<=0.5) | (q_x.shape[1]-0.5<=q_x*q_x.shape[1]) |
        (q_y*q_y.shape[0]<=0.5) | (q_y.shape[0]-0.5<=q_y*q_y.shape[0])``.

        Note that ``out_of_bounds_map_of_coord_transform`` should be considered
        **read-only**.

        """
        method_alias = \
            self.get_out_of_bounds_map_of_undistorted_then_resampled_images
        result = \
            method_alias(deep_copy=True)

        return result



    def get_out_of_bounds_map_of_distorted_then_resampled_images(
            self, deep_copy=_default_deep_copy):
        r"""Return the out-of-bounds map of distorted then resampled images.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``out_of_bounds_map_of_distorted_then_resampled_images``
            denote the attribute
            :attr:`distoptica.DistortionModel.out_of_bounds_map_of_distorted_then_resampled_images`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``out_of_bounds_map_of_distorted_then_resampled_images`` is
            returned.  Otherwise, a reference to
            ``out_of_bounds_map_of_distorted_then_resampled_images`` is
            returned.

        Returns
        -------
        out_of_bounds_map_of_distorted_then_resampled_images : `torch.Tensor` (`bool`, ndim=2)
            The attribute
            :attr:`distoptica.DistortionModel.out_of_bounds_map_of_distorted_then_resampled_images`.

        """
        if self._out_of_bounds_map_of_distorted_then_resampled_images is None:
            self._update_attr_subset_1()

        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if (deep_copy == True):
            out_of_bounds_map = \
                self._out_of_bounds_map_of_distorted_then_resampled_images
            out_of_bounds_map_of_distorted_then_resampled_images = \
                out_of_bounds_map.detach().clone()
        else:
            out_of_bounds_map_of_distorted_then_resampled_images = \
                self._out_of_bounds_map_of_distorted_then_resampled_images

        return out_of_bounds_map_of_distorted_then_resampled_images



    @property
    def out_of_bounds_map_of_distorted_then_resampled_images(self):
        r"""`torch.Tensor`: The out-of-bounds map of distorted then resampled 
        images.

        See the summary documentation of the class
        :class:`distoptica.DistortionModel` for additional context.

        ``out_of_bounds_map_of_distorted_then_resampled_images`` is a PyTorch
        tensor having a shape equal to
        :math:`\left(N_{\mathring{\mathcal{I}};y},
        N_{\mathring{\mathcal{I}};x}\right)`, where
        :math:`N_{\mathring{\mathcal{I}};x}` and
        :math:`N_{\mathring{\mathcal{I}};y}` being the number of pixels in the
        sampling grid from left to right and top to bottom respectively.

        Let ``sampling_grid`` and
        ``flow_field_of_coord_transform_right_inverse`` denote the attributes
        :attr:`distoptica.DistortionModel.sampling_grid` and
        :attr:`distoptica.DistortionModel.flow_field_of_coord_transform_right_inverse`
        respectively. Furthermore, let ``u_x =
        sampling_grid[0]+flow_field_of_coord_transform_right_inverse[0]`` and
        ``u_y =
        sampling_grid[1]+flow_field_of_coord_transform_right_inverse[1]``.

        ``out_of_bounds_map_of_distorted_then_resampled_images`` is equal to
        ``(u_x*u_x.shape[1]<=0.5) | (u_x.shape[1]-0.5<=u_x*u_x.shape[1]) |
        (u_y*u_y.shape[0]<=0.5) | (u_y.shape[0]-0.5<=u_y*u_y.shape[0])``.

        Note that ``out_of_bounds_map_of_coord_transform`` should be considered
        **read-only**.

        """
        method_alias = \
            self.get_out_of_bounds_map_of_distorted_then_resampled_images
        result = \
            method_alias(deep_copy=True)

        return result



    @property
    def device(self):
        r"""`torch.device`: The device on which computationally intensive 
        PyTorch operations are performed and attributes of the type 
        :class:`torch.Tensor` are stored.

        Note that ``device`` should be considered **read-only**.

        """
        result = copy.deepcopy(self._device)

        return result



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
    r"""The parameters of a standard trigonometric series defining a coordinate 
    transformation.

    Users are encouraged to read the summary documentation of the classes
    :class:`distoptica.DistortionModel` and
    :class:`distoptica.CoordTransformParams` before reading the documentation
    for the current class as it provides essential context to what is discussed
    below.

    Motivated by transmission electron microscopy experiments, we define a class
    of standard coordinate transformations, where each standard coordinate
    transformation comprises of two components:
    :math:`T_{⌑;x}^{\left(\text{std}\right)}\left(u_{x},u_{y}\right)` and
    :math:`T_{⌑;y}^{\left(\text{std}\right)}\left(u_{x},u_{y}\right)`.
    :math:`T_{⌑;x}^{\left(\text{std}\right)}\left(u_{x},u_{y}\right)` and
    :math:`T_{⌑;y}^{\left(\text{std}\right)}\left(u_{x},u_{y}\right)` have the
    mathematical forms of Eqs. :eq:`T_distsq_x__1` and :eq:`T_distsq_y__1`
    respectively, only that the radial cosine coefficient matrix
    :math:`A_{r;v_{1},v_{2}}` has the following form:

    .. math ::
        A_{r;v_{1},v_{2}}=\begin{pmatrix}0 & 0 & A_{r;0,2}\\
        0 & A_{r;1,1} & 0\\
        A_{r;2,0} & 0 & 0
        \end{pmatrix},
        :label: std_radial_cosine_coefficient_matrix__1

    the radial sine coefficient matrix :math:`B_{r;v_{1},v_{2}}` has the
    following form:

    .. math ::
        B_{r;v_{1},v_{2}}=\begin{pmatrix}0 & B_{r;0,1}\\
        B_{r;1,0} & 0
        \end{pmatrix},
        :label: std_radial_sine_coefficient_matrix__1

    the tangential cosine coefficient matrix :math:`A_{t;v_{1},v_{2}}` has the
    following form:

    .. math ::
        A_{t;v_{1},v_{2}}=\begin{pmatrix}0 & 0 & A_{t;0,2}\\
        0 & A_{t;1,1} & 0\\
        A_{t;2,0} & 0 & 0
        \end{pmatrix},
        :label: std_tangential_cosine_coefficient_matrix__1

    with

    .. math ::
        A_{t;2,0}=B_{r;1,0},
        :label: A_t_2_0__1

    .. math ::
        A_{t;1,1}=\frac{1}{3}B_{r;0,1},
        :label: A_r_1_1__1

    and the tangential sine coefficient matrix :math:`B_{t;v_{1},v_{2}}` has the
    following form:

    .. math ::
        B_{t;v_{1},v_{2}}=\begin{pmatrix}0 & B_{t;0,1}\\
        B_{t;1,0} & 0
        \end{pmatrix},
        :label: std_tangential_sine_coefficient_matrix__1

    with

    .. math ::
        B_{t;0,1}=-\frac{1}{3}A_{r;1,1},
        :label: B_t_0_1__1
    
    .. math ::
        B_{t;1,0}=-A_{r;2,0}.
        :label: B_t_1_0__1

    Parameters
    ----------
    center : `array_like` (`float`, shape=(2,)), optional
        The distortion center :math:`\left(x_{c;D},y_{c;D}\right)`, introduced
        in the summary documentation of the class
        :class:`distoptica.CoordTransformParams`, where ``center[0]`` and
        ``center[1]`` are :math:`x_{c;D}` and :math:`y_{c;D}` respectively.
    quadratic_radial_distortion_amplitude : `float`, optional
        The coefficient :math:`A_{r;0,2}`, which we refer to as the quadratic
        radial distortion amplitude.
    elliptical_distortion_vector : `array_like` (`float`, shape=(2,)), optional
        The coefficient pair :math:`\left(A_{r;2,0},B_{r;1,0}\right)`, which we
        refer to as the elliptical distortion vector, where
        ``elliptical_distortion_vector[0]`` and
        ``elliptical_distortion_vector[1]`` are :math:`A_{r;2,0}` and
        :math:`B_{r;1,0}` respectively.
    spiral_distortion_amplitude : `float`, optional
        The coefficient :math:`A_{t;0,2}`, which we refer to as the spiral
        distortion amplitude.
    parabolic_distortion_vector : `array_like` (`float`, shape=(2,)), optional
        The coefficient pair :math:`\left(A_{r;1,1},B_{r;0,1}\right)`, which we
        refer to as the parabolic distortion vector, where
        ``parabolic_distortion_vector[0]`` and
        ``parabolic_distortion_vector[1]`` are :math:`A_{r;1,1}` and
        :math:`B_{r;0,1}` respectively.
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
        self_core_attrs = self.get_core_attrs(deep_copy=False)

        quadratic_radial_distortion_amplitude = \
            self_core_attrs["quadratic_radial_distortion_amplitude"]
        elliptical_distortion_vector = \
            self_core_attrs["elliptical_distortion_vector"]
        spiral_distortion_amplitude = \
            self_core_attrs["spiral_distortion_amplitude"]
        parabolic_distortion_vector = \
            self_core_attrs["parabolic_distortion_vector"]

        if ((np.linalg.norm(elliptical_distortion_vector) == 0)
            and (np.linalg.norm(parabolic_distortion_vector) == 0)
            and np.abs(spiral_distortion_amplitude) == 0):
            self._is_corresponding_model_azimuthally_symmetric = True
            if np.abs(quadratic_radial_distortion_amplitude) == 0:
                self._is_corresponding_model_trivial = True
            else:
                self._is_corresponding_model_trivial = False
        else:
            self._is_corresponding_model_azimuthally_symmetric = False
            self._is_corresponding_model_trivial = False

        return None



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def is_corresponding_model_azimuthally_symmetric(self):
        r"""`bool`: A boolean variable indicating whether the corresponding 
        distortion model is azimuthally symmetric.

        If ``is_corresponding_model_azimuthally_symmetric`` is set to ``True``,
        then the distortion model corresponding to the coordinate transformation
        parameters is azimuthally symmetric. Otherwise, the distortion model is
        not azimuthally symmetric.

        Note that ``is_corresponding_model_azimuthally_symmetric`` should be
        considered **read-only**.

        """
        result = self._is_corresponding_model_azimuthally_symmetric
        
        return result



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
        result = self._is_corresponding_model_trivial
    
        return result



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



def _check_and_convert_skip_validation_and_conversion(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    skip_validation_and_conversion = czekitout.convert.to_bool(**kwargs)

    return skip_validation_and_conversion



_default_standard_coord_transform_params = None



def generate_standard_distortion_model(standard_coord_transform_params=\
                                       _default_standard_coord_transform_params,
                                       sampling_grid_dims_in_pixels=\
                                       _default_sampling_grid_dims_in_pixels,
                                       device_name=\
                                       _default_device_name,
                                       least_squares_alg_params=\
                                       _default_least_squares_alg_params,
                                       skip_validation_and_conversion=\
                                       _default_skip_validation_and_conversion):
    r"""Generate a “standard” optical distortion model.

    Users are encouraged to read the summary documentation of the classes
    :class:`distoptica.DistortionModel`,
    :class:`distoptica.CoordTransformParams`, and
    :class:`distoptica.StandardCoordTransformParams` before reading the
    documentation for the current function as it provides essential context to
    what is discussed below.

    Parameters
    ----------
    standard_coord_transform_params : :class:`distoptica.StandardCoordTransformParams` | `None`, optional
        If ``standard_coord_transform_params`` is set to ``None``, then the
        coordinate transformation :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
        T_{⌑;y}\left(u_{x},u_{y}\right)\right)` to be used is the identity
        transformation. Otherwise, ``standard_coord_transform_params`` specifies
        the parameters of the standard coordinate transformation to be used.
    sampling_grid_dims_in_pixels : `array_like` (`int`, shape=(2,)), optional
        The dimensions of the sampling grid, in units of pixels:
        ``sampling_grid_dims_in_pixels[0]`` and
        ``sampling_grid_dims_in_pixels[1]`` are the number of pixels in the
        sampling grid from left to right and top to bottom respectively.
    device_name : `str` | `None`, optional
        This parameter specifies the device to be used to perform
        computationally intensive calls to PyTorch functions and where to store
        attributes of the type :class:`torch.Tensor`. If ``device_name`` is a
        string, then it is the name of the device to be used, e.g. ``”cuda”`` or
        ``”cpu”``. If ``device_name`` is set to ``None`` and a GPU device is
        available, then a GPU device is to be used. Otherwise, the CPU is used.
    least_squares_alg_params : :class:`distoptica.LeastSquaresAlgParams` | `None`, optional
        If ``least_squares_alg_params`` is set to ``None``, then the parameters
        of the least-squares algorithm to be used to calculate the functions
        :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
        T_{\square;y}\left(q_{x},q_{y}\right)\right)`, i.e. the functions
        defined by Eqs. :eq:`defining_T_sq_x__1` and :eq:`defining_T_sq_y__1`,
        are those specified by
        ``distoptica.LeastSquaresAlgParams()``. Otherwise,
        ``least_squares_alg_params`` specifies the parameters of the
        least-squares algorithm to be used.
    skip_validation_and_conversion : `bool`, optional
        If ``skip_validation_and_conversion`` is set to ``False``, then
        validations and conversions are performed on the above
        parameters. 

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``, no
        validations and conversions are performed on the above parameters. This
        option is desired primarily when the user wants to avoid potentially
        expensive validation and/or conversion operations.

    Returns
    -------
    distortion_model : :class:`distoptica.DistortionModel`
        The distortion model generated, according to the above parameters.

    """
    params = locals()

    func_alias = _check_and_convert_skip_validation_and_conversion
    skip_validation_and_conversion = func_alias(params)

    if (skip_validation_and_conversion == False):
        for param_name in params:
            if param_name == "skip_validation_and_conversion":
                continue
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
                                        least_squares_alg_params,
                                        skip_validation_and_conversion):
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
              tangential_sine_coefficient_matrix,
              "skip_validation_and_conversion": \
              skip_validation_and_conversion}
    coord_transform_params = CoordTransformParams(**kwargs)

    kwargs = {"coord_transform_params": coord_transform_params,
              "sampling_grid_dims_in_pixels": sampling_grid_dims_in_pixels,
              "device_name": device_name,
              "least_squares_alg_params": least_squares_alg_params,
              "skip_validation_and_conversion": skip_validation_and_conversion}
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

    A_r_0_2 = quadratic_radial_distortion_amplitude
    A_r_2_0 = elliptical_distortion_vector[0]
    A_r_1_1 = parabolic_distortion_vector[0]

    radial_cosine_coefficient_matrix = ((0.00000, 0.00000, A_r_0_2),
                                        (0.00000, A_r_1_1, 0.00000), 
                                        (A_r_2_0, 0.00000, 0.00000))

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

    B_r_1_0 = elliptical_distortion_vector[1]
    B_r_0_1 = parabolic_distortion_vector[1]

    radial_sine_coefficient_matrix = ((0.00000, B_r_0_1),
                                      (B_r_1_0, 0.00000))

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

    A_t_0_2 = spiral_distortion_amplitude
    A_t_2_0 = elliptical_distortion_vector[1]
    A_t_1_1 = parabolic_distortion_vector[1]/3

    tangential_cosine_coefficient_matrix = ((0.00000, 0.00000, A_t_0_2),
                                            (0.00000, A_t_1_1, 0.00000), 
                                            (A_t_2_0, 0.00000, 0.00000))

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

    B_t_1_0 = -elliptical_distortion_vector[0]
    B_t_0_1 = -parabolic_distortion_vector[0]/3

    tangential_sine_coefficient_matrix = ((0.00000, B_t_0_1), 
                                          (B_t_1_0, 0.00000))

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

_check_and_convert_real_torch_matrix_err_msg_1 = \
    ("The object ``{}`` must be a real-valued matrix.")

_check_and_convert_images_err_msg_1 = \
    ("The object ``{}`` must be a real-valued 2D, 3D, or 4D array.")

_check_and_convert_device_name_err_msg_1 = \
    ("The object ``device_name`` must be either of the type `NoneType` or "
     "`str`, wherein the latter case, ``device_name`` must be a valid device "
     "name.")
