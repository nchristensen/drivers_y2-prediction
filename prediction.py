"""mirgecom driver for the Y2 prediction."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
from pytools.obj_array import make_obj_array
from functools import partial

from arraycontext import thaw, freeze
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray
from grudge.eager import EagerDGDiscretization
from grudge.discretization import make_discretization_collection
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import VolumeDomainTag, DOFDesc
from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    set_sim_state, logmgr_add_device_memory_usage
)

from mirgecom.artificial_viscosity import smoothness_indicator
from mirgecom.simutil import (
    check_step,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    get_sim_timestep
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)

from mirgecom.fluid import make_conserved
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
)
from mirgecom.diffusion import DirichletDiffusionBoundary
#from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
import cantera
from mirgecom.eos import IdealSingleGas, PyrometheusMixture
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state
from mirgecom.wall_model import WallModel
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_grad_t_operator,
    coupled_ns_heat_operator
)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class SparkSource:
    r"""Energy deposition from a ignition source"

    Internal energy is deposited as a gaussian  of the form:

    .. math::

        e &= e + e_{a}\exp^{(1-r^{2})}\\

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, dim, center=None, width=1.0,
                 amplitude=0., amplitude_func=None):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        center: numpy.ndarray
            center of source
        amplitude: float
            source strength modifier
        amplitude_fun: function
            variation of amplitude with time
        """

        if center is None:
            center = np.zeros(shape=(dim,))
        self._center = center
        self._dim = dim
        self._amplitude = amplitude
        self._width = width
        self._amplitude_func = amplitude_func

    def __call__(self, x_vec, cv, time, **kwargs):
        """
        Create the energy deposition at time *t* and location *x_vec*.

        the source at time *t* is created by evaluting the gaussian
        with time-dependent amplitude at *t*.

        Parameters
        ----------
        cv: :class:`mirgecom.gas_model.FluidState`
            Fluid state object with the conserved and thermal state.
        time: float
            Current time at which the solution is desired
        x_vec: numpy.ndarray
            Nodal coordinates
        """

        t = time
        if self._amplitude_func is not None:
            amplitude = self._amplitude*self._amplitude_func(t)
        else:
            amplitude = self._amplitude

        #print(f"{time=} {amplitude=}")

        loc = self._center

        # coordinates relative to lump center
        rel_center = make_obj_array(
            [x_vec[i] - loc[i] for i in range(self._dim)]
        )
        actx = x_vec[0].array_context
        r = actx.np.sqrt(np.dot(rel_center, rel_center))
        expterm = amplitude * actx.np.exp(-(r**2)/(2*self._width*self._width))

        mass = 0*cv.mass
        momentum = 0*cv.momentum
        species_mass = 0*cv.species_mass

        energy = cv.energy + cv.mass*expterm

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=momentum, species_mass=species_mass)


def sponge_source(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness) *
            ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


def get_y_from_x(x, data):
    """
    Return the linearly interpolated the value of y
    from the value in data(x,y) at x
    """

    if x <= data[0][0]:
        y = data[0][1]
    elif x >= data[-1][0]:
        y = data[-1][1]
    else:
        ileft = 0
        iright = data.shape[0]-1

        # find the bracketing points, simple subdivision search
        while iright - ileft > 1:
            ind = int(ileft+(iright - ileft)/2)
            if x < data[ind][0]:
                iright = ind
            else:
                ileft = ind

        leftx = data[ileft][0]
        rightx = data[iright][0]
        lefty = data[ileft][1]
        righty = data[iright][1]

        dx = rightx - leftx
        dy = righty - lefty
        y = lefty + (x - leftx)*dy/dx
    return y


def get_theta_from_data(data):
    """
    Calculate theta = arctan(dy/dx)
    Where data[][0] = x and data[][1] = y
    """

    theta = data.copy()
    for index in range(1, theta.shape[0]-1):
        #print(f"index {index}")
        theta[index][1] = np.arctan((data[index+1][1]-data[index-1][1]) /
                          (data[index+1][0]-data[index-1][0]))
    theta[0][1] = np.arctan(data[1][1]-data[0][1])/(data[1][0]-data[0][0])
    theta[-1][1] = np.arctan(data[-1][1]-data[-2][1])/(data[-1][0]-data[-2][0])
    return(theta)


def smooth_step(actx, x, epsilon=1e-12):
    # return actx.np.tanh(x)
    # return actx.np.where(
    #     actx.np.greater(x, 0),
    #     actx.np.tanh(x)**3,
    #     0*x)
    return (
        actx.np.greater(x, 0) * actx.np.less(x, 1) * (1 - actx.np.cos(np.pi*x))/2
        + actx.np.greater(x, 1))


class InitACTII:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0, geom_top, geom_bottom,
            P0, T0, temp_wall, temp_sigma, vel_sigma, gamma_guess,
            mass_frac=None,
            inj_pres, inj_temp, inj_vel, inj_mass_frac=None,
            inj_gamma_guess,
            inj_temp_sigma, inj_vel_sigma,
            inj_ytop, inj_ybottom,
            inj_mach
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        gamma_guess: float
            guesstimate for gamma
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        geom_top: numpy.ndarray
            coordinates for the top wall
        geom_bottom: numpy.ndarray
            coordinates for the bottom wall
        """

        if mass_frac is None:
            if nspecies > 0:
                mass_frac = np.zeros(shape=(nspecies,))

        if inj_mass_frac is None:
            if nspecies > 0:
                inj_mass_frac = np.zeros(shape=(nspecies,))

        if inj_vel is None:
            inj_vel = np.zeros(shape=(dim,))

        self._dim = dim
        self._nspecies = nspecies
        self._P0 = P0
        self._T0 = T0
        self._geom_top = geom_top
        self._geom_bottom = geom_bottom
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._gamma_guess = gamma_guess
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298
        self._mass_frac = mass_frac

        self._inj_P0 = inj_pres
        self._inj_T0 = inj_temp
        self._inj_vel = inj_vel
        self._inj_gamma_guess = inj_gamma_guess

        self._temp_sigma_injection = inj_temp_sigma
        self._vel_sigma_injection = inj_vel_sigma
        self._inj_mass_frac = inj_mass_frac
        self._inj_ytop = inj_ytop
        self._inj_ybottom = inj_ybottom
        self._inj_mach = inj_mach

    def __call__(self, discr, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        ytop = 0*x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0

        mach = zeros
        ytop = zeros
        ybottom = zeros
        theta = zeros
        gamma = self._gamma_guess

        theta_geom_top = get_theta_from_data(self._geom_top)
        theta_geom_bottom = get_theta_from_data(self._geom_bottom)

        # process the mesh piecemeal, one interval at a time
        # linearly interpolate between the data points
        area_ratio = ((self._geom_top[0][1] - self._geom_bottom[0][1]) /
                      self._throat_height)
        if self._geom_top[0][0] < self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=0.01)
        elif self._geom_top[0][0] > self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=1.01)
        else:
            mach_left = 1.0
        x_left = self._geom_top[0][0]
        ytop_left = self._geom_top[0][1]
        ybottom_left = self._geom_bottom[0][1]
        theta_top_left = theta_geom_top[0][1]
        theta_bottom_left = theta_geom_bottom[0][1]

        for ind in range(1, self._geom_top.shape[0]):
            area_ratio = ((self._geom_top[ind][1] - self._geom_bottom[ind][1]) /
                          self._throat_height)
            if self._geom_top[ind][0] < self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=0.01)
            elif self._geom_top[ind][0] > self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=1.01)
            else:
                mach_right = 1.0
            ytop_right = self._geom_top[ind][1]
            ybottom_right = self._geom_bottom[ind][1]
            theta_top_right = theta_geom_top[ind][1]
            theta_bottom_right = theta_geom_bottom[ind][1]

            # interpolate our data
            x_right = self._geom_top[ind][0]

            dx = x_right - x_left
            dm = mach_right - mach_left
            dyt = ytop_right - ytop_left
            dyb = ybottom_right - ybottom_left
            dtb = theta_bottom_right - theta_bottom_left
            dtt = theta_top_right - theta_top_left

            local_mach = mach_left + (xpos - x_left)*dm/dx
            local_ytop = ytop_left + (xpos - x_left)*dyt/dx
            local_ybottom = ybottom_left + (xpos - x_left)*dyb/dx
            local_theta_bottom = theta_bottom_left + (xpos - x_left)*dtb/dx
            local_theta_top = theta_top_left + (xpos - x_left)*dtt/dx

            local_theta = (local_theta_bottom +
                           (local_theta_top - local_theta_bottom) /
                           (local_ytop - local_ybottom)*(ypos - local_ybottom))

            # extend just a a little bit to catch the edges
            left_edge = actx.np.greater(xpos, x_left - 1.e-6)
            right_edge = actx.np.less(xpos, x_right + 1.e-6)
            inside_block = left_edge*right_edge

            mach = actx.np.where(inside_block, local_mach, mach)
            ytop = actx.np.where(inside_block, local_ytop, ytop)
            ybottom = actx.np.where(inside_block, local_ybottom, ybottom)
            theta = actx.np.where(inside_block, local_theta, theta)

            mach_left = mach_right
            ytop_left = ytop_right
            ybottom_left = ybottom_right
            theta_bottom_left = theta_bottom_right
            theta_top_left = theta_top_right
            x_left = x_right

        pressure = getIsentropicPressure(
            mach=mach,
            P0=self._P0,
            gamma=gamma
        )
        temperature = getIsentropicTemperature(
            mach=mach,
            T0=self._T0,
            gamma=gamma
        )

        # save the unsmoothed temerature, so we can use it with the injector init
        unsmoothed_temperature = temperature

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(
            actx, sigma*actx.np.abs(ypos-ybottom))
        smoothing_fore = ones
        smoothing_aft = ones
        z0 = 0.
        z1 = 0.035
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))

        smooth_temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_top*smoothing_bottom *
                                             smoothing_fore*smoothing_aft)

        # make a little region along the top of the cavity where we don't want
        # the temperature smoothed
        xc_left = zeros + 0.65163 + 0.0004
        xc_right = zeros + 0.72163 - 0.0004
        yc_top = zeros - 0.006
        yc_bottom = zeros - 0.01

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        inside_block = left_edge*right_edge*top_edge*bottom_edge
        temperature = actx.np.where(inside_block, temperature, smooth_temperature)

        y = ones*self._mass_frac
        mass = eos.get_density(pressure=pressure, temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        velocity = np.zeros(self._dim, dtype=object)
        mom = mass*velocity
        cv = make_conserved(dim=self._dim, mass=mass, momentum=mom, energy=energy,
                            species_mass=mass*y)
        velocity[0] = mach*eos.sound_speed(cv, temperature)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma
        smoothing_top = smooth_step(actx, -sigma*(ypos-ytop))
        smoothing_bottom = smooth_step(actx, sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        if self._dim == 3:
            smoothing_fore = smooth_step(actx, sigma*(zpos-z0))
            smoothing_aft = smooth_step(actx, -sigma*(zpos-z1))
        velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom *
                       smoothing_fore*smoothing_aft)

        # split into x and y components
        velocity[1] = velocity[0]*actx.np.sin(theta)
        velocity[0] = velocity[0]*actx.np.cos(theta)

        # zero out the velocity in the cavity region, let the flow develop naturally
        # initially in pressure/temperature equilibrium with the exterior flow
        zeros = 0*xpos
        xc_left = zeros + 0.65163 - 0.000001
        #xc_right = zeros + 0.72163 + 0.000001
        xc_right = zeros + 0.726 + 0.000001
        yc_top = zeros - 0.0083245
        yc_bottom = zeros - 0.0283245
        xc_bottom = zeros + 0.70163
        wall_theta = np.sqrt(2)/2.

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        inside_cavity = left_edge*right_edge*top_edge

        # smooth the temperature at the cavity walls
        sigma = self._temp_sigma
        smoothing_front = smooth_step(actx, sigma*(xpos-xc_left))
        smoothing_bottom = smooth_step(actx, sigma*(ypos-yc_bottom))
        wall_dist = (wall_theta*(ypos - yc_bottom) -
                     wall_theta*(xpos - xc_bottom))
        smoothing_slant = smooth_step(actx, sigma*wall_dist)
        cavity_temperature = (wall_temperature +
            (temperature - wall_temperature) *
             smoothing_front*smoothing_bottom*smoothing_slant)
        temperature = actx.np.where(inside_cavity, cavity_temperature, temperature)

        # zero of the velocity
        velocity[0] = actx.np.where(inside_cavity, zeros, velocity[0])

        # fuel stream initialization
        # initially in pressure/temperature equilibrium with the cavity
        #inj_left = 0.71
        # even with the bottom corner
        inj_left = 0.70563
        # even with the top corner
        #inj_left = 0.7074
        #inj_left = 0.65
        inj_right = 0.73
        inj_top = -0.0226
        inj_bottom = -0.025
        inj_fore = 0.035/2. + 1.59e-3
        inj_aft = 0.035/2. - 1.59e-3
        xc_left = zeros + inj_left
        xc_right = zeros + inj_right
        yc_top = zeros + inj_top
        yc_bottom = zeros + inj_bottom
        zc_fore = zeros + inj_fore
        zc_aft = zeros + inj_aft

        yc_center = zeros - 0.0283245 + 4e-3 + 1.59e-3/2.
        zc_center = zeros + 0.035/2.
        inj_radius = 1.59e-3/2.

        if self._dim == 2:
            radius = actx.np.sqrt((ypos - yc_center)**2)
        else:
            radius = actx.np.sqrt((ypos - yc_center)**2 + (zpos - zc_center)**2)

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        top_edge = actx.np.less(ypos, yc_top)
        aft_edge = ones
        fore_edge = ones
        if self._dim == 3:
            aft_edge = actx.np.greater(zpos, zc_aft)
            fore_edge = actx.np.less(zpos, zc_fore)
        inside_injector = (left_edge*right_edge*top_edge*bottom_edge *
                           aft_edge*fore_edge)

        inj_y = ones*self._inj_mass_frac

        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_velocity[0] = self._inj_vel[0]

        inj_mach = self._inj_mach*ones

        # smooth out the injection profile
        # relax to the cavity temperature/pressure/velocity
        inj_x0 = 0.712
        inj_fuel_x0 = 0.712 - 0.002
        inj_sigma = 1500

        # left extent
        inj_tanh = inj_sigma*(inj_fuel_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        for i in range(self._nspecies):
            inj_y[i] = y[i] + (inj_y[i] - y[i])*inj_weight

        # transition the mach number from 0 (cavitiy) to 1 (injection)
        inj_tanh = inj_sigma*(inj_x0 - xpos)
        inj_weight = 0.5*(1.0 - actx.np.tanh(inj_tanh))
        inj_mach = inj_weight*inj_mach

        # assume a smooth transition in gamma, could calculate it
        inj_gamma = (self._gamma_guess +
            (self._inj_gamma_guess - self._gamma_guess)*inj_weight)

        inj_pressure = getIsentropicPressure(
            mach=inj_mach,
            P0=self._inj_P0,
            gamma=inj_gamma
        )
        inj_temperature = getIsentropicTemperature(
            mach=inj_mach,
            T0=self._inj_T0,
            gamma=inj_gamma
        )

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                      species_mass_fractions=inj_y)

        inj_velocity = mach*np.zeros(self._dim, dtype=object)
        inj_mom = inj_mass*inj_velocity

        # the velocity magnitude
        inj_cv = make_conserved(dim=self._dim, mass=inj_mass, momentum=inj_mom,
                                energy=inj_energy, species_mass=inj_mass*inj_y)

        inj_velocity[0] = -inj_mach*eos.sound_speed(inj_cv, inj_temperature)

        # relax the pressure at the cavity/injector interface
        inj_pressure = pressure + (inj_pressure - pressure)*inj_weight
        inj_temperature = (unsmoothed_temperature +
            (inj_temperature - unsmoothed_temperature)*inj_weight)

        # we need to calculate the velocity from a prescribed mass flow rate
        # this will need to take into account the velocity relaxation at the
        # injector walls
        #inj_velocity[0] = velocity[0] + (self._inj_vel[0] - velocity[0])*inj_weight

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma_injection
        wall_temperature = self._temp_wall
        smoothing_radius = smooth_step(
            actx, -sigma*(actx.np.abs(radius - inj_radius)))
        inj_temperature = (wall_temperature +
            (inj_temperature - wall_temperature)*smoothing_radius)

        inj_mass = eos.get_density(pressure=inj_pressure,
                                   temperature=inj_temperature,
                                   species_mass_fractions=inj_y)
        inj_energy = inj_mass*eos.get_internal_energy(temperature=inj_temperature,
                                                  species_mass_fractions=inj_y)

        # modify the velocity in the near-wall region to have a smooth profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma_injection
        smoothing_radius = smooth_step(
            actx, sigma*(actx.np.abs(radius - inj_radius)))
        inj_velocity[0] = inj_velocity[0]*smoothing_radius

        # use the species field with fuel added everywhere
        for i in range(self._nspecies):
            y[i] = actx.np.where(inside_injector, inj_y[i], y[i])

        # recompute the mass and energy (outside the injector) to account for
        # the change in mass fraction
        mass = eos.get_density(pressure=pressure,
                               temperature=temperature,
                               species_mass_fractions=y)
        energy = mass*eos.get_internal_energy(temperature=temperature,
                                              species_mass_fractions=y)

        mass = actx.np.where(inside_injector, inj_mass, mass)
        velocity[0] = actx.np.where(inside_injector, inj_velocity[0], velocity[0])
        energy = actx.np.where(inside_injector, inj_energy, energy)

        mom = mass*velocity
        energy = (energy + np.dot(mom, mom)/(2.0*mass))
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy,
            species_mass=mass*y
        )


def mask_from_elements(vol_discr, actx, elements):
    mesh = vol_discr.mesh
    zeros = vol_discr.zeros(actx)

    group_arrays = []

    for igrp in range(len(mesh.groups)):
        start_elem_nr = mesh.base_element_nrs[igrp]
        end_elem_nr = start_elem_nr + mesh.groups[igrp].nelements
        grp_elems = elements[
            (elements >= start_elem_nr)
            & (elements < end_elem_nr)] - start_elem_nr
        grp_ary_np = actx.to_numpy(zeros[igrp])
        grp_ary_np[grp_elems] = 1
        group_arrays.append(actx.from_numpy(grp_ary_np))

    return DOFArray(actx, tuple(group_arrays))


@mpi_entry_point
def main(ctx_factory=cl.create_some_context,
         restart_filename=None, target_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    #log_path = "log_data/"
    log_path = ""
    logname = log_path + casename + ".sqlite"

    """
    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    """

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    if lazy:
        actx = actx_class(comm, queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1
    # verbosity for what gets written to viz dumps, increase for more stuff
    viz_level = 1
    # control the time interval for writing viz dumps
    viz_interval_type = 0

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    t_viz_interval = 1.e-8
    current_t = 0
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6
    health_temp_min = 1.0
    health_temp_max = 5000
    health_mass_frac_min = -10
    health_mass_frac_max = 10

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2
    mesh_filename = "data/actii_2d.msh"

    # material properties
    mu = 1.0e-5
    spec_diff = 1.e-4
    mu_override = False  # optionally read in from input
    nspecies = 0

    # ACTII flow properties
    total_pres_inflow = 2.745e5
    total_temp_inflow = 2076.43

    # injection flow properties
    total_pres_inj = 50400
    total_temp_inj = 300.0
    mach_inj = 1.0

    # parameters to adjust the shape of the initialization
    vel_sigma = 1000
    temp_sigma = 1250
    # adjusted to match the mass flow rate
    vel_sigma_inj = 5000
    temp_sigma_inj = 5000
    temp_wall = 300

    # wall stuff
    wall_penalty_amount = 25

    # initialize the ignition spark
    ignition = False
    spark_center = np.zeros(shape=(dim,))
    spark_center[0] = 0.677
    spark_center[1] = -0.021
    if dim == 3:
        spark_center[2] = 0.035/2.
    spark_diameter = 0.0025
    spark_strength = 40000000./current_dt
    spark_init_time = 999999999.
    spark_duration = 1.e-8

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        input_data = comm.bcast(input_data, root=0)
        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            t_viz_interval = float(input_data["t_viz_interval"])
        except KeyError:
            pass
        try:
            viz_interval_type = int(input_data["viz_interval_type"])
        except KeyError:
            pass
        try:
            viz_level = int(input_data["viz_level"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            spec_diff = float(input_data["spec_diff"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            total_pres_inj = float(input_data["total_pres_inj"])
        except KeyError:
            pass
        try:
            total_temp_inj = float(input_data["total_temp_inj"])
        except KeyError:
            pass
        try:
            mach_inj = float(input_data["mach_inj"])
        except KeyError:
            pass
        try:
            nspecies = int(input_data["nspecies"])
        except KeyError:
            pass
        try:
            vel_sigma = float(input_data["vel_sigma"])
        except KeyError:
            pass
        try:
            temp_sigma = float(input_data["temp_sigma"])
        except KeyError:
            pass
        try:
            vel_sigma_inj = float(input_data["vel_sigma_inj"])
        except KeyError:
            pass
        try:
            temp_sigma_inj = float(input_data["temp_sigma_inj"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_temp_min = float(input_data["health_temp_min"])
        except KeyError:
            pass
        try:
            health_temp_max = float(input_data["health_temp_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass
        try:
            ignition = bool(input_data["ignition"])
        except KeyError:
            pass
        try:
            spark_init_time = float(input_data["ignition_init_time"])
        except KeyError:
            pass
        try:
            mesh_filename = input_data["mesh_filename"]
        except KeyError:
            pass
        try:
            wall_penalty_amount = input_data["wall_penalty_amount"]
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    if viz_interval_type > 2:
        error_message = "Invalid value for viz_interval_type [0-2]"
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####\n")

    if rank == 0:
        print("\n#### Visualization setup: ####")
        if viz_level >= 0:
            print("\tBasic visualization output enabled.")
            print("\t(cv, dv, cfl)")
        if viz_level >= 1:
            print("\tExtra visualization output enabled for derived quantities.")
            print("\t(velocity, mass_fractions, etc.)")
        if viz_level >= 2:
            print("\tNon-dimensional parameter visualization output enabled.")
            print("\t(Re, Pr, etc.)")
        if viz_level >= 3:
            print("\tDebug visualization output enabled.")
            print("\t(rhs, grad_cv, etc.)")
        if viz_interval_type == 0:
            print(f"\tWriting viz data every {nviz} steps.")
        if viz_interval_type == 1:
            print(f"\tWriting viz data roughly every {t_viz_interval} seconds.")
        if viz_interval_type == 2:
            print(f"\tWriting viz data exactly every {t_viz_interval} seconds.")
        print("#### Visualization setup: ####")

    if rank == 0:
        print("\n#### Simluation setup data: ####")
        print(f"\ttotal_pres_injection = {total_pres_inj}")
        print(f"\ttotal_temp_injection = {total_temp_inj}")
        print(f"\tvel_sigma = {vel_sigma}")
        print(f"\ttemp_sigma = {temp_sigma}")
        print(f"\tvel_sigma_injection = {vel_sigma_inj}")
        print(f"\ttemp_sigma_injection = {temp_sigma_inj}")
        print("#### Simluation setup data: ####")

    if rank == 0 and ignition:
        print("\n#### Ignition control parameters ####")
        print(f"spark center ({spark_center[0]},{spark_center[1]})")
        print(f"spark FWHM {spark_diameter}")
        print(f"spark strength {spark_strength}")
        print(f"ignition time {spark_init_time}")
        print(f"ignition duration {spark_duration}")
        print("#### Ignition control parameters ####\n")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o2 = 15.999*2
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    mf_c2h4 = 0.5
    mf_h2 = 0.5
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    if mu_override:
        mu = mu_input
    else:
        mu = mu_mix

    # Trouble getting AV working with lazy+wall, so crank up viscosity
    # instead for now
    mu *= 100

    kappa = cp*mu/Pr
    init_temperature = 300.0

    # Averaging from https://www.azom.com/article.aspx?ArticleID=1630
    wall_insert_rho = 1625
    wall_insert_cp = 770
    wall_insert_kappa = 247.5

    # Averaging from http://www.matweb.com/search/datasheet.aspx?bassnum=MS0001
    wall_surround_rho = 7.9e3
    wall_surround_cp = 470
    wall_surround_kappa = 48

    wall_time_scale = 250

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")
        print(f"\tnspecies = {nspecies}")
        if nspecies == 0:
            print("\tno passive scalars, uniform ideal gas eos")
        elif nspecies == 2:
            print("\tpassive scalars to track air/fuel mixture, ideal gas eos")
        else:
            print("\tfull multi-species initialization with pyrometheus eos")
        print(f"\tWall density = {wall_insert_rho}")
        print(f"\tWall cp = {wall_insert_cp}")
        print(f"\tWall kappa = {wall_insert_kappa}")
        print(f"\tWall surround density = {wall_surround_rho}")
        print(f"\tWall surround cp = {wall_surround_cp}")
        print(f"\tWall surround kappa = {wall_surround_kappa}")
        print("#### Simluation material properties: ####")

    spec_diffusivity = spec_diff * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=spec_diffusivity)

    #
    # stagnation tempertuare 2076.43 K
    # stagnation pressure 2.745e5 Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=54e-3m) and
    # the throat (r=3.167e-3)
    #
    vel_inflow = np.zeros(shape=(dim,))
    vel_outflow = np.zeros(shape=(dim,))
    vel_injection = np.zeros(shape=(dim,))

    throat_height = 3.61909e-3
    inlet_height = 54.129e-3
    outlet_height = 28.54986e-3
    inlet_area_ratio = inlet_height/throat_height
    outlet_area_ratio = outlet_height/throat_height

    # initialize eos and species mass fractions
    y = np.zeros(nspecies)
    y_fuel = np.zeros(nspecies)
    if nspecies == 2:
        y[0] = 1
        y_fuel[1] = 1
        species_names = ["air", "fuel"]
    elif nspecies > 2:
        from mirgecom.mechanisms import get_mechanism_cti
        mech_cti = get_mechanism_cti("uiuc")

        cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
        cantera_nspecies = cantera_soln.n_species
        if nspecies != cantera_nspecies:
            if rank == 0:
                print(f"specified {nspecies=}, but cantera mechanism"
                      f" needs nspecies={cantera_nspecies}")
            raise RuntimeError()

        i_c2h4 = cantera_soln.species_index("C2H4")
        i_h2 = cantera_soln.species_index("H2")
        i_ox = cantera_soln.species_index("O2")
        i_di = cantera_soln.species_index("N2")
        # Set the species mass fractions to the free-stream flow
        y[i_ox] = mf_o2
        y[i_di] = 1. - mf_o2
        # Set the species mass fractions to the free-stream flow
        y_fuel[i_c2h4] = mf_c2h4
        y_fuel[i_h2] = mf_h2

        cantera_soln.TPY = init_temperature, 101325, y

    # make the eos
    if nspecies < 3:
        eos = IdealSingleGas(gamma=gamma, gas_const=r)
    else:
        from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
        pyro_mech = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
        eos = PyrometheusMixture(pyro_mech, temperature_guess=init_temperature)
        species_names = pyro_mech.species_names

    gas_model = GasModel(eos=eos, transport=transport_model)

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma,
                                      mach_guess=0.01)
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=total_pres_inflow,
                                        gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=total_temp_inflow,
                                           gamma=gamma)

    if nspecies < 3:
        rho_inflow = pres_inflow/temp_inflow/r
        sos = math.sqrt(gamma*pres_inflow/rho_inflow)
    else:
        cantera_soln.TPY = temp_inflow, pres_inflow, y
        rho_inflow = cantera_soln.density
        inlet_gamma = cantera_soln.cp_mass/cantera_soln.cv_mass

        gamma_error = (gamma - inlet_gamma)
        gamma_guess = inlet_gamma
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:

            inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                              gamma=gamma_guess,
                                              mach_guess=0.01)
            pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                                P0=total_pres_inflow,
                                                gamma=gamma_guess)
            temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                                   T0=total_temp_inflow,
                                                   gamma=gamma_guess)
            cantera_soln.TPY = temp_inflow, pres_inflow, y
            rho_inflow = cantera_soln.density
            inlet_gamma = cantera_soln.cp_mass/cantera_soln.cv_mass
            gamma_error = (gamma_guess - inlet_gamma)
            gamma_guess = inlet_gamma

        sos = math.sqrt(inlet_gamma*pres_inflow/rho_inflow)

    vel_inflow[0] = inlet_mach*sos

    if rank == 0:
        print("#### Simluation initialization data: ####")
        print(f"\tinlet Mach number {inlet_mach}")
        print(f"\tinlet gamma {inlet_gamma}")
        print(f"\tinlet temperature {temp_inflow}")
        print(f"\tinlet pressure {pres_inflow}")
        print(f"\tinlet rho {rho_inflow}")
        print(f"\tinlet velocity {vel_inflow[0]}")
        #print(f"final inlet pressure {pres_inflow_final}")

    outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                       gamma=gamma,
                                       mach_guess=1.1)
    pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                         P0=total_pres_inflow,
                                         gamma=gamma)
    temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                            T0=total_temp_inflow,
                                            gamma=gamma)

    if nspecies < 3:
        rho_outflow = pres_outflow/temp_outflow/r
        sos = math.sqrt(gamma*pres_outflow/rho_outflow)
    else:
        cantera_soln.TPY = temp_outflow, pres_outflow, y
        rho_outflow = cantera_soln.density
        outlet_gamma = cantera_soln.cp_mass/cantera_soln.cv_mass

        gamma_error = (gamma - outlet_gamma)
        gamma_guess = outlet_gamma
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:

            outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                              gamma=gamma_guess,
                                              mach_guess=0.01)
            pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                                P0=total_pres_inflow,
                                                gamma=gamma_guess)
            temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                   T0=total_temp_inflow,
                                                   gamma=gamma_guess)
            cantera_soln.TPY = temp_outflow, pres_outflow, y
            rho_outflow = cantera_soln.density
            outlet_gamma = cantera_soln.cp_mass/cantera_soln.cv_mass
            gamma_error = (gamma_guess - outlet_gamma)
            gamma_guess = outlet_gamma

    vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

    if rank == 0:
        print("\t********")
        print(f"\toutlet Mach number {outlet_mach}")
        print(f"\toutlet gamma {outlet_gamma}")
        print(f"\toutlet temperature {temp_outflow}")
        print(f"\toutlet pressure {pres_outflow}")
        print(f"\toutlet rho {rho_outflow}")
        print(f"\toutlet velocity {vel_outflow[0]}")

    # injection mach number
    if nspecies < 3:
        gamma_injection = gamma
    else:
        #MJA: Todo, get the gamma from cantera to get the correct inflow properties
        # needs to be iterative with the call below
        gamma_injection = 0.5*(1.24 + 1.4)

    pres_injection = getIsentropicPressure(mach=mach_inj,
                                           P0=total_pres_inj,
                                           gamma=gamma_injection)
    temp_injection = getIsentropicTemperature(mach=mach_inj,
                                              T0=total_temp_inj,
                                              gamma=gamma_injection)

    if nspecies < 3:
        rho_injection = pres_injection/temp_injection/r
        sos = math.sqrt(gamma*pres_injection/rho_injection)
    else:
        cantera_soln.TPY = temp_outflow, pres_outflow, y_fuel
        rho_injection = cantera_soln.density
        gamma_injection = cantera_soln.cp_mass/cantera_soln.cv_mass

        gamma_error = (gamma - gamma_injection)
        gamma_guess = gamma_injection
        toler = 1.e-6
        # iterate over the gamma/mach since gamma = gamma(T)
        while gamma_error > toler:

            outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                              gamma=gamma_guess,
                                              mach_guess=0.01)
            pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                                P0=total_pres_inj,
                                                gamma=gamma_guess)
            temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                                   T0=total_temp_inj,
                                                   gamma=gamma_guess)
            cantera_soln.TPY = temp_outflow, pres_outflow, y_fuel
            rho_outflow = cantera_soln.density
            outlet_gamma = cantera_soln.cp_mass/cantera_soln.cv_mass
            gamma_error = (gamma_guess - gamma_injection)
            gamma_guess = gamma_injection

        sos = math.sqrt(gamma_injection*pres_injection/rho_injection)

    vel_injection[0] = -mach_inj*sos

    if rank == 0:
        print("\t********")
        print(f"\tinjector Mach number {mach_inj}")
        print(f"\tinjector gamma {gamma_injection}")
        print(f"\tinjector temperature {temp_injection}")
        print(f"\tinjector pressure {pres_injection}")
        print(f"\tinjector rho {rho_injection}")
        print(f"\tinjector velocity {vel_injection[0]}")
        print("#### Simluation initialization data: ####\n")

    # read geometry files
    geometry_bottom = None
    geometry_top = None
    if rank == 0:
        from numpy import loadtxt
        geometry_bottom = loadtxt("nozzleBottom.dat", comments="#", unpack=False)
        geometry_top = loadtxt("nozzleTop.dat", comments="#", unpack=False)
    geometry_bottom = comm.bcast(geometry_bottom, root=0)
    geometry_top = comm.bcast(geometry_top, root=0)

    inj_ymin = -0.0243245
    inj_ymax = -0.0227345
    bulk_init = InitACTII(dim=dim,
                          geom_top=geometry_top, geom_bottom=geometry_bottom,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma, nspecies=nspecies,
                          mass_frac=y, gamma_guess=inlet_gamma,
                          inj_gamma_guess=gamma_injection,
                          inj_pres=total_pres_inj,
                          inj_temp=total_temp_inj,
                          inj_vel=vel_injection, inj_mass_frac=y_fuel,
                          inj_temp_sigma=temp_sigma_inj,
                          inj_vel_sigma=vel_sigma_inj,
                          inj_ytop=inj_ymax, inj_ybottom=inj_ymin,
                          inj_mach=mach_inj)

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )

    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "fluid": ["fluid"],
                "wall": ["wall_insert", "wall_surround"]}
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

    local_nelements = (
        volume_to_local_mesh_data["fluid"][0].nelements
        + volume_to_local_mesh_data["wall"][0].nelements)

    # target data, used for sponge and prescribed boundary condtitions
    if target_filename:  # read the grid from restart data
        target_filename = f"{target_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        target_data = read_restart_data(actx, target_filename)
        #volume_to_local_mesh_data = target_data["volume_to_local_mesh_data"]
        global_nelements = target_data["global_nelements"]
        target_order = int(target_data["order"])

        assert target_data["nparts"] == nparts
        assert target_data["nspecies"] == nspecies
        assert target_data["global_nelements"] == global_nelements
    else:
        logger.warning("No target file specied, using restart as target")

    if rank == 0:
        logger.info("Making discretization")

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
          default_simplex_group_factory, QuadratureSimplexGroupFactory

    discr = make_discretization_collection(
        actx,
        volumes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1)
        },
        _result_type=EagerDGDiscretization)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    dd_vol_fluid = DOFDesc(VolumeDomainTag("fluid"), DISCR_TAG_BASE)
    dd_vol_wall = DOFDesc(VolumeDomainTag("wall"), DISCR_TAG_BASE)

    wall_vol_discr = discr.discr_from_dd(dd_vol_wall)
    wall_tag_to_elements = volume_to_local_mesh_data["wall"][1]
    wall_insert_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_insert"])
    wall_surround_mask = mask_from_elements(
        wall_vol_discr, actx, wall_tag_to_elements["wall_surround"])

    if rank == 0:
        logger.info("Before restart/init")

    def get_fluid_state(cv, temperature_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed)

    create_fluid_state = actx.compile(get_fluid_state)

    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")
        temperature_seed = restart_data["temperature_seed"]
        if restart_order != order:
            restart_discr = make_discretization_collection(
                actx,
                volumes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=restart_order,
                _result_type=EagerDGDiscretization)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd(dd_vol_fluid),
                restart_discr.discr_from_dd(dd_vol_fluid)
            )
            wall_connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd(dd_vol_wall),
                restart_discr.discr_from_dd(dd_vol_wall)
            )
            current_cv = fluid_connection(restart_data["cv"])
            temperature_seed = fluid_connection(restart_data["temperature_seed"])
            current_wall_temperature = wall_connection(
                restart_data["wall_temperature"])
        else:
            current_cv = restart_data["cv"]
            current_wall_temperature = restart_data["wall_temperature"]
        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logger.info("Initializing soln.")
        current_cv = bulk_init(
            discr=discr, x_vec=thaw(discr.nodes(dd_vol_fluid), actx), eos=eos,
            time=0)
        current_wall_temperature = temp_wall * (0*wall_insert_mask + 1)
        temperature_seed = init_temperature

    if target_filename:
        if rank == 0:
            logger.info("Reading target soln.")
        if target_order != order:
            target_discr = make_discretization_collection(
                actx,
                volumes={
                    vol: mesh
                    for vol, (mesh, _) in volume_to_local_mesh_data.items()},
                order=target_order,
                _result_type=EagerDGDiscretization)
            from meshmode.discretization.connection import make_same_mesh_connection
            fluid_connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd(dd_vol_fluid),
                target_discr.discr_from_dd(dd_vol_fluid)
            )
            wall_connection = make_same_mesh_connection(
                actx,
                discr.discr_from_dd(dd_vol_wall),
                target_discr.discr_from_dd(dd_vol_wall)
            )
            target_cv = fluid_connection(target_data["cv"])
        else:
            target_cv = target_data["cv"]

    else:
        # Set the current state from time 0
        target_cv = current_cv

    # MJA: check this
    current_fluid_state = create_fluid_state(current_cv, temperature_seed)
    target_fluid_state = create_fluid_state(target_cv, temperature_seed)
    temperature_seed = current_fluid_state.temperature

    stepper_state = make_obj_array([current_cv, current_wall_temperature,
                                    temperature_seed])

    def _ref_state_func(discr, dd_bdry, gas_model, ref_state, **kwargs):
        from mirgecom.gas_model import project_fluid_state
        dd_vol_fluid = dd_bdry.with_domain_tag(
            VolumeDomainTag(dd_bdry.domain_tag.volume_tag))
        return project_fluid_state(
            discr, dd_vol_fluid,
            #as_dofdesc(dd_bdry).with_discr_tag(quadrature_tag),
            dd_bdry,
            ref_state, gas_model)

    _ref_boundary_state_func = partial(_ref_state_func, ref_state=target_fluid_state)

    ref_state = PrescribedFluidBoundary(boundary_state_func=_ref_boundary_state_func)
    isothermal_wall = IsothermalWallBoundary(temp_wall)
    wall_farfield = DirichletDiffusionBoundary(temp_wall)

    fluid_boundaries = {
        dd_vol_fluid.trace("inflow").domain_tag: ref_state,
        dd_vol_fluid.trace("outflow").domain_tag: ref_state,
        dd_vol_fluid.trace("injection").domain_tag: ref_state,
        dd_vol_fluid.trace("isothermal_wall").domain_tag: isothermal_wall,
    }

    wall_boundaries = {
        dd_vol_wall.trace("wall_farfield").domain_tag: wall_farfield
    }

    # if you divide by 2.355, 50% of the spark is within this diameter
    spark_diameter /= 2.355
    # if you divide by 6, 99% of the energy is deposited in this time
    spark_duration /= 6.0697

    # gaussian application in time
    def spark_time_func(t):
        expterm = actx.np.exp((-(t - spark_init_time)**2) /
                              (2*spark_duration*spark_duration))
        return expterm

    #spark_strength = 0.0
    ignition_source = SparkSource(dim=dim, center=spark_center,
                                  amplitude=spark_strength,
                                  amplitude_func=spark_time_func,
                                  width=spark_diameter)

    # initialize the sponge field
    sponge_thickness = 0.09
    sponge_amp = 1.0/current_dt/1000
    sponge_x0 = 0.9

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    x_vec = thaw(discr.nodes(dd_vol_fluid), actx)
    sponge_sigma = sponge_init(x_vec=x_vec)

    wall_model = WallModel(
        density=(
            wall_insert_rho * wall_insert_mask
            + wall_surround_rho * wall_surround_mask),
        heat_capacity=(
            wall_insert_cp * wall_insert_mask
            + wall_surround_cp * wall_surround_mask),
        thermal_conductivity=(
            wall_insert_kappa * wall_insert_mask
            + wall_surround_kappa * wall_surround_mask))

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    fluid_visualizer = make_visualizer(discr, volume_dd=dd_vol_fluid)
    wall_visualizer = make_visualizer(discr, volume_dd=dd_vol_wall)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    # some utility functions
    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(discr, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(discr, "vol", x))[()]

    def vol_min(x):
        return actx.to_numpy(nodal_min(discr, "vol", x))[()]

    def vol_max(x):
        return actx.to_numpy(nodal_max(discr, "vol", x))[()]

    def my_write_status(cv, dv, wall_temperature, dt, cfl):
        # MJA TODO: Add status for wall CFL
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        temperature = thaw(freeze(dv.temperature, actx), actx)
        pressure = thaw(freeze(dv.pressure, actx), actx)
        wall_temp = thaw(freeze(wall_temperature, actx), actx)

        pmin = vol_min(pressure)
        pmax = vol_max(pressure)
        tmin = vol_min(temperature)
        tmax = vol_max(temperature)
        twmin = vol_min(wall_temp)
        twmax = vol_max(wall_temp)

        from pytools.obj_array import obj_array_vectorize
        y_min = obj_array_vectorize(lambda x: vol_min(x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(x),
                                      cv.species_mass_fractions)

        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        dv_status_msg += (
            f"\n-------- T_fluid (min, max) (K)  = ({tmin:7g}, {tmax:7g})")
        dv_status_msg += (
            f"\n-------- T_wall (min, max) (K)  = ({twmin:7g}, {twmax:7g})")
        for i in range(nspecies):
            dv_status_msg += (
                f"\n-------- y_{species_names[i]} (min, max) = "
                f"({y_min[i]:1.3e}, {y_max[i]:1.3e})")
        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def _grad_t_operator(t, fluid_state, wall_temperature):
        fluid_grad_t, wall_grad_t = coupled_grad_t_operator(
            discr,
            gas_model, wall_model,
            dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_temperature,
            time=t,
            quadrature_tag=quadrature_tag)
        return make_obj_array([fluid_grad_t, wall_grad_t])

    grad_t_operator = actx.compile(_grad_t_operator)

    def my_write_viz(step, t, fluid_state, wall_temperature, ts_field, alpha_field):

        if rank == 0:
            print(f"******** Writing Visualization File at {step}, "
                  f"sim time {t:1.6e} s ********")

        cv = fluid_state.cv
        dv = fluid_state.dv

        # basic viz quantities, things here are difficult (or impossible) to compute
        # in post-processing
        fluid_viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        wall_viz_fields = [
            ("temperature", wall_temperature)]

        # extra viz quantities, things here are often used for post-processing
        if viz_level > 0:
            mach = cv.speed / dv.speed_of_sound
            tagged_cells = smoothness_indicator(discr, cv.mass, s0=s0_sc,
                                                kappa=kappa_sc,
                                                volume_dd=dd_vol_fluid)

            fluid_viz_ext = [("mach", mach),
                       ("velocity", cv.velocity),
                       ("alpha", alpha_field),
                       ("tagged_cells", tagged_cells)]
            fluid_viz_fields.extend(fluid_viz_ext)
            # species mass fractions
            fluid_viz_fields.extend(
                ("Y_"+species_names[i], cv.species_mass_fractions[i])
                for i in range(nspecies))

        # additional viz quantities, add in some non-dimensional numbers
        if viz_level > 1:
            from grudge.dt_utils import characteristic_lengthscales
            char_length = characteristic_lengthscales(cv.array_context, discr,
                                                      dd=dd_vol_fluid)
            cell_Re = cv.mass*cv.speed*char_length/fluid_state.viscosity
            cp = gas_model.eos.heat_capacity_cp(cv, fluid_state.temperature)
            alpha_heat = fluid_state.thermal_conductivity/cp/fluid_state.viscosity
            cell_Pe_heat = char_length*cv.speed/alpha_heat
            from mirgecom.viscous import get_local_max_species_diffusivity
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )
            cell_Pe_mass = char_length*cv.speed/d_alpha_max
            # these are useful if our transport properties
            # are not constant on the mesh
            # prandtl
            # schmidt_number
            # damkohler_number

            viz_ext = [("Re", cell_Re),
                       ("Pe_mass", cell_Pe_mass),
                       ("Pe_heat", cell_Pe_heat)]
            fluid_viz_fields.extend(viz_ext)

        # debbuging viz quantities, things here are used for diagnosing run issues
        if viz_level > 2:
            """
            from mirgecom.fluid import (
                velocity_gradient,
                species_mass_fraction_gradient
            )
            ns_rhs, grad_cv, grad_t = \
                ns_operator(discr, state=fluid_state, time=t,
                            boundaries=boundaries, gas_model=gas_model,
                            return_gradients=True)
            grad_v = velocity_gradient(cv, grad_cv)
            grad_y = species_mass_fraction_gradient(cv, grad_cv)
            """

            grad_temperature = grad_t_operator(t, fluid_state, wall_temperature)
            fluid_grad_temperature = grad_temperature[0]
            wall_grad_temperature = grad_temperature[1]

            viz_ext = [("grad_temperature", fluid_grad_temperature)]
            """
            viz_ext = [("rhs", ns_rhs),
                       ("grad_temperature", fluid_grad_temperature),
                       ("grad_v_x", grad_v[0]),
                       ("grad_v_y", grad_v[1])]
            if dim == 3:
                viz_ext.extend(("grad_v_z", grad_v[2]))

            viz_ext.extend(("grad_Y_"+species_names[i], grad_y[i])
                           for i in range(nspecies))
            fluid_viz_fields.extend(viz_ext)
            """

            viz_ext = [("grad_temperature", wall_grad_temperature)]
            wall_viz_fields.extend(viz_ext)

        write_visfile(
            discr, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t, overwrite=True)
        write_visfile(
            discr, wall_viz_fields, wall_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True)

        if rank == 0:
            print("******** Done Writing Visualization File ********\n")

    def my_write_restart(step, t, state):
        if rank == 0:
            print(f"******** Writing Restart File at {step=}, "
                  f"sim time {t:1.6e} s ********")

        cv, wall_temperature, tseed = state
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "wall_temperature": wall_temperature,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

        if rank == 0:
            print("******** Done Writing Restart File ********\n")

    def my_health_check(cv, dv):
        # FIXME: Add health check for wall temperature?
        health_error = False
        if check_naninf_local(discr, dd_vol_fluid, dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(discr, dd_vol_fluid, dv.pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = vol_min(dv.pressure)
            p_max = vol_max(dv.pressure)
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        if global_reduce(check_range_local(discr, "vol", dv.temperature,
                                     health_temp_min, health_temp_max),
                                     op="lor"):
            health_error = True
            t_min = vol_min(dv.temperature)
            t_max = vol_max(dv.temperature)
            logger.info(f"Temperature range violation ({t_min=}, {t_max=})")

        for i in range(nspecies):
            if global_reduce(check_range_local(discr, "vol",
                                               cv.species_mass_fractions[i],
                                         health_mass_frac_min, health_mass_frac_max),
                                         op="lor"):
                health_error = True
                y_min = vol_min(cv.species_mass_fractions[i])
                y_max = vol_max(cv.species_mass_fractions[i])
                logger.info(f"Species mass fraction range violation. "
                            f"{species_names[i]}: ({y_min=}, {y_max=})")

        return health_error

    def my_get_viscous_timestep(discr, fluid_state, alpha):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        discr: grudge.eager.EagerDGDiscretization
            the discretization to use
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.dof_array.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        from grudge.dt_utils import characteristic_lengthscales

        length_scales = characteristic_lengthscales(
            fluid_state.array_context, discr, dd=dd_vol_fluid)

        nu = 0
        d_alpha_max = 0

        if fluid_state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            nu = fluid_state.viscosity/fluid_state.mass_density
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    fluid_state.array_context,
                    fluid_state.species_diffusivity
                )

        return(
            length_scales / (fluid_state.wavespeed
            + ((nu + d_alpha_max + alpha) / length_scales))
        )

    def my_get_viscous_cfl(discr, dt, fluid_state, alpha):
        """Calculate and return node-local CFL based on current state and timestep.

        Parameters
        ----------
        discr: :class:`grudge.eager.EagerDGDiscretization`
            the discretization to use
        dt: float or :class:`~meshmode.dof_array.DOFArray`
            A constant scalar dt or node-local dt
        fluid_state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.dof_array.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The CFL at each node.
        """
        return dt / my_get_viscous_timestep(
            discr, fluid_state=fluid_state, alpha=alpha)

    def my_get_timestep(t, dt, fluid_state, alpha):
        # FIXME: Take into account wall timestep restriction
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * my_get_viscous_timestep(
                discr, fluid_state=fluid_state, alpha=alpha)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(discr, dd_vol_fluid, ts_field))
            cfl = current_cfl
        else:
            ts_field = my_get_viscous_cfl(
                discr, dt=dt, fluid_state=fluid_state, alpha=alpha)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(discr, dd_vol_fluid, ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_get_alpha(discr, fluid_state, alpha):
        """ Scale alpha by the element characteristic length """
        from grudge.dt_utils import characteristic_lengthscales
        array_context = fluid_state.array_context
        length_scales = characteristic_lengthscales(
            array_context, discr, dd=dd_vol_fluid)

        #from mirgecom.fluid import compute_wavespeed
        #wavespeed = compute_wavespeed(eos, fluid_state)

        vmag = array_context.np.sqrt(
            np.dot(fluid_state.velocity, fluid_state.velocity))
        #alpha_field = alpha*wavespeed*length_scales
        alpha_field = alpha*vmag*length_scales
        #alpha_field = wavespeed*0 + alpha*current_step
        #alpha_field = state.mass

        return alpha_field

    def my_pre_step(step, t, dt, state):
        cv, wall_temperature, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state, alpha_field)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl=cfl, cv=cv, dv=dv,
                                wall_temperature=wall_temperature)

            if do_restart:
                my_write_restart(step=step, t=t, state=state)

            if do_viz:
                my_write_viz(
                    step=step, t=t, fluid_state=fluid_state,
                    wall_temperature=wall_temperature, ts_field=ts_field,
                    alpha_field=alpha_field)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(
                step=step, t=t, fluid_state=fluid_state,
                wall_temperature=wall_temperature, ts_field=ts_field,
                alpha_field=alpha_field)
            my_write_restart(step=step, t=t, state=state)
            raise

        dt = get_sim_timestep(
            discr, fluid_state, t, dt, current_cfl, t_final, constant_cfl,
            fluid_volume_dd=dd_vol_fluid)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state[0], eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        cv, wall_temperature, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        alpha_field = my_get_alpha(discr, fluid_state, alpha_sc)
        fluid_rhs, wall_rhs = coupled_ns_heat_operator(
            discr,
            gas_model, wall_model,
            dd_vol_fluid, dd_vol_wall,
            fluid_boundaries, wall_boundaries,
            fluid_state, wall_temperature,
            time=t,
            use_av=True,
            av_kwargs={
                "alpha": alpha_field,
                "s0": s0_sc,
                "kappa": kappa_sc,
                "boundary_kwargs": {
                    "time": t,
                    "gas_model": gas_model}},
            wall_time_scale=wall_time_scale, wall_penalty_amount=wall_penalty_amount,
            quadrature_tag=quadrature_tag)

        fluid_rhs += sponge_source(cv=fluid_state.cv, cv_ref=target_cv,
                         sigma=sponge_sigma)

        if nspecies > 3:
            fluid_rhs += eos.get_species_source_terms(cv,
                             temperature=fluid_state.temperature)

        if ignition:
            fluid_rhs += ignition_source(x_vec=x_vec, cv=cv, time=t)

        tseed_rhs = fluid_state.temperature - temperature_seed
        return make_obj_array([fluid_rhs, wall_rhs, tseed_rhs])

    current_dt = get_sim_timestep(
        discr, current_fluid_state, current_t, current_dt, current_cfl, t_final,
        constant_cfl, fluid_volume_dd=dd_vol_fluid)

    current_step, current_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      #state=current_state)
                      state=stepper_state)
    current_cv, current_wall_temperature, tseed = stepper_state
    current_fluid_state = create_fluid_state(current_cv, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_fluid_state.dv
    alpha_field = my_get_alpha(discr, current_fluid_state, alpha_sc)
    ts_field, cfl, dt = my_get_timestep(
        t=current_t, dt=current_dt, fluid_state=current_fluid_state,
        alpha=alpha_field)
    my_write_status(dt=dt, cfl=cfl, dv=final_dv, cv=current_cv,
                    wall_temperature=current_wall_temperature)

    my_write_viz(
        step=current_step, t=current_t, fluid_state=current_fluid_state,
        wall_temperature=current_wall_temperature, ts_field=ts_field,
        alpha_field=alpha_field)
    my_write_restart(step=current_step, t=current_t, state=stepper_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    #root_logger = logging.getLogger()

    #logging.debug("A DEBUG message")
    #logging.info("An INFO message")
    #logging.warning("A WARNING message")
    #logging.error("An ERROR message")
    #logging.critical("A CRITICAL message")

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-t", "--target_file", type=ascii, dest="target_file",
                        nargs="?", action="store", help="simulation target file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "isolator"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    target_filename = None
    if args.target_file:
        target_filename = (args.target_file).replace("'", "")
        print(f"Target file specified: {target_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, target_filename=target_filename,
         user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
