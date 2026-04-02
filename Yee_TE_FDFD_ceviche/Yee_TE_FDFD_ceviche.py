import numpy as np
from numpy import sqrt
import autograd.numpy as npa
import scipy.sparse as sp

import scipy.sparse.linalg as spl

import time

import autograd as ag
from autograd.numpy.numpy_boxes import ArrayBox

import copy
import matplotlib.pylab as plt
from autograd.extend import primitive, vspace, defvjp, defjvp


from copy import copy, deepcopy

from autograd.core import make_vjp, make_jvp
from autograd.wrap_util import unary_to_nary

from numpy.fft import fft, fftfreq



name = "ceviche_3D"

__version__ = '0.1.3'



"""
This file contains constants that are used throghout the codebase
"""

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental chargeimport numpy as np


"""
This file contains functions related to performing derivative operations used in the simulation tools.
-  The FDTD method requires autograd-compatible curl operations, which are performed using numpy.roll
-  The FDFD method requires sparse derivative matrices, with PML added, which are constructed here.
"""


"""================================== CURLS FOR FDTD ======================================"""

def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        return (npa.roll(Ez, shift=-1, axis=1) - Ez) / dL - (npa.roll(Ey, shift=-1, axis=2) - Ey) / dL
    elif axis == 1:
        return (npa.roll(Ex, shift=-1, axis=2) - Ex) / dL - (npa.roll(Ez, shift=-1, axis=0) - Ez) / dL
    elif axis == 2:
        return (npa.roll(Ey, shift=-1, axis=0) - Ey) / dL - (npa.roll(Ex, shift=-1, axis=1) - Ex) / dL

def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        return (Hz - npa.roll(Hz, shift=1, axis=1)) / dL - (Hy - npa.roll(Hy, shift=1, axis=2)) / dL
    elif axis == 1:
        return (Hx - npa.roll(Hx, shift=1, axis=2)) / dL - (Hz - npa.roll(Hz, shift=1, axis=0)) / dL
    elif axis == 2:
        return (Hy - npa.roll(Hy, shift=1, axis=0)) / dL - (Hx - npa.roll(Hx, shift=1, axis=1)) / dL

"""======================= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ==========================="""

def compute_derivative_matrices(omega, shape, npml, dL, kx, ky, bloch_x=0.0, bloch_y=0.0, bloch_z=0.0):
    """ Returns sparse derivative matrices.  Currently works for 2D and 1D 
            omega: angular frequency (rad/sec)
            shape: shape of the FDFD grid
            npml: list of number of PML cells in x and y.
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    # Construct derivate matrices without PML
    Dxf = createDws('x', 'f', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dxb = createDws('x', 'b', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dyf = createDws('y', 'f', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dyb = createDws('y', 'b', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dxfn = createDws('x', 'fn', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dxbn = createDws('x', 'bn', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dyfn = createDws('y', 'fn', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)
    Dybn = createDws('y', 'bn', shape, dL, kx, ky, bloch_x=bloch_x, bloch_y=bloch_y, bloch_z=bloch_z)



    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape, npml, dL)

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf)
    Dxb = Sxb.dot(Dxb)
    Dyf = Syf.dot(Dyf)
    Dyb = Syb.dot(Dyb)

    Dxfn = Sxf.dot(Dxfn)
    Dxbn = Sxb.dot(Dxbn)
    Dyfn = Syf.dot(Dyfn)
    Dybn = Syb.dot(Dybn)


    return make_Dij(Dxf,1,1), make_Dij(Dxf,0,1), make_Dij(Dxf,1,0), make_Dij(Dxb,1,1), make_Dij(Dxb,0,1), make_Dij(Dxb,1,0), make_Dij(Dyf,0,0), make_Dij(Dyf,0,1), make_Dij(Dyf,1,0), make_Dij(Dyb,0,0), make_Dij(Dyb,0,1), make_Dij(Dyb,1,0), Dxf, Dxb, Dyf, Dyb, Dxfn, Dyfn, Dxbn, Dybn

""" Derivative Matrices (no PML) """

def createDws(component, dir, shape, dL, kx, ky, bloch_x=0.0, bloch_y=0.0, bloch_z=0.0):
    """ creates the derivative matrices
            component: one of 'x' or 'y' for derivative in x or y direction
            dir: one of 'f' or 'b', whether to take forward or backward finite difference
            shape: shape of the FDFD grid
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    Nx, Ny = shape    

    # special case, a 1D problem
    if component == 'x' and Nx == 1:
        return sp.eye(Ny)
    if component == 'y' and Ny == 1:
        return sp.eye(Nx)

    # select a `make_D` function based on the component and direction
    component_dir = component + dir
    if component_dir == 'xf':
        return make_Dxf(dL, shape, kx, bloch_x=bloch_x)
    elif component_dir == 'xb':
        return make_Dxb(dL, shape, kx, bloch_x=bloch_x)
    elif component_dir == 'yf':
        return make_Dyf(dL, shape, ky, bloch_y=bloch_y)
    elif component_dir == 'yb':
        return make_Dyb(dL, shape, ky, bloch_y=bloch_y)
    elif component_dir == 'xfn':
        return make_Dxfn(dL, shape, kx, bloch_x=bloch_x)
    elif component_dir == 'xbn':
        return make_Dxbn(dL, shape, kx, bloch_x=bloch_x)
    elif component_dir == 'yfn':
        return make_Dyfn(dL, shape, ky, bloch_y=bloch_y)
    elif component_dir == 'ybn':
        return make_Dybn(dL, shape, ky, bloch_y=bloch_y)
    
    else:
        raise ValueError("component and direction {} and {} not recognized".format(component, dir))

def make_Dxf(dL, shape, kx, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxf = sp.diags([-1-1j*kx*dL, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=np.complex128)
    Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny))
    return Dxf

def make_Dxb(dL, shape, kx, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxb = sp.diags([1-1j*kx*dL, -1, -np.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=np.complex128)
    Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny))
    return Dxb

def make_Dyf(dL, shape, ky, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyf = sp.diags([-1-1j*ky*dL, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny), dtype=np.complex128)
    Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf)
    return Dyf

def make_Dyb(dL, shape, ky, bloch_y=0.0):
    """ Backward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = sp.diags([1-1j*ky*dL, -1, -np.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny), dtype=np.complex128)
    Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb)
    return Dyb

def make_Dxfn(dL, shape, kx, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxf = sp.diags([-1, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=np.complex128)
    Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny))
    return Dxf

def make_Dxbn(dL, shape, kx, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxb = sp.diags([1, -1, -np.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=np.complex128)
    Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny))
    return Dxb

def make_Dyfn(dL, shape, ky, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny), dtype=np.complex128)
    Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf)
    return Dyf

def make_Dybn(dL, shape, ky, bloch_y=0.0):
    """ Backward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = sp.diags([1, -1, -np.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny), dtype=np.complex128)
    Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb)
    return Dyb


def make_Dij(D,i,j):
    ij_mat = sp.coo_matrix((np.array([1]),(np.array([i]),np.array([j]))),shape=(2,2),dtype=npa.complex128)
    return sp.kron(ij_mat,D)
""" PML Functions """

def create_S_matrices(omega, shape, npml, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]
    Nx_pml, Ny_pml = npml    

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_3D = np.zeros(shape, dtype=np.complex128)
    Sx_b_3D = np.zeros(shape, dtype=np.complex128)
    Sy_f_3D = np.zeros(shape, dtype=np.complex128)
    Sy_b_3D = np.zeros(shape, dtype=np.complex128)

    # insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0, Ny):
        Sx_f_3D[:, i] = 1 / s_vector_x_f
        Sx_b_3D[:, i] = 1 / s_vector_x_b
    for i in range(0, Nx):
        Sy_f_3D[i, :] = 1 / s_vector_y_f
        Sy_b_3D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_3D.flatten()
    Sx_b_vec = Sx_b_3D.flatten()
    Sy_f_vec = Sy_f_3D.flatten()
    Sy_b_vec = Sy_b_3D.flatten()

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b

def create_sfactor(dir, omega, dL, N, N_pml):
    """ creates the S-factor cross section needed in the S-matrices """

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == 'f':
        return create_sfactor_f(omega, dL, N, N_pml, dw)
    elif dir == 'b':
        return create_sfactor_b(omega, dL, N, N_pml, dw)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))

def create_sfactor_f(omega, dL, N, N_pml, dw):
    """ S-factor profile for forward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array

def create_sfactor_b(omega, dL, N, N_pml, dw):
    """ S-factor profile for backward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)
# notataion is similar to that used in: https://www.jpier.org/ac_api/download.php?id=11092006

class fdfd():
    """ Base class for FDFD simulation """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        """ initialize with a given structure and source
                omega: angular frequency (rad/s)
                dL: grid cell size (m)
                eps_r: array containing relative permittivity
                npml: list of number of PML grid cells in [x, y]
                bloch_{x,y} phase difference across {x,y} boundaries for bloch periodic boundary conditions (default = 0 = periodic)
        """

        self.omega = omega
        self.dL = dL
        self.kx = kx
        self.ky = ky
        self.npml = npml

        self._setup_bloch_phases(bloch_phases)

        self.eps_r = eps_r

        self._setup_derivatives()

    """ what happens when you reassign the permittivity of the fdfd object """

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self._eps_r
    
    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self._save_shape(new_eps)
        self._eps_r = new_eps


    """ classes inherited from fdfd() must implement their own versions of these functions for `fdfd.solve()` to work """

    def _make_A(self, eps_r, ind=0):
        """ This method constucts the entries and indices into the system matrix """
        raise NotImplementedError("need to make a _make_A() method")

    def _solve_fn(self, entries_a, indices_a, source_vec):
        """ This method takes the system matrix and source and returns the x, y, and z field components """
        raise NotImplementedError("need to implement function to solve for field components")

    """ You call this to function to solve for the electromagnetic fields """

    def solve(self, source_z):
        """ Outward facing function (what gets called by user) that takes a source grid and returns the field components """

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a, ind = self._make_A(eps_vec)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(eps_vec, entries_a, indices_a, source_vec, ind)

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        try:
            Fz = self._vec_to_grid(Fz_vec)
        except:
            Fz = Fz_vec

        return Fx, Fy, Fz

    """ Utility functions for FDFD object """

    def _setup_derivatives(self):
        """ Makes the sparse derivative matrices and does some processing for ease of use """

        # Creates all of the operators needed for later
        derivs = compute_derivative_matrices(self.omega, self.shape, self.npml, self.dL, self.kx, self.ky, bloch_x=self.bloch_x, bloch_y=self.bloch_y, bloch_z=self.bloch_z)

        # stores the raw sparse matrices
        self.Dxf11, self.Dxf01, self.Dxf10, self.Dxb11, self.Dxb01, self.Dxb10, self.Dyf00, self.Dyf01, self.Dyf10, self.Dyb00, self.Dyb01, self.Dyb10, self.Dxf, self.Dxb, self.Dyf, self.Dyb, self.Dxfn, self.Dxbn, self.Dyfn, self.Dybn = derivs

        # store the entries and elements
        self.entries_Dxf11, self.indices_Dxf11 = get_entries_indices(self.Dxf11)
        self.entries_Dxf01, self.indices_Dxf01 = get_entries_indices(self.Dxf01)
        self.entries_Dxf10, self.indices_Dxf10 = get_entries_indices(self.Dxf10)
        self.entries_Dyf00, self.indices_Dyf00 = get_entries_indices(self.Dyf00)
        self.entries_Dyf01, self.indices_Dyf01 = get_entries_indices(self.Dyf01)
        self.entries_Dyf10, self.indices_Dyf10 = get_entries_indices(self.Dyf10)

        self.entries_Dxb11, self.indices_Dxb11 = get_entries_indices(self.Dxb11)
        self.entries_Dxb01, self.indices_Dxb01 = get_entries_indices(self.Dxb01)
        self.entries_Dxb10, self.indices_Dxb10 = get_entries_indices(self.Dxb10)
        self.entries_Dyb00, self.indices_Dyb00 = get_entries_indices(self.Dyb00)
        self.entries_Dyb01, self.indices_Dyb01 = get_entries_indices(self.Dyb01)
        self.entries_Dyb10, self.indices_Dyb10 = get_entries_indices(self.Dyb10)

        
        self.entries_Dxf, self.indices_Dxf = get_entries_indices(self.Dxf)
        self.entries_Dxb, self.indices_Dxb = get_entries_indices(self.Dxb)
        self.entries_Dyf, self.indices_Dyf = get_entries_indices(self.Dyf)
        self.entries_Dyb, self.indices_Dyb = get_entries_indices(self.Dyb)
        self.entries_Dxfn, self.indices_Dxfn = get_entries_indices(self.Dxfn)
        self.entries_Dxbn, self.indices_Dxbn = get_entries_indices(self.Dxbn)
        self.entries_Dyfn, self.indices_Dyfn = get_entries_indices(self.Dyfn)
        self.entries_Dybn, self.indices_Dybn = get_entries_indices(self.Dybn)





        # stores some convenience functions for multiplying derivative matrices by a vector `vec`
        self.sp_mult_Dxf = lambda vec: sp_mult(self.entries_Dxf, self.indices_Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sp_mult(self.entries_Dxb, self.indices_Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sp_mult(self.entries_Dyf, self.indices_Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sp_mult(self.entries_Dyb, self.indices_Dyb, vec)
        self.sp_mult_Dxfn = lambda vec: sp_mult(self.entries_Dxfn, self.indices_Dxfn, vec)
        self.sp_mult_Dxbn = lambda vec: sp_mult(self.entries_Dxbn, self.indices_Dxbn, vec)
        self.sp_mult_Dyfn = lambda vec: sp_mult(self.entries_Dyfn, self.indices_Dyfn, vec)
        self.sp_mult_Dybn = lambda vec: sp_mult(self.entries_Dybn, self.indices_Dybn, vec)


    def _setup_bloch_phases(self, bloch_phases):
        """ Saves the x y and z bloch phases based on list of them 'bloch_phases' """

        self.bloch_x = 0.0
        self.bloch_y = 0.0
        self.bloch_z = 0.0
        if bloch_phases is not None:
            self.bloch_x = bloch_phases[0]
            if len(bloch_phases) > 1:
                self.bloch_y = bloch_phases[1]
            if len(bloch_phases) > 2:
                self.bloch_z = bloch_phases[2]

    def _vec_to_grid(self, vec):
        """ converts a vector quantity into an array of the shape of the FDFD simulation """
        return npa.reshape(vec, self.shape)

    def _grid_to_vec(self, grid):
        """ converts a grid of the shape of the FDFD simulation to a flat vector """
        return grid.flatten()

    def _save_shape(self, grid):
        """ Sores the shape and size of `grid` array to the FDFD object """
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape
        self.N = self.Nx * self.Ny

    @staticmethod
    def _default_val(val, default_val=None):
        # not used yet
        return val if val is not None else default_val

    """ Field conversion functions for 2D.  Function names are self explanatory """

    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return  1 / 1j / self.omega / MU_0 * (self.sp_mult_Dxf(Ey_vec) - self.sp_mult_Dyf(Ex_vec))

    def _Ex_Ez_to_Hy(self, Ex_vec, Ez_vec):
        return  1 / 1j / self.omega / MU_0 * (-self.sp_mult_Dxf(Ez_vec))

    def _Ey_Ez_to_Hx(self, Ey_vec, Ez_vec):
        return  1 / 1j / self.omega / MU_0 * self.sp_mult_Dyf(Ez_vec)


    # addition of 1e-5 is for numerical stability when tracking gradients of eps_xx, and eps_yy -> 0
    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return  -1 / 1j / self.omega / EPSILON_0 / (eps_vec_zz + 1e-5) * (self.sp_mult_Dxb(Hy_vec) - self.sp_mult_Dyb(Hx_vec))

    def _Hx_Hz_to_Ey(self, Hx_vec, Hz_vec, eps_vec_yy):
        return  -1 / 1j / self.omega / EPSILON_0 / (eps_vec_yy + 1e-5) * (-self.sp_mult_Dxb(Hz_vec))

    def _Hy_Hz_to_Ex(self, Hy_vec, Hz_vec, eps_vec_xx):
        return  -1 / 1j / self.omega / EPSILON_0 / (eps_vec_xx + 1e-5) * self.sp_mult_Dyb(Hz_vec)



""" These are the fdfd classes that you'll actually want to use """

class fdfd_TM(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, kx, ky, eps_r, npml, bloch_phases=bloch_phases)

    def _make_A(self, eps_vec, ind=0):

        C = - 1 / MU_0 * self.Dxf.dot(self.Dxb) \
            - 1 / MU_0 * self.Dyf.dot(self.Dyb) 
        entries_c, indices_c = get_entries_indices(C)

        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec
        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):

        b_vec = 1j * self.omega * Jz_vec
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        return Ez_vec, Ez_vec, Ez_vec

class fdfd_TEx(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, kx, ky, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):

        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=-1))
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=-1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _grid_shift_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_shifted = npa.roll(eps_grid, axis=1, shift=1)
        eps_vec_shifted = self._grid_to_vec(eps_grid_shifted)
        return eps_vec_shifted

    def _make_A(self, eps_vec, ind=0):

        eps_vec2D = npa.hstack((eps_vec,eps_vec))
        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec2D
        indices_diag = npa.vstack((npa.arange(2*self.N), npa.arange(2*self.N)))
 
        entries_Dyc10 = npa.hstack((self.entries_Dyf10, self.entries_Dyb10))
        indices_Dyc10 = npa.hstack((self.indices_Dyf10, self.indices_Dyb10))
        entries_Dxc11 = npa.hstack((self.entries_Dxb11, self.entries_Dxf11))
        indices_Dxc11 = npa.hstack((self.indices_Dxb11, self.indices_Dxf11))

        entries_Dxc01 = npa.hstack((self.entries_Dxf01, self.entries_Dxb01))
        indices_Dxc01 = npa.hstack((self.indices_Dxf01, self.indices_Dxb01))
        entries_Dyc00 = npa.hstack((self.entries_Dyb00, self.entries_Dyf00))
        indices_Dyc00 = npa.hstack((self.indices_Dyb00, self.indices_Dyf00))

        entries_Dxc11Dxc11,   indices_Dxc11Dxc11 = spsp_mult(-entries_Dxc11/2., indices_Dxc11, entries_Dxc11/2., indices_Dxc11, 2*self.N)

        entries_Dyc00Dyc00,   indices_Dyc00Dyc00 = spsp_mult(entries_Dyc00/2., indices_Dyc00, -entries_Dyc00/2., indices_Dyc00, 2*self.N)

        entries_Dxb11Dyf10, indices_Dxb11Dyf10 = spsp_mult(entries_Dxc11/2., indices_Dxc11, entries_Dyc10/2., indices_Dyc10, 2*self.N)

        entries_Dyb00Dxf01, indices_Dyb00Dxf01 = spsp_mult(entries_Dyc00/2., indices_Dyc00, entries_Dxc01/2., indices_Dxc01, 2*self.N)

        entries_d = 1 / MU_0 * npa.hstack((entries_Dxc11Dxc11, entries_Dyc00Dyc00, entries_Dxb11Dyf10, entries_Dyb00Dxf01))
        indices_d = npa.hstack((indices_Dxc11Dxc11, indices_Dyc00Dyc00, indices_Dxb11Dyf10, indices_Dyb00Dxf01))


        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))

        return entries_a, indices_a, ind

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec, ind):

        for ii in range(ind):
            Mz_vec = self._grid_shift_2d(Mz_vec)
        Jz_vec = npa.hstack((Mz_vec,0*Mz_vec))
        b_vec = 1j * self.omega * Jz_vec
        E_vec = sp_solve(entries_a, indices_a, b_vec)
        return E_vec[:self.N], E_vec[self.N:2*self.N], ind


class fdfd_TEy(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, kx, ky, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):

        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=-1))
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=-1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _grid_shift_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_shifted = npa.roll(eps_grid, axis=1, shift=1)
        eps_vec_shifted = self._grid_to_vec(eps_grid_shifted)
        return eps_vec_shifted

    def _make_A(self, eps_vec, ind=0):

        eps_vec2D = npa.hstack((eps_vec,eps_vec))
        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec2D
        indices_diag = npa.vstack((npa.arange(2*self.N), npa.arange(2*self.N)))
 
        entries_Dyc10 = npa.hstack((self.entries_Dyf10, self.entries_Dyb10))
        indices_Dyc10 = npa.hstack((self.indices_Dyf10, self.indices_Dyb10))
        entries_Dxc11 = npa.hstack((self.entries_Dxb11, self.entries_Dxf11))
        indices_Dxc11 = npa.hstack((self.indices_Dxb11, self.indices_Dxf11))

        entries_Dxc01 = npa.hstack((self.entries_Dxf01, self.entries_Dxb01))
        indices_Dxc01 = npa.hstack((self.indices_Dxf01, self.indices_Dxb01))
        entries_Dyc00 = npa.hstack((self.entries_Dyb00, self.entries_Dyf00))
        indices_Dyc00 = npa.hstack((self.indices_Dyb00, self.indices_Dyf00))

        entries_Dxc11Dxc11,   indices_Dxc11Dxc11 = spsp_mult(-entries_Dxc11/2., indices_Dxc11, entries_Dxc11/2., indices_Dxc11, 2*self.N)

        entries_Dyc00Dyc00,   indices_Dyc00Dyc00 = spsp_mult(entries_Dyc00/2., indices_Dyc00, -entries_Dyc00/2., indices_Dyc00, 2*self.N)

        entries_Dxb11Dyf10, indices_Dxb11Dyf10 = spsp_mult(entries_Dxc11/2., indices_Dxc11, entries_Dyc10/2., indices_Dyc10, 2*self.N)

        entries_Dyb00Dxf01, indices_Dyb00Dxf01 = spsp_mult(entries_Dyc00/2., indices_Dyc00, entries_Dxc01/2., indices_Dxc01, 2*self.N)

        entries_d = 1 / MU_0 * npa.hstack((entries_Dxc11Dxc11, entries_Dyc00Dyc00, entries_Dxb11Dyf10, entries_Dyb00Dxf01))
        indices_d = npa.hstack((indices_Dxc11Dxc11, indices_Dyc00Dyc00, indices_Dxb11Dyf10, indices_Dyb00Dxf01))


        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))
        
        return entries_a, indices_a, ind

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec, ind):

        for ii in range(ind):
            Mz_vec = self._grid_shift_2d(Mz_vec)
        Jz_vec = npa.hstack((0*Mz_vec,Mz_vec))
        b_vec = 1j * self.omega * Jz_vec
        E_vec = sp_solve(entries_a, indices_a, b_vec)
        return E_vec[:self.N], E_vec[self.N:2*self.N], ind


class fdfd_TEx_vac(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, kx, ky, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):

        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=-1))
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=-1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _grid_shift_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_shifted = npa.roll(eps_grid, axis=1, shift=1)
        eps_vec_shifted = self._grid_to_vec(eps_grid_shifted)
        return eps_vec_shifted

    def _make_A(self, eps_vec, ind=0):

        eps_vec2D = npa.hstack((eps_vec,eps_vec))
        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec2D
        indices_diag = npa.vstack((npa.arange(2*self.N), npa.arange(2*self.N)))
 
        
        entries_Dxf11Dxb11,   indices_Dxf11Dxb11 = spsp_mult(-self.entries_Dxf11, self.indices_Dxf11, self.entries_Dxb11, self.indices_Dxb11, 2*self.N)

        entries_Dyf00Dyb00,   indices_Dyf00Dyb00 = spsp_mult(self.entries_Dyf00, self.indices_Dyf00, -self.entries_Dyb00, self.indices_Dyb00, 2*self.N)

        entries_Dxb11Dyf10, indices_Dxb11Dyf10 = spsp_mult(self.entries_Dxb11, self.indices_Dxb11, self.entries_Dyf10, self.indices_Dyf10, 2*self.N)

        entries_Dyb00Dxf01, indices_Dyb00Dxf01 = spsp_mult(self.entries_Dyb00, self.indices_Dyb00, self.entries_Dxf01, self.indices_Dxf01, 2*self.N)

        entries_d = 1 / MU_0 * npa.hstack((entries_Dxf11Dxb11, entries_Dyf00Dyb00, entries_Dxb11Dyf10, entries_Dyb00Dxf01))
        indices_d = npa.hstack((indices_Dxf11Dxb11, indices_Dyf00Dyb00, indices_Dxb11Dyf10, indices_Dyb00Dxf01))


        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))
        
        return entries_a, indices_a, ind

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec, ind):

        for ii in range(ind):
            Mz_vec = self._grid_shift_2d(Mz_vec)
        Jz_vec = npa.hstack((Mz_vec,0*Mz_vec))
        b_vec = 1j * self.omega * Jz_vec
        E_vec = sp_solve(entries_a, indices_a, b_vec)
        return E_vec[:self.N], E_vec[self.N:2*self.N], ind


class fdfd_TEy_vac(fdfd):
    """ FDFD class for linear Ez polarization """

    def __init__(self, omega, dL, kx, ky, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, kx, ky, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):

        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=-1))
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=-1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _grid_shift_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_shifted = npa.roll(eps_grid, axis=1, shift=1)
        eps_vec_shifted = self._grid_to_vec(eps_grid_shifted)
        return eps_vec_shifted

    def _make_A(self, eps_vec, ind=0):

        eps_vec2D = npa.hstack((eps_vec,eps_vec))
        # indices into the diagonal of a sparse matrix
        entries_diag = - EPSILON_0 * self.omega**2 * eps_vec2D
        indices_diag = npa.vstack((npa.arange(2*self.N), npa.arange(2*self.N)))
 
        
        entries_Dxf11Dxb11,   indices_Dxf11Dxb11 = spsp_mult(-self.entries_Dxf11, self.indices_Dxf11, self.entries_Dxb11, self.indices_Dxb11, 2*self.N)

        entries_Dyf00Dyb00,   indices_Dyf00Dyb00 = spsp_mult(self.entries_Dyf00, self.indices_Dyf00, -self.entries_Dyb00, self.indices_Dyb00, 2*self.N)

        entries_Dxb11Dyf10, indices_Dxb11Dyf10 = spsp_mult(self.entries_Dxb11, self.indices_Dxb11, self.entries_Dyf10, self.indices_Dyf10, 2*self.N)

        entries_Dyb00Dxf01, indices_Dyb00Dxf01 = spsp_mult(self.entries_Dyb00, self.indices_Dyb00, self.entries_Dxf01, self.indices_Dxf01, 2*self.N)

        entries_d = 1 / MU_0 * npa.hstack((entries_Dxf11Dxb11, entries_Dyf00Dyb00, entries_Dxb11Dyf10, entries_Dyb00Dxf01))
        indices_d = npa.hstack((indices_Dxf11Dxb11, indices_Dyf00Dyb00, indices_Dxb11Dyf10, indices_Dyb00Dxf01))


        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))
        
        return entries_a, indices_a, ind

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec, ind):

        for ii in range(ind):
            Mz_vec = self._grid_shift_2d(Mz_vec)
        Jz_vec = npa.hstack((0*Mz_vec,Mz_vec))
        b_vec = 1j * self.omega * Jz_vec
        E_vec = sp_solve(entries_a, indices_a, b_vec)
        return E_vec[:self.N], E_vec[self.N:2*self.N], ind



class fdfd_mf_ez(fdfd):
    """ FDFD class for multifrequency linear Ez polarization. New variables:
            omega_mod: angular frequency of modulation (rad/s)
            delta: array of shape (Nfreq, Nx, Ny) containing pointwise modulation depth for each modulation harmonic (1,...,Nfreq)
            phi: array of same shape as delta containing pointwise modulation phase for each modulation harmonic
            Nsb: number of numerical sidebands to consider when solving for fields. 
            This is not the same as the number of modulation frequencies Nfreq. For physically meaningful results, Nsb >= Nfreq. 
    """

    def __init__(self, omega, dL, eps_r, omega_mod, delta, phi, Nsb, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)
        self.omega_mod = omega_mod
        self.delta = delta
        self.phi = phi
        self.Nsb = Nsb

    def solve(self, source_z):
        """ Outward facing function (what gets called by user) that takes a source grid and returns the field components """
        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)
        Nfreq = npa.shape(self.delta)[0]
        delta_matrix = self.delta.reshape([Nfreq, npa.prod(self.shape)])
        phi_matrix = self.phi.reshape([Nfreq, npa.prod(self.shape)])
        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec, delta_matrix, phi_matrix)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(eps_vec, entries_a, indices_a, source_vec)

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    def _make_A(self, eps_vec, delta_matrix, phi_matrix):
        """ Builds the multi-frequency electromagnetic operator A in Ax = b """
        M = 2*self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb,self.Nsb+1)*self.omega_mod

        C = sp.kron(sp.eye(M), - 1 / MU_0 * self.Dxf.dot(self.Dxb) - 1 / MU_0 * self.Dyf.dot(self.Dyb))
        entries_c, indices_c = get_entries_indices(C)

        # diagonal entries representing static refractive index
        # this part is just a block diagonal version of the single frequency fdfd_ez
        entries_diag = - EPSILON_0 * npa.kron(W**2, eps_vec)
        indices_diag = npa.vstack((npa.arange(M*N), npa.arange(M*N)))

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        # off-diagonal entries representing dynamic modulation
        # this part couples different frequencies due to modulation
        # for a derivation of these entries, see Y. Shi, W. Shin, and S. Fan. Optica 3(11), 2016.
        Nfreq = npa.shape(delta_matrix)[0]
        for k in npa.arange(Nfreq):
            # super-diagonal entries (note the +1j phase)
            mod_p = - 0.5 * EPSILON_0 * delta_matrix[k,:] * npa.exp(1j*phi_matrix[k,:])
            entries_p = npa.kron(W[:-k-1]**2, mod_p)
            indices_p = npa.vstack((npa.arange((M-k-1)*N), npa.arange((k+1)*N, M*N)))
            entries_a = npa.hstack((entries_p, entries_a))
            indices_a = npa.hstack((indices_p,indices_a))
            # sub-diagonal entries (note the -1j phase)
            mod_m = - 0.5 * EPSILON_0 * delta_matrix[k,:] * npa.exp(-1j*phi_matrix[k,:]) 
            entries_m = npa.kron(W[k+1:]**2, mod_m)
            indices_m = npa.vstack((npa.arange((k+1)*N, M*N), npa.arange((M-k-1)*N)))
            entries_a = npa.hstack((entries_m, entries_a))
            indices_a = npa.hstack((indices_m,indices_a))

        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):
        """ Multi-frequency version of _solve_fn() defined in fdfd_ez """
        M = 2*self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb,self.Nsb+1)*self.omega_mod 
        P = sp.kron(sp.spdiags(W,[0],M,M), sp.eye(N))
        entries_p, indices_p = get_entries_indices(P)
        b_vec = 1j * sp_mult(entries_p,indices_p,Jz_vec)
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    def _Ez_to_Hx(self, Ez_vec):
        """ Multi-frequency version of _Ez_to_Hx() defined in fdfd """
        M = 2*self.Nsb + 1
        Winv = 1/(self.omega + npa.arange(-self.Nsb,self.Nsb+1)*self.omega_mod)
        Dyb_mf = sp.kron(sp.spdiags(Winv,[0],M,M), self.Dyb)
        entries_Dyb_mf, indices_Dyb_mf = get_entries_indices(Dyb_mf)
        return -1 / 1j / MU_0 * sp_mult(entries_Dyb_mf, indices_Dyb_mf, Ez_vec)

    def _Ez_to_Hy(self, Ez_vec):
        """ Multi-frequency version of _Ez_to_Hy() defined in fdfd """
        M = 2*self.Nsb + 1
        Winv = 1/(self.omega + npa.arange(-self.Nsb,self.Nsb+1)*self.omega_mod)
        Dxb_mf = sp.kron(sp.spdiags(Winv,[0],M,M), self.Dxb)
        entries_Dxb_mf, indices_Dxb_mf = get_entries_indices(Dxb_mf)
        return  1 / 1j / MU_0 * sp_mult(entries_Dxb_mf, indices_Dxb_mf, Ez_vec)

    def _Ez_to_Hx_Hy(self, Ez_vec):
        """ Multi-frequency version of _Ez_to_Hx_Hy() defined in fdfd """
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    def _vec_to_grid(self, vec):
        """ Multi-frequency version of _vec_to_grid() defined in fdfd """
        # grid shape has Nx*Ny cells per frequency sideband
        grid_shape = (2*self.Nsb + 1, self.Nx, self.Ny)
        return npa.reshape(vec, grid_shape)

class fdfd_3d(fdfd):
    """ 3D FDFD class (work in progress) """

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        raise NotImplementedError

    def _grid_average_3d(self, eps_vec):
        raise NotImplementedError

    def _make_A(self, eps_vec):

        # notation: C = [[C11, C12], [C21, C22]]
        C11 = -1 / MU_0 * self.Dyf.dot(self.Dyb)
        C22 = -1 / MU_0 * self.Dxf.dot(self.Dxb)
        C12 =  1 / MU_0 * self.Dyf.dot(self.Dxb)
        C21 =  1 / MU_0 * self.Dxf.dot(self.Dyb)

        # get entries and indices
        entries_c11, indices_c11 = get_entries_indices(C11)
        entries_c22, indices_c22 = get_entries_indices(C22)
        entries_c12, indices_c12 = get_entries_indices(C12)
        entries_c21, indices_c21 = get_entries_indices(C21)

        # shift the indices into each of the 4 quadrants
        indices_c22 += self.N       # shift into bottom right quadrant
        indices_c12[1,:] += self.N  # shift into top right quadrant
        indices_c21[0,:] += self.N  # shift into bottom left quadrant

        # get full matrix entries and indices
        entries_c = npa.hstack((entries_c11, entries_c12, entries_c21, entries_c22))
        indices_c = npa.hstack((indices_c11, indices_c12, indices_c21, indices_c22))

        # indices into the diagonal of a sparse matrix
        eps_vec_xx, eps_vec_yy, eps_vec_zz = self._grid_average_3d(eps_vec)
        entries_diag = - EPSILON_0 * self.omega**2 * npa.hstack((eps_vec_xx, eps_vec_yy))
        indices_diag = npa.vstack((npa.arange(2 * self.N), npa.arange(2 * self.N)))

        # put together the big A and return entries and indices
        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))
        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec):

        # convert the Mz current into Jx, Jy
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)
        Jx_vec, Jy_vec = self._Hz_to_Ex_Ey(Mz_vec, eps_vec_xx, eps_vec_yy)

        # lump the current sources together and solve for electric field
        source_J_vec = npa.hstack((Jx_vec, Jy_vec))
        E_vec = sp_solve(entries_a, indices_a, source_J_vec)

        # strip out the x and y components of E and find the Hz component
        Ex_vec = E_vec[:self.N]
        Ey_vec = E_vec[self.N:]
        Hz_vec = self._Ex_Ey_to_Hz(Ex_vec, Ey_vec)

        return Ex_vec, Ey_vec, Hz_vec



class fdtd():

    def __init__(self, eps_r, dL, npml):
        """ Makes an FDTD object
                eps_r: the relative permittivity (array > 1)
                    if eps_r.shape = 3, it holds a single permittivity
                    if eps_r.shape = 4, the last index is the batch index (running several simulations at once)
                dL: the grid size(s) (float/int or list of 3 floats/ints for dx, dy, dz)
                npml: the number of PML grids in each dimension (list of 3 ints)
        """

        # set the grid shape
        eps_r = reshape_to_ND(eps_r, N=3)
        self.Nx, self.Ny, self.Nz = self.grid_shape = eps_r.shape

        # set the attributes
        self.dL = dL
        self.npml = npml
        self.eps_r = eps_r

    def __repr__(self):
        return "FDTD(eps_r.shape={}, dL={}, NPML={})".format(self.grid_shape, self.dL, self.npml)

    def __str__(self):
        return "FDTD object:\n\tdomain size = {}\n\tdL = {}\n\tNPML = {}".format(self.grid_shape, self.dL, self.npml)

    @property
    def dL(self):
        """ Returns the grid size """
        return self.__dL

    @dL.setter
    def dL(self, new_dL):
        """ Resets the time step when dL is set. """
        self.__dL = new_dL
        self._set_time_step()

    @property
    def npml(self):
        """ Returns the pml grid size list """
        return self.__npml

    @npml.setter
    def npml(self, new_npml):
        """ Defines some attributes when npml is set. """
        self.__npml = new_npml
        self._compute_sigmas()

    @property
    def eps_r(self):
        """ Returns the relative permittivity grid """
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """ Defines some attributes when eps_r is set. """
        self.__eps_r = new_eps
        self.eps_xx, self.eps_yy, self.eps_zz = grid_center_to_xyz(self.__eps_r)
        self.eps_arr = self.__eps_r.flatten()
        self.N = self.eps_arr.size
        self.grid_shape = self.Nx, self.Ny, self.Nz = self.__eps_r.shape
        self._compute_update_parameters()
        self.initialize_fields()

    def forward(self, Jx=None, Jy=None, Jz=None):
        """ one time step of FDTD """

        self.t_index += 1

        # get curls of E
        CEx = curl_E(0, self.Ex, self.Ey, self.Ez, self.dL)
        CEy = curl_E(1, self.Ex, self.Ey, self.Ez, self.dL)
        CEz = curl_E(2, self.Ex, self.Ey, self.Ez, self.dL)

        # update the curl E integrals
        self.ICEx = self.ICEx + CEx
        self.ICEy = self.ICEy + CEy
        self.ICEz = self.ICEz + CEz

        # update the H field integrals
        self.IHx = self.IHx + self.Hx
        self.IHy = self.IHy + self.Hy
        self.IHz = self.IHz + self.Hz

        # update the H fields
        self.Hx = self.mHx1 * self.Hx + self.mHx2 * CEx + self.mHx3 * self.ICEx + self.mHx4 * self.IHx
        self.Hy = self.mHy1 * self.Hy + self.mHy2 * CEy + self.mHy3 * self.ICEy + self.mHy4 * self.IHy
        self.Hz = self.mHz1 * self.Hz + self.mHz2 * CEz + self.mHz3 * self.ICEz + self.mHz4 * self.IHz

        # update fields dict
        self.fields['Hx'] = self.Hx
        self.fields['Hy'] = self.Hy
        self.fields['Hz'] = self.Hz

        # get curls of H
        CHx = curl_H(0, self.Hx, self.Hy, self.Hz, self.dL)
        CHy = curl_H(1, self.Hx, self.Hy, self.Hz, self.dL)
        CHz = curl_H(2, self.Hx, self.Hy, self.Hz, self.dL)

        # update the curl E integrals
        self.ICHx = self.ICHx + CHx
        self.ICHy = self.ICHy + CHy
        self.ICHz = self.ICHz + CHz

        # update the D field integrals
        self.IDx = self.IDx + self.Dx
        self.IDy = self.IDy + self.Dy
        self.IDz = self.IDz + self.Dz    

        # update the D fields
        self.Dx = self.mDx1 * self.Dx + self.mDx2 * CHx + self.mDx3 * self.ICHx + self.mDx4 * self.IDx
        self.Dy = self.mDy1 * self.Dy + self.mDy2 * CHy + self.mDy3 * self.ICHy + self.mDy4 * self.IDy
        self.Dz = self.mDz1 * self.Dz + self.mDz2 * CHz + self.mDz3 * self.ICHz + self.mDz4 * self.IDz

        # add sources to the electric fields
        self.Dx += 0 if Jx is None else Jx
        self.Dy += 0 if Jy is None else Jy
        self.Dz += 0 if Jz is None else Jz

        # update field dict
        self.fields['Dx'] = self.Dx
        self.fields['Dy'] = self.Dy
        self.fields['Dz'] = self.Dz

        # update the E fields
        self.Ex = self.mEx1 * self.Dx 
        self.Ey = self.mEy1 * self.Dy
        self.Ez = self.mEz1 * self.Dz           

        # update field dict
        self.fields['Ex'] = self.Ex
        self.fields['Ey'] = self.Ey
        self.fields['Ez'] = self.Ez

        return self.fields


    def initialize_fields(self):
        """ Initializes:
              - the H, D, and E fields for updating
              - the integration terms needed to deal with PML
              - the curls of the fields
        """

        self.t_index = 0

        # magnetic fields
        self.Hx = npa.zeros(self.grid_shape)
        self.Hy = npa.zeros(self.grid_shape)
        self.Hz = npa.zeros(self.grid_shape)

        # E field curl integrals
        self.ICEx = npa.zeros(self.grid_shape)
        self.ICEy = npa.zeros(self.grid_shape)
        self.ICEz = npa.zeros(self.grid_shape)

        # H field integrals
        self.IHx = npa.zeros(self.grid_shape)
        self.IHy = npa.zeros(self.grid_shape)
        self.IHz = npa.zeros(self.grid_shape)

        # E field curls
        self.CEx = npa.zeros(self.grid_shape)
        self.CEy = npa.zeros(self.grid_shape)
        self.CEz = npa.zeros(self.grid_shape)

        # H field curl integrals
        self.ICHx = npa.zeros(self.grid_shape)
        self.ICHy = npa.zeros(self.grid_shape)
        self.ICHz = npa.zeros(self.grid_shape)

        # D field integrals
        self.IDx = npa.zeros(self.grid_shape)
        self.IDy = npa.zeros(self.grid_shape)
        self.IDz = npa.zeros(self.grid_shape)

        # H field curls
        self.CHx = npa.zeros(self.grid_shape)
        self.CHy = npa.zeros(self.grid_shape)
        self.CHz = npa.zeros(self.grid_shape)

        # electric displacement fields
        self.Dx = npa.zeros(self.grid_shape)
        self.Dy = npa.zeros(self.grid_shape)
        self.Dz = npa.zeros(self.grid_shape)

        # electric fields
        self.Ex = npa.zeros(self.grid_shape)
        self.Ey = npa.zeros(self.grid_shape)
        self.Ez = npa.zeros(self.grid_shape)

        # field dictionary to return layer
        self.fields = {'Ex': npa.zeros(self.grid_shape),
                       'Ey': npa.zeros(self.grid_shape),
                       'Ez': npa.zeros(self.grid_shape), 
                       'Dx': npa.zeros(self.grid_shape),
                       'Dy': npa.zeros(self.grid_shape),
                       'Dz': npa.zeros(self.grid_shape),
                       'Hx': npa.zeros(self.grid_shape),
                       'Hy': npa.zeros(self.grid_shape),
                       'Hz': npa.zeros(self.grid_shape)
                      }

    def _set_time_step(self, stability_factor=0.5):
        """ Set the time step based on the generalized Courant stability condition
                Delta T < 1 / C_0 / sqrt(1 / dx^2 + 1/dy^2 + 1/dz^2)
                dt = courant_condition * stability_factor, so stability factor should be < 1
        """

        dL_sum = 3 / self.dL ** 2
        dL_avg = 1 / npa.sqrt(dL_sum)
        courant_stability = dL_avg / C_0
        self.dt = courant_stability * stability_factor

    def _compute_sigmas(self):
        """ Computes sigma tensors for PML """

        # initialize sigma matrices on the full 2X grid

        grid_shape_2X = (2 * self.Nx, 2 * self.Ny, 2 * self.Nz)
        sigx2 = np.zeros(grid_shape_2X)
        sigy2 = np.zeros(grid_shape_2X)
        sigz2 = np.zeros(grid_shape_2X)

        # sigma vector in the X direction
        for nx in range(2 * self.npml[0]):
            nx1 = 2 * self.npml[0] - nx + 1
            nx2 = 2 * self.Nx - 2 * self.npml[0] + nx            
            sigx2[nx1, :, :] = (0.5 * EPSILON_0 / self.dt) * (nx / 2 / self.npml[0])**3
            sigx2[nx2, :, :] = (0.5 * EPSILON_0 / self.dt) * (nx / 2 / self.npml[0])**3

        # sigma arrays in the Y direction
        for ny in range(2 * self.npml[1]):
            ny1 = 2 * self.npml[1] - ny + 1
            ny2 = 2 * self.Ny - 2 * self.npml[1] + ny
            sigy2[:, ny1, :] = (0.5 * EPSILON_0 / self.dt) * (ny / 2 / self.npml[1])**3
            sigy2[:, ny2, :] = (0.5 * EPSILON_0 / self.dt) * (ny / 2 / self.npml[1])**3

        # sigma arrays in the Z direction
        for nz in range(2 * self.npml[2]):
            nz1 = 2 * self.npml[2] - nz + 1
            nz2 = 2 * self.Nz - 2 * self.npml[2] + nz
            sigz2[:, :, nz1] = (0.5 * EPSILON_0 / self.dt) * (nz / 2 / self.npml[2])**3
            sigz2[:, :, nz2] = (0.5 * EPSILON_0 / self.dt) * (nz / 2 / self.npml[2])**3            

        # # PML tensors for H field
        self.sigHx = sigx2[1::2,  ::2,  ::2]
        self.sigHy = sigy2[ ::2, 1::2,  ::2]
        self.sigHz = sigz2[ ::2,  ::2, 1::2]

        # # PML tensors for D field
        self.sigDx = sigx2[ ::2, 1::2, 1::2]
        self.sigDy = sigy2[1::2,  ::2, 1::2]
        self.sigDz = sigz2[1::2, 1::2,  ::2]

    def _compute_update_parameters(self, mu_r=1.0):
        """ Computes update coefficients based on values computed earlier.
                For more details, see http://emlab.utep.edu/ee5390fdtd/Lecture%2014%20--%203D%20Update%20Equations%20with%20PML.pdf
                NOTE: relative permeability set = 1 for now
        """

        # H field update coefficients
        self.mHx0 = (1 / self.dt + (self.sigHy + self.sigHz) / 2 / EPSILON_0 + self.sigHy * self.sigHz * self.dt / 4 / EPSILON_0**2)
        self.mHy0 = (1 / self.dt + (self.sigHx + self.sigHz) / 2 / EPSILON_0 + self.sigHx * self.sigHz * self.dt / 4 / EPSILON_0**2)
        self.mHz0 = (1 / self.dt + (self.sigHx + self.sigHy) / 2 / EPSILON_0 + self.sigHx * self.sigHy * self.dt / 4 / EPSILON_0**2)

        self.mHx1 = (1 / self.mHx0 * (1/self.dt - (self.sigHy + self.sigHz) / 2 / EPSILON_0 - self.sigHy * self.sigHz * self.dt / 4 / EPSILON_0**2))
        self.mHy1 = (1 / self.mHy0 * (1/self.dt - (self.sigHx + self.sigHz) / 2 / EPSILON_0 - self.sigHx * self.sigHz * self.dt / 4 / EPSILON_0**2))
        self.mHz1 = (1 / self.mHz0 * (1/self.dt - (self.sigHx + self.sigHy) / 2 / EPSILON_0 - self.sigHx * self.sigHy * self.dt / 4 / EPSILON_0**2))

        self.mHx2 = (-1 / self.mHx0 * C_0 / mu_r)
        self.mHy2 = (-1 / self.mHy0 * C_0 / mu_r)
        self.mHz2 = (-1 / self.mHz0 * C_0 / mu_r)

        self.mHx3 = (-1 / self.mHx0 * C_0 * self.dt * self.sigHx / EPSILON_0 / mu_r)
        self.mHy3 = (-1 / self.mHy0 * C_0 * self.dt * self.sigHy / EPSILON_0 / mu_r)
        self.mHz3 = (-1 / self.mHz0 * C_0 * self.dt * self.sigHz / EPSILON_0 / mu_r)

        self.mHx4 = (-1 / self.mHx0 * self.dt * self.sigHy * self.sigHz / EPSILON_0**2)
        self.mHy4 = (-1 / self.mHy0 * self.dt * self.sigHx * self.sigHz / EPSILON_0**2)
        self.mHz4 = (-1 / self.mHz0 * self.dt * self.sigHx * self.sigHy / EPSILON_0**2)

        # D field update coefficients
        self.mDx0 = (1 / self.dt + (self.sigDy + self.sigDz) / 2 / EPSILON_0 + self.sigDy * self.sigDz * self.dt / 4 / EPSILON_0**2)
        self.mDy0 = (1 / self.dt + (self.sigDx + self.sigDz) / 2 / EPSILON_0 + self.sigDx * self.sigDz * self.dt / 4 / EPSILON_0**2)
        self.mDz0 = (1 / self.dt + (self.sigDx + self.sigDy) / 2 / EPSILON_0 + self.sigDx * self.sigDy * self.dt / 4 / EPSILON_0**2)

        self.mDx1 = (1 / self.mDx0 * (1/self.dt - (self.sigDy + self.sigDz) / 2 / EPSILON_0 - self.sigDy * self.sigDz * self.dt / 4 / EPSILON_0**2))
        self.mDy1 = (1 / self.mDy0 * (1/self.dt - (self.sigDx + self.sigDz) / 2 / EPSILON_0 - self.sigDx * self.sigDz * self.dt / 4 / EPSILON_0**2))
        self.mDz1 = (1 / self.mDz0 * (1/self.dt - (self.sigDx + self.sigDy) / 2 / EPSILON_0 - self.sigDx * self.sigDy * self.dt / 4 / EPSILON_0**2))

        self.mDx2 = (1 / self.mDx0 * C_0)
        self.mDy2 = (1 / self.mDy0 * C_0)
        self.mDz2 = (1 / self.mDz0 * C_0)

        self.mDx3 = (1 / self.mDx0 * C_0 * self.dt * self.sigDx / EPSILON_0)
        self.mDy3 = (1 / self.mDy0 * C_0 * self.dt * self.sigDy / EPSILON_0)
        self.mDz3 = (1 / self.mDz0 * C_0 * self.dt * self.sigDz / EPSILON_0)

        self.mDx4 = (-1 / self.mDx0 * self.dt * self.sigDy * self.sigDz / EPSILON_0**2)
        self.mDy4 = (-1 / self.mDy0 * self.dt * self.sigDx * self.sigDz / EPSILON_0**2)
        self.mDz4 = (-1 / self.mDz0 * self.dt * self.sigDx * self.sigDy / EPSILON_0**2)

        # D -> E update coefficients
        self.mEx1 = (1 / self.eps_xx)
        self.mEy1 = (1 / self.eps_yy)
        self.mEz1 = (1 / self.eps_zz)
# used for setup.py



"""
This file provides wrappers to autograd that compute jacobians.  
The only function you'll want to use in your code is `jacobian`, 
where you can specify the mode of differentiation (reverse, forward, or numerical)
"""

def jacobian(fun, argnum=0, mode='reverse', step_size=1e-6):
    """ Computes jacobian of `fun` with respect to argument number `argnum` using automatic differentiation """

    if mode == 'reverse':
        return jacobian_reverse(fun, argnum)
    elif mode == 'forward':
        return jacobian_forward(fun, argnum)
    elif mode == 'numerical':
        return jacobian_numerical(fun, argnum, step_size=step_size)
    else:
        raise ValueError("'mode' kwarg must be either 'reverse' or 'forward' or 'numerical', given {}".format(mode))


@unary_to_nary
def jacobian_reverse(fun, x):
    """ Compute jacobian of fun with respect to x using reverse mode differentiation"""
    vjp, ans = make_vjp(fun, x)
    grads = map(vjp, vspace(ans).standard_basis())
    m, n = _jac_shape(x, ans)
    return npa.reshape(npa.stack(grads), (n, m))


@unary_to_nary
def jacobian_forward(fun, x):
    """ Compute jacobian of fun with respect to x using forward mode differentiation"""
    jvp = make_jvp(fun, x)
    # ans = fun(x)
    val_grad = map(lambda b: jvp(b), vspace(x).standard_basis())
    vals, grads = zip(*val_grad)
    ans = npa.zeros((list(vals)[0].size,))  # fake answer so that dont have to compute it twice
    m, n = _jac_shape(x, ans)
    if _iscomplex(x):
        grads_real = npa.array(grads[::2])
        grads_imag = npa.array(grads[1::2])
        grads = grads_real - 1j * grads_imag
    return npa.reshape(npa.stack(grads), (m, n)).T


@unary_to_nary
def jacobian_numerical(fn, x, step_size=1e-7):
    """ numerically differentiate `fn` w.r.t. its argument `x` """
    in_array = float_2_array(x).flatten()
    out_array = float_2_array(fn(x)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (n, m)
    jacobian = npa.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[:, i] = get_value_arr(get_value(grad_i))  # need to convert both the grad_i array and its contents to actual data.

    return jacobian


def _jac_shape(x, ans):
    """ computes the shape of the jacobian where function has input x and output ans """
    m = float_2_array(x).size
    n = float_2_array(ans).size
    return (m, n)


def _iscomplex(x):
    """ Checks if x is complex-valued or not """
    if isinstance(x, npa.ndarray):
        if x.dtype == npa.complex128:
            return True
    if isinstance(x, complex):
        return True
    return False


if __name__ == '__main__':

    """ Some simple test """

    N = 3
    M = 2
    A = npa.random.random((N,M))
    B = npa.random.random((N,M))
    print('A = \n', A)

    def fn(x, b):
        return A @ x + B @ b

    x0 = npa.random.random((M,))
    b0 = npa.random.random((M,))    
    print('Jac_rev = \n', jacobian(fn, argnum=0, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=0, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=0, mode='numerical')(x0, b0))

    print('B = \n', B)
    print('Jac_rev = \n', jacobian(fn, argnum=1, mode='reverse')(x0, b0))
    print('Jac_for = \n', jacobian(fn, argnum=1, mode='forward')(x0, b0))
    print('Jac_num = \n', jacobian(fn, argnum=1, mode='numerical')(x0, b0))




def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    """ Solve for the modes of a waveguide cross section
        ARGUMENTS
            eps_cross: the permittivity profile of the waveguide
            omega:     angular frequency of the modes
            dL:        grid size of the cross section
            npml:      number of PML points on each side of the cross section
            m:         number of modes to solve for
            filtering: whether to filter out evanescent modes
        RETURNS
            vals:      array of effective indeces of the modes
            vectors:   array containing the corresponding mode profiles
    """

    k0 = omega / C_0

    N = eps_cross.size

    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)

    Dxf, Dxb, Dyf, Dyb = matrices

    diag_eps_r = sp.spdiags(eps_cross.flatten(), [0], N, N)
    A = diag_eps_r + Dxf.dot(Dxb) * (1 / k0) ** 2

    n_max = np.sqrt(np.max(eps_cross))
    vals, vecs = solver_eigs(A, m, guess_value=n_max**2)

    if filtering:
        filter_re = lambda vals: np.real(vals) > 0.0
        # filter_im = lambda vals: np.abs(np.imag(vals)) <= 1e-12
        filters = [filter_re]
        vals, vecs = filter_modes(vals, vecs, filters=filters)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    vecs = normalize_modes(vecs)

    return vals, vecs


def insert_mode(omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is suppled, if the target array is not
    supplied, then a target array is created with the same shape as epsr, and the mode is
    inserted into it.
    """
    if target is None:
        target = np.zeros(epsr.shape, dtype=complex)

    epsr_cross = epsr[x, y]
    _, mode_field = get_modes(epsr_cross, omega, dx, npml, m=m, filtering=filtering)
    target[x, y] = np.atleast_2d(mode_field)[:,m-1].squeeze()

    return target


def solver_eigs(A, Neigs, guess_value=1.0):
    """ solves for `Neigs` eigenmodes of A
            A:            sparse linear operator describing modes
            Neigs:        number of eigenmodes to return
            guess_value:  estimate for the eigenvalues
        For more info, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
    """

    values, vectors = spl.eigs(A, k=Neigs, sigma=guess_value, v0=None, which='LM')

    return values, vectors


def filter_modes(values, vectors, filters=None):
    """ Generic Filtering Function
        ARGUMENTS
            values: array of effective index values
            vectors: array of mode profiles
            filters: list of functions of `values` that return True for modes satisfying the desired filter condition
        RETURNS
            vals:      array of filtered effective indeces of the modes
            vectors:   array containing the corresponding, filtered mode profiles
    """

    # if no filters, just return
    if filters is None:
        return values, vectors

    # elements to keep, all for starts
    keep_elements = np.ones(values.shape)

    for f in filters:
        keep_f = f(values)
        keep_elements = np.logical_and(keep_elements, keep_f)

    # get the indeces you want to keep
    keep_indeces = np.where(keep_elements)[0]

    # filter and return arrays
    return values[keep_indeces], vectors[:, keep_indeces]


def normalize_modes(vectors):
    """ Normalize each `vec` in `vectors` such that `sum(|vec|^2)=1`
            vectors: array with shape (n_points, n_vectors)
        NOTE: eigs already normalizes for you, so you technically dont need this function
    """

    powers = np.sum(np.square(np.abs(vectors)), axis=0)

    return vectors / np.sqrt(powers)

def Ez_to_H(Ez, omega, dL, npml):
    """ Converts the Ez output of mode solver to Hx and Hy components
    """

    N = Ez.size
    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)
    Dxf, Dxb, Dyf, Dyb = matrices

    # save to a dictionary for convenience passing to primitives
    info_dict = {}
    info_dict['Dxf'] = Dxf
    info_dict['Dxb'] = Dxb
    info_dict['Dyf'] = Dyf
    info_dict['Dyb'] = Dyb

    Hx, Hy = Ez_to_Hx_Hy(Ez)

    return Hx, Hy

def adam_optimize(objective, params, jac, step_size=1e-2, Nsteps=100, bounds=None, direction='min', beta1=0.9, beta2=0.999, callback=None, verbose=True):
    """Performs Nsteps steps of ADAM minimization of function `objective` with gradient `jac`.
    The `bounds` are set abruptly by rejecting an update step out of bounds."""
    of_list = []

    np.set_printoptions(formatter={'float': '{: 1.4f}'.format})

    for iteration in range(Nsteps):

        if callback:
            callback(iteration, of_list, params)

        t_start = time.time()
        if jac==True:
            of, grad = objective(params)
        else:
            of = objective(params)
            grad = jac(params)
        t_elapsed = time.time() - t_start

        of_list.append(of._value if type(of) is ArrayBox else of) 

        if verbose:
            print("Epoch: %3d/%3d | Duration: %.2f secs | Value: %5e" %(iteration+1, Nsteps, t_elapsed, of_list[-1]))

        if iteration == 0:
            mopt = np.zeros(grad.shape)
            vopt = np.zeros(grad.shape)

        (grad_adam, mopt, vopt) = step_adam(grad, mopt, vopt, iteration, beta1, beta2)

        if direction == 'min':
            params = params - step_size*grad_adam
        elif direction == 'max':
            params = params + step_size*grad_adam
        else:
            raise ValueError("The 'direction' parameter should be either 'min' or 'max'")

        if bounds:
            params[params < bounds[0]] = bounds[0]
            params[params > bounds[1]] = bounds[1]

    return (params, of_list)


def step_adam(gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
    """ Performs one step of adam optimization"""

    mopt = beta1 * mopt_old + (1 - beta1) * gradient
    mopt_t = mopt / (1 - beta1**(iteration + 1))
    vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
    vopt_t = vopt / (1 - beta2**(iteration + 1))
    grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

    return (grad_adam, mopt, vopt)


""" This file defines the very lowest level sparse matrix primitives that allow autograd to
be compatible with FDFD.  One needs to define the derivatives of Ax = b and x = A^-1 b for sparse A.

This is done using the entries and indices of A, instead of the sparse matrix objects, since autograd doesn't
know how to handle those as arguments to functions.
"""

""" GUIDE TO THE PRIMITIVES DEFINED BELOW:
        naming convention for gradient functions:
           "def grad_{function_name}_{argument_name}_{mode}"
        defines the derivative of `function_name` with respect to `argument_name` using `mode`-mode differentiation    
        where 'mode' is one of 'reverse' or 'forward'

    These functions define the basic operations needed for FDFD and also their derivatives
    in a form that autograd can understand.
    This allows you to use fdfd classes in autograd functions.
    The code is organized so that autograd never sees sparse matrices in arguments, since it doesn't know how to handle them
    Look but don't touch!

    NOTES for the curious (since this information isnt in autograd documentation...)

        To define a function as being trackable by autograd, need to add the 
        @primitive decorator

    REVERSE MODE
        'vjp' defines the vector-jacobian product for reverse mode (adjoint)
        a vjp_maker function takes as arguments
            1. the output of the @primitive
            2. the rest of the original arguments in the @primitive
        and returns
            a *function* of the backprop vector (v) that defines the operation
            (d{function} / d{argument_i})^T @ v

    FORWARD MODE:
        'jvp' defines the jacobian-vector product for forward mode (FMD)
        a jvp_maker function takes as arguments
            1. the forward propagating vector (g)
            2. the output of the @primitive
            3. the rest of the original arguments in the @primitive
        and returns
            (d{function} / d{argument_i}) @ g

    After this, you need to link the @primitive to its vjp/jvp using
    defvjp(function, arg1's vjp, arg2's vjp, ...)
    defjvp(function, arg1's jvp, arg2's jvp, ...)
"""

""" ========================== Sparse Matrix-Vector Multiplication =========================="""

@ag.primitive
def sp_mult(entries, indices, x):
    """ Multiply a sparse matrix (A) by a dense vector (x)
    Args:
      entries: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries into A.
      indices: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries into A.
      x: 1d numpy array specifying the vector to multiply by.
    Returns:
      1d numpy array corresponding to the result (b) of A * x = b.
    """
    N = x.size
    A = make_sparse(entries, indices, shape=(N, N))
    return A.dot(x)

def grad_sp_mult_entries_reverse(ans, entries, indices, x):
    # x^T @ dA/de^T @ v => the outer product of x and v using the indices of A
    ia, ja = indices
    def vjp(v):
        return v[ia] * x[ja]
    return vjp

def grad_sp_mult_x_reverse(b, entries, indices, x):
    # dx/de^T @ A^T @ v => multiplying A^T by v
    indices_T = transpose_indices(indices)
    def vjp(v):
        return sp_mult(entries, indices_T, v)
    return vjp

ag.extend.defvjp(sp_mult, grad_sp_mult_entries_reverse, None, grad_sp_mult_x_reverse)

def grad_sp_mult_entries_forward(g, b, entries, indices, x):
    # dA/de @ x @ g => use `g` as the entries into A and multiply by x
    return sp_mult(g, indices, x)

def grad_sp_mult_x_forward(g, b, entries, indices, x):
    # A @ dx/de @ g -> simply multiply A @ g
    return sp_mult(entries, indices, g)

ag.extend.defjvp(sp_mult, grad_sp_mult_entries_forward, None, grad_sp_mult_x_forward)


""" ========================== Sparse Matrix-Vector Solve =========================="""

@ag.primitive
def sp_solve(entries, indices, b):
    """ Solve a sparse matrix (A) with source (b)
    Args:
      entries: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries.
      indices: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries.
      b: 1d numpy array specifying the source.
    Returns:
      1d numpy array corresponding to the solution of A * x = b.
    Note: Calls a customizable solving function from ceviche.solvers, could add options to sp_solve() here eventually
    """
    N = b.size
    A = make_sparse(entries, indices, shape=(N, N))
    # calls a customizable solving function from ceviche.solvers, could add options to sp_solve() here eventually
    return solve_linear(A, b)

def grad_sp_solve_entries_reverse(x, entries, indices, b):
    # x^T @ dA/de^T @ A_inv^T @ -v => do the solve on the RHS, then take outer product with x using indices of A
    indices_T = transpose_indices(indices)
    i, j = indices
    def vjp(v):
        adj = sp_solve(entries, indices_T, -v)
        return adj[i] * x[j]
    return vjp

def grad_sp_solve_b_reverse(ans, entries, indices, b):
    # dx/de^T @ A_inv^T @ v => do the solve on the RHS and you're done.
    indices_T = transpose_indices(indices)
    def vjp(v):
        return sp_solve(entries, indices_T, v)
    return vjp

ag.extend.defvjp(sp_solve, grad_sp_solve_entries_reverse, None, grad_sp_solve_b_reverse)

def grad_sp_solve_entries_forward(g, x, entries, indices, b):
    # -A_inv @ dA/de @ A_inv @ b @ g => insert x = A_inv @ b and multiply with g using A indices.  Then solve as source for A_inv.
    forward = sp_mult(g, indices, x)
    return sp_solve(entries, indices, -forward)

def grad_sp_solve_b_forward(g, x, entries, indices, b):
    # A_inv @ db/de @ g => simply solve A_inv @ g
    return sp_solve(entries, indices, g)

ag.extend.defjvp(sp_solve, grad_sp_solve_entries_forward, None, grad_sp_solve_b_forward)


""" ==========================Sparse Matrix-Sparse Matrix Multiplication ========================== """

@ag.primitive
def spsp_mult(entries_a, indices_a, entries_x, indices_x, N):
    """ Multiply a sparse matrix (A) by a sparse matrix (X) A @ X = B
    Args:
      entries_a: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries into A.
      indices_a: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries into A.
      entries_x: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries into X.
      indices_x: numpy array with shape (2, num_non_zeros) giving x and y indices for
        non-zero matrix entries into X.
      N: all matrices are assumed of shape (N, N) (need to specify because no dense vector supplied)
    Returns:
      entries_b: numpy array with shape (num_non_zeros,) giving values for non-zero
        matrix entries into the result B.
      indices_b: numpy array with shape (2, num_non_zeros) giving i, j indices for
        non-zero matrix entries into the result B.      
    """
    A = make_sparse(entries_a, indices_a, shape=(N, N))
    X = make_sparse(entries_x, indices_x, shape=(N, N))
    B = A.dot(X)
    entries_b, indices_b = get_entries_indices(B)
    return entries_b, indices_b

def grad_spsp_mult_entries_a_reverse(b_out, entries_a, indices_a, entries_x, indices_x, N):
    """ For AX=B, we want to relate the entries of A to the entries of B.
        The goal is to compute the gradient of the output entries with respect to the input.
        For this, though, we need to convert into matrix form, do our computation, and convert back to the entries.
        If you write out the matrix elements and do the calculation, you can derive the code below, but it's a hairy derivation.
    """

    # make the indices matrices for A
    _, indices_b = b_out
    Ia, Oa = make_IO_matrices(indices_a, N)

    def vjp(v):

        # multiply the v_entries with X^T using the indices of B
        entries_v, _ = v
        indices_xT = transpose_indices(indices_x)
        entries_vxt, indices_vxt = spsp_mult(entries_v, indices_b, entries_x, indices_xT, N)

        # rutn this into a sparse matrix and convert to the basis of A's indices
        VXT = make_sparse(entries_vxt, indices_vxt, shape=(N, N))
        M = (Ia.T).dot(VXT).dot(Oa.T)

        # return the diagonal elements, which contain the entries
        return M.diagonal()

    return vjp

def grad_spsp_mult_entries_x_reverse(b_out, entries_a, indices_a, entries_x, indices_x, N):
    """ Now we wish to do the gradient with respect to the X matrix in AX=B
        Instead of doing it all out again, we just use the previous grad function on the transpose equation X^T A^T = B^T 
    """

    # get the transposes of the original problem
    entries_b, indices_b = b_out
    indices_aT = transpose_indices(indices_a)
    indices_xT = transpose_indices(indices_x)
    indices_bT = transpose_indices(indices_b)
    b_T_out = entries_b, indices_bT

    # call the vjp maker for AX=B using the substitution A=>X^T, X=>A^T, B=>B^T
    vjp_XT_AT = grad_spsp_mult_entries_a_reverse(b_T_out, entries_x, indices_xT, entries_a, indices_aT, N)

    # return the function of the transpose vjp maker being called on the backprop vector
    return lambda v: vjp_XT_AT(v)

ag.extend.defvjp(spsp_mult, grad_spsp_mult_entries_a_reverse, None, grad_spsp_mult_entries_x_reverse, None, None)

def grad_spsp_mult_entries_a_forward(g, b_out, entries_a, indices_a, entries_x, indices_x, N):
    """ Forward mode is not much better than reverse mode, but the same general logic aoplies:
        Convert to matrix form, do the calculation, convert back to entries.        
            dA/de @ x @ g
    """

    # get the IO indices matrices for B
    _, indices_b = b_out
    Mb = indices_b.shape[1]
    Ib, Ob = make_IO_matrices(indices_b, N)

    # multiply g by X using a's index
    entries_gX, indices_gX = spsp_mult(g, indices_a, entries_x, indices_x, N)
    gX = make_sparse(entries_gX, indices_gX, shape=(N, N))

    # convert these entries and indides into the basis of the indices of B
    M = (Ib.T).dot(gX).dot(Ob.T)

    # return the diagonal (resulting entries) and indices of 0 (because indices are not affected by entries)
    return M.diagonal(), npa.zeros(Mb)

def grad_spsp_mult_entries_x_forward(g, b_out, entries_a, indices_a, entries_x, indices_x, N):
    """ Same trick as before: Reuse the previous VJP but for the transpose system """

    # Transpose A, X, and B
    indices_aT = transpose_indices(indices_a)
    indices_xT = transpose_indices(indices_x)
    entries_b, indices_b = b_out
    indices_bT = transpose_indices(indices_b)
    b_T_out = entries_b, indices_bT

    # return the jvp of B^T = X^T A^T
    return grad_spsp_mult_entries_a_forward(g, b_T_out, entries_x, indices_xT, entries_a, indices_aT, N)

ag.extend.defjvp(spsp_mult, grad_spsp_mult_entries_a_forward, None, grad_spsp_mult_entries_x_forward, None, None)


""" ========================== Nonlinear Solve ========================== """

# this is just a sketch of how to do problems involving sparse matrix solves with nonlinear elements...  WIP.

def sp_solve_nl(parameters, a_indices, b, fn_nl):
    """
        parameters: entries into matrix A are function of parameters and solution x
        a_indices: indices into sparse A matrix
        b: source vector for A(xx = b
        fn_nl: describes how the entries of a depend on the solution of A(x,p) @ x = b and the parameters  `a_entries = fn_nl(params, x)`
    """

    # do the actual nonlinear solve in `_solve_nl_problem` (using newton, picard, whatever)
    # this tells you the final entries into A given the parameters and the nonlinear function.
    a_entries = ceviche.solvers._solve_nl_problem(parameters, a_indices, fn_nl, a_entries0=None)  # optinally, give starting a_entries
    x = sp_solve(a_entries, a_indices, b)  # the final solution to A(x) x = b
    return x

def grad_sp_solve_nl_parameters(x, parameters, a_indices, b, fn_nl):

    """ 
    We are finding the solution (x) to the nonlinear function:

        f = A(x, p) @ x - b = 0

    And need to define the vjp of the solution (x) with respect to the parameters (p)

        vjp(v) = (dx / dp)^T @ v

    To do this (see Eq. 5 of https://pubs-acs-org.stanford.idm.oclc.org/doi/pdf/10.1021/acsphotonics.8b01522)
    we need to solve the following linear system:

        [ df  / dx,  df  / dx*] [ dx  / dp ] = -[ df  / dp]
        [ df* / dx,  df* / dx*] [ dx* / dp ]    [ df* / dp]
    
    Note that we need to explicitly make A a function of x and x* for complex x

    In our case:

        (df / dx)  = (dA / dx) @ x + A
        (df / dx*) = (dA / dx*) @ x
        (df / dp)  = (dA / dp) @ x

    How do we put this into code?  Let

        A(x, p) @ x -> Ax = sp_mult(entries_a(x, p), indices_a, x)

    Since we already defined the primitive of sp_mult, we can just do:

        (dA / dx) @ x -> ag.jacobian(Ax, 0)

    Now how about the source term?

        (dA / dp) @ x -> ag.jacobian(Ax, 1)

    Note that this is a matrix, not a vector. 
    We'll have to handle dA/dx* but this can probably be done, maybe with autograd directly.

    Other than this, assuming entries_a(x, p) is fully autograd compatible, we can get these terms no problem!

    Coming back to our problem, we actually need to compute:

        (dx / dp)^T @ v

    Because

        (dx / dp) = -(df / dx)^{-1} @ (df / dp)

    (ignoring the complex conjugate terms).  We can write this vjp as

        (df / dp)^T @ (df / dx)^{-T} @ v

    Since df / dp is a matrix, not a vector, its more efficient to do the mat_mul on the right first.
    So we first solve

        adjoint(v) = -(df / dx)^{-T} @ v
                   => sp_solve(entries_a_big, transpose(indices_a_big), -v)

    and then it's a simple matter of doing the matrix multiplication

        vjp(v) = (df / dp)^T @ adjoint(v)
               => sp_mult(entries_dfdp, transpose(indices_dfdp), adjoint)

    and then return the result, making sure to strip the complex conjugate.

        return vjp[:N]
    """

    def vjp(v):
        raise NotImplementedError
    return vjp

def grad_sp_solve_nl_b(x, parameters, a_indices, b, fn_nl):

    """ 
    Computing the derivative w.r.t b is simpler

        f = A(x) @ x - b(p) = 0

    And now the terms we need are

        df / dx  = (dA / dx) @ x + A
        df / dx* = (dA / dx*) @ x
        df / dp  = -(db / dp)

    So it's basically the same problem with a differenct source term now.
    """

    def vjp(v):
        raise NotImplementedError
    return vjp

ag.extend.defvjp(sp_solve_nl, grad_sp_solve_nl_parameters, None, grad_sp_solve_nl_b, None)


""" This file stores the various sparse linear system solvers you can use for FDFD """

# try to import MKL but just use scipy sparse solve if not
try:
    from pyMKL import pardisoSolver
    HAS_MKL = True
    # print('using MKL for direct solvers')
except:
    HAS_MKL = False
    # print('using scipy.sparse for direct solvers.  Note: using MKL will make things significantly faster.')

# default iterative method to use
# for reference on the methods available, see:  https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
DEFAULT_ITERATIVE_METHOD = 'bicg'

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """

def solve_linear(A, b, iterative_method=False):
    """ Master function to call the others """

    if iterative_method and iterative_method is not None:
        # if iterative solver string is supplied, use that method
        return _solve_iterative(A, b, iterative_method=iterative_method)
    elif iterative_method and iterative_method is None:
        # if iterative_method is supplied as None, use the default
        return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
    else:
        # otherwise, use a direct solver
        return _solve_direct(A, b)

def _solve_direct(A, b):
    """ Direct solver """

    if HAS_MKL:
        # prefered method using MKL. Much faster (on Mac at least)
        pSolve = pardisoSolver(A, mtype=13)
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()
        return x
    else:
        # scipy solver.
        return spl.spsolve(A, b)

def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """ Iterative solver """

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL)
    return x

def _solve_cuda(A, b, **kwargs):
    """ You could put some other solver here if you're feeling adventurous """
    raise NotImplementedError("Please implement something fast and exciting here!")


""" ============================ SPEED TESTS ============================= """

# to run speed tests use `python -W ignore ceviche/solvers.py` to suppress warnings

""" Source functions go here.  For now it's just TFSF """

def b_TFSF(fdfd, inside_mask, theta):
    """ Returns a source vector for FDFD that will implement TFSF 
            A: the FDFD system matrix
            inside_mask: a binary mask (vector) specifying the inside of the TFSF region
            theta: [0, 2pi] the angle of the source relative to y=0+

                      y ^
                        |
                        |
                  <-----|----- > x
                        |\
                        | \                                     
                        v |\
                          theta             
    see slide 32 of https://empossible.net/wp-content/uploads/2019/08/Lecture-4d-FDFD-Formulation.pdf                                                       
    """

    lambda0 = 2 * npa.pi * C_0 / fdfd.omega
    f_src = compute_f(theta, lambda0, fdfd.dL,  inside.shape)

    Q = compute_Q(inside_mask) / fdfd.omega # why this omega??
    A = fdfd.make_A(fdfd.eps_r.copy().flatten())

    quack = (Q.dot(A) - A.dot(Q))

    return quack.dot(f_src)

def compute_Q(inside_mask):
    """ Compute the matrix used in PDF to get source """

    # convert masks to vectors and get outside portion
    inside_vec = inside_mask.flatten()
    outside_vec = 1 - inside_vec
    N = outside_vec.size

    # make a sparse diagonal matrix and return
    Q = sp.diags([outside_vec], [0], shape=(N, N))
    return Q


def compute_f(theta, lambda0, dL, shape):
    """ Compute the 'vacuum' field vector """

    # get plane wave k vector components (in units of grid cells)
    k0 = 2 * npa.pi / lambda0 * dL
    kx =  k0 * npa.sin(theta)
    ky = -k0 * npa.cos(theta)  # negative because downwards

    # array to write into
    f_src = npa.zeros(shape, dtype=npa.complex128)

    # get coordinates
    Nx, Ny = shape
    xpoints = npa.arange(Nx)
    ypoints = npa.arange(Ny)
    xv, yv = npa.meshgrid(xpoints, ypoints, indexing='ij')

    # compute values and insert into array
    x_PW = npa.exp(1j * xpoints * kx)[:, None]
    y_PW = npa.exp(1j * ypoints * ky)[:, None]

    f_src[xv, yv] = npa.outer(x_PW, y_PW)

    return f_src.flatten()
""" Useful functions """


""" ==================== SPARSE MATRIX UTILITIES ==================== """

def make_sparse(entries, indices, shape):
    """Construct a sparse csr matrix
    Args:
      entries: numpy array with shape (M,) giving values for non-zero
        matrix entries.
      indices: numpy array with shape (2, M) giving x and y indices for
        non-zero matrix entries.
      shape: shape of resulting matrix
    Returns:
      sparse, complex, matrix with specified values
    """  
    coo = sp.coo_matrix((entries, indices), shape=shape, dtype=npa.complex128)
    return coo.tocsr()

def get_entries_indices(csr_matrix):
    # takes sparse matrix and returns the entries and indeces in form compatible with 'make_sparse'
    shape = csr_matrix.shape
    coo_matrix = csr_matrix.tocoo()
    entries = csr_matrix.data
    cols = coo_matrix.col
    rows = coo_matrix.row
    indices = npa.vstack((rows, cols))
    return entries, indices

def transpose_indices(indices):
    # returns the transposed indices for transpose sparse matrix creation
   return npa.flip(indices, axis=0)

def block_4(A, B, C, D):
    """ Constructs a big matrix out of four sparse blocks
        returns [A B]
                [C D]
    """
    left = sp.vstack([A, C])
    right = sp.vstack([B, D])
    return sp.hstack([left, right])    

def make_IO_matrices(indices, N):
    """ Makes matrices that relate the sparse matrix entries to their locations in the matrix
            The kth column of I is a 'one hot' vector specifing the k-th entries row index into A
            The kth column of J is a 'one hot' vector specifing the k-th entries columnn index into A
            O = J^T is for notational convenience.
            Armed with a vector of M entries 'a', we can construct the sparse matrix 'A' as:
                A = I @ diag(a) @ O
            where 'diag(a)' is a (MxM) matrix with vector 'a' along its diagonal.
            In index notation:
                A_ij = I_ik * a_k * O_kj
            In an opposite way, given sparse matrix 'A' we can strip out the entries `a` using the IO matrices as follows:
                a = diag(I^T @ A @ O^T)
            In index notation:
                a_k = I_ik * A_ij * O_kj
    """
    M = indices.shape[1]                                 # number of indices in the matrix
    entries_1 = npa.ones(M)                              # M entries of all 1's
    ik, jk = indices                                     # separate i and j components of the indices
    indices_I = npa.vstack((ik, npa.arange(M)))          # indices into the I matrix
    indices_J = npa.vstack((jk, npa.arange(M)))          # indices into the J matrix
    I = make_sparse(entries_1, indices_I, shape=(N, M))  # construct the I matrix
    J = make_sparse(entries_1, indices_J, shape=(N, M))  # construct the J matrix
    O = J.T                                              # make O = J^T matrix for consistency with my notes.
    return I, O


""" ==================== DATA GENERATION UTILITIES ==================== """

def make_rand(N):
    # makes a random vector of size N with elements between -0.5 and 0.5
    return npa.random.random(N) - 0.5

def make_rand_complex(N):
    # makes a random complex-valued vector of size N with re and im parts between -0.5 and 0.5
    return make_rand(N) + 1j * make_rand(N)

def make_rand_indeces(N, M):
    # make M random indeces into an NxN matrix
    return npa.random.randint(low=0, high=N, size=(2, M))

def make_rand_entries_indices(N, M):
    # make M random indeces and corresponding entries
    entries = make_rand_complex(M)
    indices = make_rand_indeces(N, M)
    return entries, indices

def make_rand_sparse(N, M):
    # make a random sparse matrix of shape '(N, N)' and 'M' non-zero elements
    entries, indices = make_rand_entries_indices(N, M)
    return make_sparse(entries, indices, shape=(N, N))

def make_rand_sparse_density(N, density=1):
    """ Makes a sparse NxN matrix, another way to do it with density """
    return sp.random(N, N, density=density) + 1j * sp.random(N, N, density=density)


""" ==================== NUMERICAL DERIVAITVES ==================== """

def der_num(fn, arg, index, delta):
    # numerical derivative of `fn(arg)` with respect to `index` into arg and numerical step size `delta`
    arg_i_for  = arg.copy()
    arg_i_back = arg.copy()
    arg_i_for[index] += delta / 2
    arg_i_back[index] -= delta / 2
    df_darg = (fn(arg_i_for) - fn(arg_i_back)) / delta
    return df_darg

def grad_num(fn, arg, delta=1e-6):
    # take a (complex) numerical gradient of function 'fn' with argument 'arg' with step size 'delta'
    N = arg.size
    grad = npa.zeros((N,), dtype=npa.complex128)
    f0 = fn(arg)
    for i in range(N):
        grad[i] = der_num(fn, arg, i, delta)        # real part
        grad[i] += der_num(fn, arg, i, 1j * delta)  # imaginary part
    return grad

def jac_num(fn, arg, step_size=1e-7):
    """ DEPRICATED: use 'numerical' in jacobians.py instead
    numerically differentiate `fn` w.r.t. its argument `arg` 
    `arg` can be a numpy array of arbitrary shape
    `step_size` can be a number or an array of the same shape as `arg` """

    in_array = float_2_array(arg).flatten()
    out_array = float_2_array(fn(arg)).flatten()

    m = in_array.size
    n = out_array.size
    shape = (m, n)
    jacobian = np.zeros(shape)

    for i in range(m):
        input_i = in_array.copy()
        input_i[i] += step_size
        arg_i = input_i.reshape(in_array.shape)
        output_i = fn(arg_i).flatten()
        grad_i = (output_i - out_array) / step_size
        jacobian[i, :] = get_value(grad_i)

    return jacobian

""" ==================== FDTD AND FDFD UTILITIES ==================== """

def grid_center_to_xyz(Q_mid, averaging=True):
    """ Computes the interpolated value of the quantity Q_mid felt at the Ex, Ey, Ez positions of the Yee latice
        Returns these three components
    """

    # initialize
    Q_xx = copy.copy(Q_mid)
    Q_yy = copy.copy(Q_mid)
    Q_zz = copy.copy(Q_mid)

    # if averaging, set the respective xx, yy, zz components to the midpoint in the Yee lattice.
    if averaging:

        # get the value from the middle of the next cell over
        Q_x_r = npa.roll(Q_mid, shift=1, axis=0)
        Q_y_r = npa.roll(Q_mid, shift=1, axis=1)
        Q_z_r = npa.roll(Q_mid, shift=1, axis=2)

        # average with the two middle values
        Q_xx = (Q_mid + Q_x_r)/2
        Q_yy = (Q_mid + Q_y_r)/2
        Q_zz = (Q_mid + Q_z_r)/2

    return Q_xx, Q_yy, Q_zz


def grid_xyz_to_center(Q_xx, Q_yy, Q_zz):
    """ Computes the interpolated value of the quantitys Q_xx, Q_yy, Q_zz at the center of Yee latice
        Returns these three components
    """

    # compute the averages
    Q_xx_avg = (Q_xx.astype('float') + npa.roll(Q_xx, shift=1, axis=0))/2
    Q_yy_avg = (Q_yy.astype('float') + npa.roll(Q_yy, shift=1, axis=1))/2
    Q_zz_avg = (Q_zz.astype('float') + npa.roll(Q_zz, shift=1, axis=2))/2

    return Q_xx_avg, Q_yy_avg, Q_zz_avg

def vec_zz_to_xy(info_dict, vec_zz, grid_averaging=True):
    """ does grid averaging on z vector vec_zz """
    arr_zz = vec_zz.reshape(info_dict['shape'])[:,:,None]
    arr_xx, arr_yy, _ = grid_center_to_xyz(arr_zz, averaging=grid_averaging)
    vec_xx, vec_yy = arr_xx.flatten(), arr_yy.flatten()
    return vec_xx, vec_yy

""" ===================== TESTING AND DEBUGGING ===================== """

def float_2_array(x):
    if not isinstance(x, np.ndarray):
        return np.array([x])
    else:
        return x

def reshape_to_ND(arr, N):
    """ Adds dimensions to arr until it is dimension N
    """

    ND = len(arr.shape)
    if ND > N:
        raise ValueError("array is larger than {} dimensional, given shape {}".format(N, arr.shape))
    extra_dims = (N - ND) * (1,)
    return arr.reshape(arr.shape + extra_dims)


""" ========================= TOOLS USEFUL FOR WORKING WITH AUTOGRAD ====================== """


def get_value(x):
    if type(x) == npa.numpy_boxes.ArrayBox:
        return x._value
    else:
        return x

get_value_arr = np.vectorize(get_value)


def get_shape(x):
    """ Gets the shape of x, even if it is not an array """
    if isinstance(x, float) or isinstance(x, int):
        return (1,)
    elif isinstance(x, tuple) or isinstance(x, list):
        return (len(x),)
    else:
        return vspace(x).shape


def vjp_maker_num(fn, arg_inds, steps):
    """ Makes a vjp_maker for the numerical derivative of a function `fn`
    w.r.t. argument at position `arg_ind` using step sizes `steps` """

    def vjp_single_arg(ia):
        arg_ind = arg_inds[ia]
        step = steps[ia]

        def vjp_maker(fn_out, *args):
            shape = args[arg_ind].shape
            num_p = args[arg_ind].size
            step = steps[ia]

            def vjp(v):

                vjp_num = np.zeros(num_p)
                for ip in range(num_p):
                    args_new = list(args)
                    args_rav = args[arg_ind].flatten()
                    args_rav[ip] += step
                    args_new[arg_ind] = args_rav.reshape(shape)
                    dfn_darg = (fn(*args_new) - fn_out)/step
                    vjp_num[ip] = np.sum(v * dfn_darg)

                return vjp_num

            return vjp

        return vjp_maker

    vjp_makers = []
    for ia in range(len(arg_inds)):
        vjp_makers.append(vjp_single_arg(ia=ia))

    return tuple(vjp_makers)


""" =================== PLOTTING AND MEASUREMENT OF FDTD =================== """


def aniplot(F, source, steps, component='Ez', num_panels=10):
    """ Animate an FDTD (F) with `source` for `steps` time steps.
    display the `component` field components at `num_panels` equally spaced.
    """
    F.initialize_fields()

    # initialize the plot
    f, ax_list = plt.subplots(1, num_panels, figsize=(20*num_panels,20))
    Nx, Ny, _ = F.eps_r.shape
    ax_index = 0

    # fdtd time loop
    for t_index in range(steps):
        fields = F.forward(Jz=source(t_index))

        # if it's one of the num_panels panels
        if t_index % (steps // num_panels) == 0:

            if ax_index < num_panels:   # extra safety..sometimes tries to access num_panels-th elemet of ax_list, leading to error

                print('working on axis {}/{} for time step {}'.format(ax_index, num_panels, t_index))

                # grab the axis
                ax = ax_list[ax_index]

                # plot the fields
                im_t = ax.pcolormesh(np.zeros((Nx, Ny)), cmap='RdBu')
                max_E = np.abs(fields[component]).max()
                im_t.set_array(fields[component][:, :, 0].ravel().T)
                im_t.set_clim([-max_E / 2.0, max_E / 2.0])
                ax.set_title('time = {} seconds'.format(F.dt*t_index))

                # update the axis
                ax_index += 1
    plt.show()


def measure_fields(F, source, steps, probes, component='Ez'):
    """ Returns a time series of the measured `component` fields from FDFD `F`
        driven by `source and measured at `probe`.
    """
    F.initialize_fields()
    if not isinstance(probes, list):
        probes = [probes]
    N_probes = len(probes)
    measured = np.zeros((steps, N_probes))
    for t_index in range(steps):
        if t_index % (steps//20) == 0:
            print('{:.2f} % done'.format(float(t_index)/steps*100.0))
        fields = F.forward(Jz=source(t_index))
        for probe_index, probe in enumerate(probes):
            field_probe = np.sum(fields[component] * probe)
            measured[t_index, probe_index] = field_probe
    return measured


def imarr(arr):
    """ puts array 'arr' into form ready to plot """
    arr_value = get_value(arr)
    arr_plot = arr_value.copy()
    if len(arr.shape) == 3:
        arr_plot = arr_plot[:,:,0]
    return np.flipud(arr_plot.T)


""" ====================== FOURIER TRANSFORMS  ======================"""


@primitive
def my_fft(x):    
    """ 
    Wrapper for numpy's FFT, so I can add a primitive to it
        FFT(x) is like a DFT matrix (D) dot with x
    """
    return np.fft.fft(x)


def fft_grad(g, ans, x):
    """ 
    Define the jacobian-vector product of my_fft(x)
        The gradient of FFT times g is the vjp
        ans = fft(x) = D @ x
        jvp(fft(x))(g) = d{fft}/d{x} @ g
                       = D @ g
        Therefore, it looks like the FFT of g
    """
    return np.fft.fft(g)

defjvp(my_fft, fft_grad)


def get_spectrum(series, dt):
    """ Get FFT of series """

    steps = len(series)
    times = np.arange(steps) * dt

    # reshape to be able to multiply by hamming window
    series = series.reshape((steps, -1))

    # multiply with hamming window to get rid of numerical errors
    hamming_window = np.hamming(steps).reshape((steps, 1))
    signal_f = my_fft(hamming_window * series)

    freqs = np.fft.fftfreq(steps, d=dt)
    return freqs, signal_f


def get_max_power_freq(series, dt):

    freqs, signal_f = get_spectrum(series, dt)
    return freqs[np.argmax(signal_f)]


def get_spectral_power(series, dt):

    freqs, signal_f = get_spectrum(series, dt)
    return freqs, np.square(np.abs(signal_f))


def plot_spectral_power(series, dt, f_top=2e14):
    steps = len(series)
    freqs, signal_f_power = get_spectral_power(series, dt)

    # only plot half (other is redundant)
    plt.plot(freqs[:steps//2], signal_f_power[:steps//2])
    plt.xlim([0, f_top])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power (|signal|^2)')
    plt.show()

""" Utilities for plotting and visualization """

def real(val, outline=None, ax=None, cbar=False, cmap='RdBu', outline_alpha=0.5):
    """Plots the real part of 'val', optionally overlaying an outline of 'outline'
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    
    vmax = np.abs(val).max()
    h = ax.imshow(np.real(val.T), cmap=cmap, origin='lower', vmin=-vmax, vmax=vmax)
    
    if outline is not None:
        ax.contour(outline.T, 0, colors='k', alpha=outline_alpha)
    
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if cbar:
        plt.colorbar(h, ax=ax)
    
    return ax

def abs(val, outline=None, ax=None, cbar=False, cmap='magma', outline_alpha=0.5, outline_val=None):
    """Plots the absolute value of 'val', optionally overlaying an outline of 'outline'
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)      
    
    vmax = np.abs(val).max()
    h = ax.imshow(np.abs(val.T), cmap=cmap, origin='lower', vmin=0, vmax=vmax)
    
    if outline_val is None and outline is not None: outline_val = 0.5*(outline.min()+outline.max())
    if outline is not None:
        ax.contour(outline.T, [outline_val], colors='w', alpha=outline_alpha)
    
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if cbar:
        plt.colorbar(h, ax=ax)
    
    return ax


