import numpy as np
import autograd.numpy as npa
#from autograd import jacobian
import matplotlib.pyplot as plt
#from scipy.optimize import approx_fprime as jacobian
import ceviche
from ceviche import jacobian
from ceviche.constants import ETA_0, C_0, MU_0, EPSILON_0
import Yee_TM_FDFD_ceviche, Yee_TE_FDFD_ceviche, Yee_TE_FDFD_Gamma_ceviche
import random
import sys

# calculate and return the local density of states (dos) for each complex 
# frequency at which the dos is being evaluated
def source_dos(source, Exs, Eys, Ezs, dl, Num_Poles, ii):
    dos = 0
    Polefactor = 1./np.sin(np.pi/(2.*Num_Poles))
    dos += dl**2 * 0.5 * npa.imag((np.exp(1j*(np.pi+2.*ii*np.pi)/(2.*Num_Poles))/Polefactor*npa.sum(npa.conj(source) * (Exs+Eys+Ezs))))
    return dos

# rescale a vector of continuous variables representing the degrees of freedom 
# in the grid, and that all lie between 0 and 1, to permittivity values 
def scale_dof_to_eps(dof, epsmin, epsmax):
    return epsmin + dof * (epsmax-epsmin)

# construct the material (fill in the appropriate permittivity at each pixel 
# in the grid)
def eps_parametrization(dof, epsmin, epsmax, designMask, epsBkg):

    eps = scale_dof_to_eps(dof, epsmin, epsmax) * designMask + epsBkg * (1-designMask)
    return eps


# construct material and calculate dos
def dos_objective(dof, epsval, designMask, dl, source, simsx, simsy, simsz, kx, ky, Mx, My, Num_Poles, polarization):
    Nx,Ny = source.shape
    dof = dof.reshape((Nx,Ny))
        
    epsBkg = np.ones((Nx,Ny), dtype=complex)
    eps = eps_parametrization(dof, 1.0, epsval, designMask, epsBkg)
    Ezs = []
    dos = 0
    xl = np.linspace(0,Mx-1,Mx)/Mx
    yl = np.linspace(0,My-1,My)/My
    xg, yg = np.meshgrid(xl,yl)
    if Mx == 1:
        xg = 0
    if My == 1:
        yg = 0
    for ii in range(max([len(simsx),len(simsy),len(simsz)])):
        if polarization == 'TE':
            simsx[ii].eps_r = eps
            simsy[ii].eps_r = eps
        if polarization == 'TM':
            simsz[ii].eps_r = eps
        if polarization == 'TE':
            Ex1,_,_ = simsx[ii].solve(source*np.exp(1j*(kx[ii]*xg+ky[ii]*yg)))
            _,Ey1,_ = simsy[ii].solve(source*np.exp(1j*(kx[ii]*xg+ky[ii]*yg)))
        else:
            Ex1 = 0
            Ey1 = 0
        if polarization == 'TM':
            _,_,Ez1 = simsz[ii].solve(source*np.exp(1j*(kx[ii]*xg+ky[ii]*yg)))
        else:
            Ez1 = 0
        dos += source_dos(source*np.exp(1j*(kx[ii]*xg+ky[ii]*yg)), Ex1, Ey1, Ez1, dl, Num_Poles, ii//(Mx*My))
    return dos


def designdof_dos_objective(designdof, designgrad, epsval, designMask, dl, source, omega, Num_Poles, gap, epsVac, Npml, opt_data, Mx, My, reciprocal_lattice, kpointgrid, polarization):
    """
    optimization objective to be used with NLOPT
    opt_data is dictionary with auxiliary info such as current iteration number and output base
    """
    
    omegas = []
    
    xl = np.linspace(0,Mx-1,Mx)
    yl = np.linspace(0,My-1,My)
    xg, yg = np.meshgrid(xl,yl)
    simsx = []
    simsy = []
    simsz = []
    kx = []
    ky = []
    ind = 0
    for nn in range(Num_Poles):
        omegas += [omega * (1-np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*Num_Poles))/2.*gap)]
        if reciprocal_lattice=='fullBZ':
            for nx in range(-int(np.ceil(Mx/2))+1,int(np.floor(Mx/2))+1):
                for ny in range(-int(np.ceil(My/2))+1,int(np.floor(My/2))+1):
                    kx += [2*np.pi*nx/Mx]
                    ky += [2*np.pi*ny/My] 
                    if polarization == 'TE':
                        simsx += [Yee_TE_FDFD_ceviche.fdfd_TEx(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml,Npml])]
                        simsy += [Yee_TE_FDFD_ceviche.fdfd_TEy(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml,Npml])]
                    if polarization == 'TM':
                        simsz += [Yee_TM_FDFD_ceviche.fdfd_TM(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml,Npml])]
                    ind += 1
        elif reciprocal_lattice=='Gamma':
            kx += [0]
            ky += [0]
            Mx = 1 
            My = 1
            if polarization == 'TE':
                simsx += [Yee_TE_FDFD_Gamma_ceviche.fdfd_TEx(omegas[nn], dl, epsVac, [Npml,Npml])]
                simsy += [Yee_TE_FDFD_Gamma_ceviche.fdfd_TEy(omegas[nn], dl, epsVac, [Npml,Npml])]
            if polarization == 'TM':
                simsz += [ceviche.fdfd_ez(omegas[nn], dl, epsVac, [Npml,Npml])]
            ind += 1
        elif reciprocal_lattice=='file':
            for nxy in range(kpointgrid.shape[0]):
                kx += [2*np.pi*kpointgrid[nxy,0]]
                ky += [2*np.pi*kpointgrid[nxy,1]]
                if polarization == 'TE':
                    simsx += [Yee_TE_FDFD_ceviche.fdfd_TEx(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml, Npml])]
                    simsy += [Yee_TE_FDFD_ceviche.fdfd_TEy(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml, Npml])]
                if polarization == 'TM':
                    simsz += [Yee_TM_FDFD_ceviche.fdfd_TM(omegas[nn], dl, kx[ind], ky[ind], epsVac, [Npml, Npml])]
                ind += 1
    Nx,Ny = source.shape
    dof = np.zeros((Nx,Ny))
    dof[designMask] = designdof[:]
    objfunc = lambda d: dos_objective(d, epsval, designMask, dl, source, simsx, simsy, simsz, kx, ky, Mx, My, Num_Poles, polarization)
    obj = objfunc(dof.flatten())
    opt_data['count'] += 1
    print('at iteration #', opt_data['count'], 'the dos value is', obj, ' with enhancement', obj/opt_data['vac_dos'], flush=True)
    if opt_data['count'] % opt_data['output_base'] == 0:
        np.savetxt(opt_data['name']+'_dof'+str(opt_data['count'])+'.txt', designdof[:])

    if len(designgrad)>0:
        jac_objfunc = jacobian(objfunc, mode='reverse')
        fullgrad = jac_objfunc(dof.flatten())
        designgrad[:] = np.reshape(fullgrad, (Nx,Ny))[designMask]

    return obj

