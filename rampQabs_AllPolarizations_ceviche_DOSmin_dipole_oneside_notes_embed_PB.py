import numpy as np
import autograd.numpy as npa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.colors import DivergingNorm

import time,sys,argparse

from objective_AllPolarizations_ceviche_DOS_notes_PB import designdof_dos_objective, eps_parametrization
import ceviche
import Yee_TM_FDFD_ceviche, Yee_TE_FDFD_ceviche, Yee_TE_FDFD_Gamma_ceviche
from ceviche.constants import C_0, ETA_0, MU_0, EPSILON_0

import time,sys,argparse

import nlopt


parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)

parser.add_argument('-gap_start',action='store',type=float,default=0.1)
parser.add_argument('-gap_end',action='store',type=float,default=0.3)
parser.add_argument('-gap_num',action='store',type=int,default=3)

parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)

###design area size, design area is rectangular with central rectangular hole where the dipole lives###
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)
parser.add_argument('-Num_Poles', action='store', type=int, default=1)
parser.add_argument('-omega_factor', action='store', type=float, default=0.4)
parser.add_argument('-reciprocal_lattice', action='store', type=str, default='fullBZ')
parser.add_argument('-reciprocal_lattice_file', action='store', type=str, default='kpoint_file.txt')
parser.add_argument('-polarization', action='store', type=str, default='TM')

parser.add_argument('-emitter_x',action='store',type=float,default=0.05)
parser.add_argument('-emitter_y',action='store',type=float,default=0.05)

#separation between pml inner boundary and source walls
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-init_type',action='store',type=str,default='vac')
parser.add_argument('-init_file',action='store',type=str,default='test.txt')
parser.add_argument('-maxeval',action='store',type=int,default=10000)
parser.add_argument('-output_base',action='store',type=int,default=10)
parser.add_argument('-name',action='store',type=str,default='test')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

opt_data = {'count':0, 'output_base':args.output_base, 'name':args.name}

# construct base frequency omega_0 and grid increments
k = 2*np.pi/args.wavelength
omega = args.omega_factor * C_0 * k
dl = 1.0/args.gpr

# construct design region
Mx = int(args.design_x/dl)
My = int(args.design_y/dl)
Npml = int(np.round(args.pml_thick/dl))
Npmlsep = int(np.round(args.pml_sep/dl))
Emitterx = int(np.round(args.emitter_x / dl))
Emittery = int(np.round(args.emitter_y / dl))

Nx = Mx + 2*(Npmlsep+Npml)
Ny = My + 2*(Npmlsep+Npml)

design_mask = np.zeros((Nx,Ny), dtype=bool)
design_mask[Npml+Npmlsep:Npml+Npmlsep+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True
    
chi = args.ReChi - 1j*args.ImChi #ceviche has +iwt time convention
epsval = 1.0 + chi
print('epsval', epsval, flush=True)
emitter_mask = np.zeros((Nx,Ny), dtype=bool)
emitter_mask[Npml+Npmlsep+(Mx-Emitterx)//2:Npml+Npmlsep+(Mx-Emitterx)//2+Emitterx,Npml+Npmlsep+(My-Emittery)//2:Npml+Npmlsep+(My-Emittery)//2+Emittery] = True

#set TE dipole source
source = np.zeros((Nx,Ny), dtype=complex)
source[emitter_mask] = 1.0 / (dl*dl)

# initialize design region with material
ndof = np.sum(design_mask)
kpointgrid = 0
if args.init_type=='vac':
    designdof = np.zeros(ndof)
if args.init_type=='slab':
    designdof = np.ones(ndof)
if args.init_type=='stripes':
    designdof = np.ones(ndof)
    designdof = np.reshape(designdof,(args.gpr,args.gpr))
    designdof[::2,:] = 0
    designdof = designdof.flatten()
if args.init_type=='checkers':
    designdof = np.ones(ndof)
    designdof = np.reshape(designdof,(args.gpr,args.gpr))
    for ii in range(0,args.gpr,2):
        for jj in range(args.gpr):
            if jj % 2 == 0:
                designdof[ii,jj] = 0
            else:
                designdof[ii-1,jj] = 0
    if args.gpr % 2 == 1:
        designdof[-1,1::2] = 1.
    designdof = designdof.flatten()
if args.init_type=='half':
    designdof = 0.5*np.ones(ndof)
if args.init_type=='rand':
    designdof = np.random.rand(ndof)
if args.init_type=='file':
    designdof = np.loadtxt(args.init_file)
if args.reciprocal_lattice=='file':
    kpointgrid = np.loadtxt(args.reciprocal_lattice_file)

epsVac = np.ones((Nx,Ny), dtype=complex)
xl = np.linspace(0,Mx-1,Mx)/Mx
yl = np.linspace(0,My-1,My)/My
xg, yg = np.meshgrid(xl,yl)
vac_dos = 0

if args.reciprocal_lattice=='fullBZ':
    for nx in range(-int(np.ceil(Mx/2))+1,int(np.floor(Mx/2))+1):
        for ny in range(-int(np.ceil(My/2))+1,int(np.floor(My/2))+1):
            kx = 2*np.pi/Mx*nx
            ky = 2*np.pi/My*ny
            if args.polarization == 'TE':
                sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEx(omega, dl, kx, ky, epsVac, [Npml,Npml])
                vac_fieldx,_,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEy(omega, dl, kx, ky, epsVac, [Npml,Npml])
                _,vac_fieldy,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
            else:
                vac_fieldx = 0
                vac_fieldy = 0
            if args.polarization == 'TM':
                sim_vac = Yee_TM_FDFD_ceviche.fdfd_TM(omega, dl, kx, ky, epsVac, [Npml,Npml])
                _,_,vac_fieldz = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
            else:
                vac_fieldz = 0
            vac_dos += np.real(np.sum(np.conj(source)*(vac_fieldx+vac_fieldy+vac_fieldz)*np.exp(-1j*(kx*xg+ky*yg)))) * 0.5 * dl**2
elif args.reciprocal_lattice=='Gamma':
    if args.polarization == 'TE':
        sim_vac = Yee_TE_FDFD_Gamma_ceviche.fdfd_TEx(omega, dl, epsVac, [Npml,Npml])
        vac_fieldx,_,_ = sim_vac.solve(source)
        sim_vac = Yee_TE_FDFD_Gamma_ceviche.fdfd_TEy(omega, dl, epsVac, [Npml,Npml])
        _,vac_fieldy,_ = sim_vac.solve(source)
    else:
        vac_fieldx = 0
        vac_fieldy = 0
    if args.polarization == 'TM':
        sim_vac = ceviche.fdfd_ez(omega, dl, epsVac, [Npml,Npml])
        _,_,vac_fieldz = sim_vac.solve(source)
    else:
        vac_fieldz = 0
    vac_dos += np.real(np.sum(np.conj(source)*(vac_fieldx+vac_fieldy+vac_fieldz))) * 0.5 * dl**2
elif args.reciprocal_lattice=='file':
    for nn in range(kpointgrid.shape[0]):
        kx = 2*np.pi*kpointgrid[nn,0]
        ky = 2*np.pi*kpointgrid[nn,1]
        if args.polarization == 'TE':
            sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEx(omega, dl, kx, ky, epsVac, [Npml,Npml])
            vac_fieldx,_,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
            sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEy(omega, dl, kx, ky, epsVac, [Npml,Npml])
            _,vac_fieldy,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
        else:
            vac_fieldx = 0
            vac_fieldy = 0
        if args.polarization == 'TM':
            sim_vac = Yee_TM_FDFD_ceviche.fdfd_TM(omega, dl, kx, ky, epsVac, [Npml,Npml])
            _,_,vac_fieldz = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
        else:
            vac_fieldz = 0
        vac_dos += np.real(np.sum(np.conj(source)*(vac_fieldx+vac_fieldy+vac_fieldz)*np.exp(-1j*(kx*xg+ky*yg)))) * 0.5 * dl**2

opt_data['vac_dos'] = vac_dos
print('vacuum DOS', vac_dos)
    
#check configuration
config = np.zeros((Nx,Ny))
config[design_mask] = 1.0
config[emitter_mask] = 2.0
plt.imshow(config)
plt.savefig(args.name+'_check_config.png')
   
gaplist = np.linspace(args.gap_start, args.gap_end, args.gap_num)
gaplist = gaplist.tolist()

# run optimization for various quality factors (quality factors represent
# domega - the width of the frequency window over which to establish a photonic
# bandgap)
for gap in gaplist:
    print('at gap', gap)
    opt_data['count'] = 0 #refresh the iteration count
    
    omega_gap = omega * (1-1j/2*gap)
    opt_data['name'] = args.name + f'_gap{gap:.1e}' 

    omegas = []
    vac_dos = 0
    Polefactor = 1./np.sin(np.pi/(2.*args.Num_Poles))
    for nn in range(args.Num_Poles):
        omegas += [omega * (1-np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))/2.*gap)]
        if args.reciprocal_lattice=='fullBZ':
            for nx in range(-int(np.ceil(Mx/2))+1,int(np.floor(Mx/2))+1):
                for ny in range(-int(np.ceil(My/2))+1,int(np.floor(My/2))+1):
                    kx = 2*np.pi*nx/Mx
                    ky = 2*np.pi*ny/My
                    if args.polarization == 'TE':
                        sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEx(omegas[nn], dl, kx,ky, epsVac, [Npml,Npml])
                        vac_fieldx,_,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                        sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEy(omegas[nn], dl, kx,ky, epsVac, [Npml,Npml])
                        _,vac_fieldy,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                    else:
                        vac_fieldx = 0
                        vac_fieldy = 0
                    if args.polarization == 'TM':
                        sim_vac = Yee_TM_FDFD_ceviche.fdfd_TM(omegas[nn], dl, kx,ky, epsVac, [Npml,Npml])
                        _,_,vac_fieldz = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                    else:
                        vac_fieldz = 0
                    vac_dos += dl**2 * 0.5 * np.imag((np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))/Polefactor*np.sum(np.conj(source) * (vac_fieldx+vac_fieldy+vac_fieldz)*np.exp(-1j*(kx*xg+ky*yg)))))
        elif args.reciprocal_lattice=='Gamma':
            if args.polarization == 'TE':
                sim_vac = Yee_TE_FDFD_Gamma_ceviche.fdfd_TEx(omegas[nn], dl, epsVac, [Npml,Npml])
                vac_fieldx,_,_ = sim_vac.solve(source)
                sim_vac = Yee_TE_FDFD_Gamma_ceviche.fdfd_TEy(omegas[nn], dl, epsVac, [Npml,Npml])
                _,vac_fieldy,_ = sim_vac.solve(source)
            else:
                vac_fieldx = 0
                vac_fieldy = 0
            if args.polarization == 'TM':
                sim_vac = ceviche.fdfd_ez(omegas[nn], dl, epsVac, [Npml,Npml])
                _,_,vac_fieldz = sim_vac.solve(source)
            else:
                vac_fieldz = 0
            vac_dos += dl**2 * 0.5 * np.imag((np.exp(1j*(np.pi+nn*2.*np.pi)/(2.*args.Num_Poles))/Polefactor*np.sum(np.conj(source)*(vac_fieldx+vac_fieldy+vac_fieldz))))
        elif args.reciprocal_lattice=='file':
            for nxy in range(kpointgrid.shape[0]):
                kx = 2*np.pi*kpointgrid[nxy,0]
                ky = 2*np.pi*kpointgrid[nxy,1]
                if args.polarization == 'TE':
                    sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEx(omegas[nn], dl, kx, ky, epsVac, [Npml,Npml])
                    vac_fieldx,_,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                    sim_vac = Yee_TE_FDFD_ceviche.fdfd_TEy(omegas[nn], dl, kx, ky, epsVac, [Npml,Npml])
                    _,vac_fieldy,_ = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                else:
                    vac_fieldx = 0
                    vac_fieldy = 0
                if args.polarization == 'TM':
                    sim_vac = Yee_TM_FDFD_ceviche.fdfd_TM(omegas[nn], dl, kx, ky, epsVac, [Npml,Npml])
                    _,_,vac_fieldz = sim_vac.solve(source*np.exp(1j*(kx*xg+ky*yg)))
                else:
                    vac_fieldz = 0
                vac_dos += dl**2 * 0.5 * np.imag((np.exp(1j*(np.pi+nn*2*np.pi)/(2.*args.Num_Poles))/Polefactor*np.sum(np.conj(source) * (vac_fieldx+vac_fieldy+vac_fieldz) * np.exp(-1j*(kx*xg+ky*yg)))))
    opt_data['vac_dos'] = vac_dos
    optfunc = lambda dof, grad: designdof_dos_objective(dof, grad, epsval, design_mask, dl, source, omega, args.Num_Poles, gap, epsVac, Npml, opt_data, Mx, My, args.reciprocal_lattice, kpointgrid, args.polarization)
    lb = np.zeros(ndof)
    ub = np.ones(ndof)

    opt = nlopt.opt(nlopt.LD_MMA, int(ndof))
        
    opt.set_xtol_rel(1e-8)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(args.maxeval)

    opt.set_min_objective(optfunc)
    designdof = opt.optimize(designdof.flatten())
        
    min_dos = opt.last_optimum_value()
    min_enh = min_dos / vac_dos
    print('vacuum DOS', vac_dos)

    print(f'Gap{gap:.1e} best DOS and enhancement found via topology optimization', min_dos, min_enh)
    np.savetxt(opt_data['name'] + '_optdof.txt', designdof)

    opt_design = np.zeros((Nx,Ny))
    opt_design[design_mask] = designdof
    plt.figure()
    plt.imshow(np.reshape(opt_design[Npml+Npmlsep:Npml+Npmlsep+Mx,Npml+Npmlsep:Npml+Npmlsep+My], (Mx,My)))
    plt.savefig(opt_data['name']+'_opt_design.png')
   
