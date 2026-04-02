#!/bin/bash
#SBATCH --job-name=AllPolarizationsDOSmin_polarizationTM_gap0d1_omegafactor0d4_des1by1_chi7d9_emit1by1_maxeval20000_gpr10_Num_Poles10_stripes1
#SBATCH --output=AllPolarizations_DOSmin_dipole_polarizationTM_gap0d1_omegafactor0d4_des1by1_chi7d9_emit1by1_maxeval20000_gpr10_Num_Poles10_stripes1.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20-00:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --error=AllPolarizations_errormin_polarizationTM_gap0d1_omegafactor0d4_des1by1_chi7d9_emit1by1_maxeval20000_gpr10_Num_Poles10_stripes1.err
 
# need anaconda installed, then pip install ceviche and NLopt in an environment

# defines the program name to be run
prog=rampQabs_AllPolarizations_ceviche_DOSmin_dipole_oneside_notes_embed_PB.py

# input parameters for the calculation
wavelength=1.0
gap_start=0.1
gap_end=0.1
gap_num=1
Num_Poles=10
omega_factor=0.4
reciprocal_lattice='fullBZ' # options are fullBZ, Gamma, and file 
reciprocal_lattice_file='kpoint_file.txt'
polarization='TM' # options are TM and TE

ReChi=7.9
ImChi=0.0

# number of gridpoints per length equal to 1
gpr=10

# size of design region (in x and y)
design_x=1.0
design_y=1.0

# size of region of dipoles (in x and y)
emitter_x=1.0
emitter_y=1.0

# more simulation parameters (for convergence)
pml_thick=0.0
pml_sep=0.0

# initialization of design region with material
init_type='stripes'
init_file='run_polarizationTM_gap0d1_omegafactor0d4_maxeval20000_des1by1_chi7d9_Num_Poles10_AllPolarizations_stripes1.txt'

# number of iterations after which to output the current design
output_base=50

# maximum number of iterations
maxeval=20000

# name of ouput file
name='AllPolarizationsmin_dipole_polarizationTM_gap0d1_omegafactor0d4_emit1by1_maxeval20000_gpr10_oneside_des1by1_cav0d0_chi7d9_Num_Poles10_stripes1'


# run the program
python3 $prog -wavelength $wavelength -gap_start $gap_start -gap_end $gap_end -gap_num $gap_num -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -emitter_x $emitter_x -emitter_y $emitter_y -pml_thick $pml_thick -pml_sep $pml_sep -init_type $init_type -init_file $init_file -output_base $output_base -name $name -maxeval $maxeval -Num_Poles $Num_Poles -omega_factor $omega_factor -reciprocal_lattice $reciprocal_lattice -reciprocal_lattice_file $reciprocal_lattice_file -polarization $polarization >> AllPolarizations_DOSmin_dipole_polarizationTM_gap0d1_omegafactor0d4_oneside_chi7d9_des1by1_emit1by1_maxeval20000_gpr10_Num_Poles10_stripes1.txt

