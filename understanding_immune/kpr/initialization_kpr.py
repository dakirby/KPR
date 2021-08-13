""" initialization script, import and copy data to like named variables in other modules.
"""
# necessary imports to define following functions
import random
from phievo.Networks import mutation

# ranges over which parmeters in classes_eds2 classes can vary. Defined in mutation.py
# NB use 1.0 etc to keep params real not int.

T=1.0 #typical time scale
C=1.0 #typical concentration
L=1.0 #typical size for diffusion
R=10000 # typical number of receptors

#dictionary key is the Class.attribute whose range is give an a real number or as a LIST
# indicating the interval over which it has to vary

dictionary_ranges={}
dictionary_ranges['Species.degradation']=1.0/T
dictionary_ranges['Species.diffusion']=L   # for ligands diffusion
dictionary_ranges['TModule.rate']=C/T
dictionary_ranges['TModule.basal']=0.0
dictionary_ranges['CorePromoter.delay']=0   # convert to int(T/dt) in run_evolution.py
dictionary_ranges['TFHill.hill']=5.0
dictionary_ranges['TFHill.threshold']=C
dictionary_ranges['PPI.association']=1.0/(C*T)
dictionary_ranges['PPI.disassociation']=1.0/T
dictionary_ranges['Phosphorylation.rate']=1.0/T
dictionary_ranges['Phosphorylation.hill']=5.0
dictionary_ranges['Phosphorylation.threshold']=C
dictionary_ranges['Phosphorylation.dephosphorylation']=1.0/T
dictionary_ranges['Simple_Phosphorylation.rate']=1.0/(C*T)
dictionary_ranges['Simple_Phosphorylation.spontaneous_dephospho']=1.0/T
dictionary_ranges['Initial_Concentration.concentration'] = C



#################################################################################
# names of c-code files needed by deriv2.py (. for extension only)
# normally need 'header', 'utilities' (loaded before python derived routines and
# ['fitness', 'geometry', 'init_history', 'input', 'integrator', 'main' ] placed afterwards
# skip by setting cfile[] = ' ' or ''

cfile = {}
cfile['header'] = 'header_kpr.h'
cfile['utilities'] = 'utilities_kpr.c'
cfile['fitness'] = 'fitness_kpr.c'
cfile['geometry'] = 'linear_geometry.c'
cfile['init_history'] = 'init_history_kpr.c'
cfile['integrator'] = 'integrator_kpr.c'
cfile['main'] = 'main_kpr.c'
cfile['input'] = 'input_kpr.c'

pfile = {}
pfile["deriv2"] = "deriv2_kpr"
pfile["interaction"] = "interaction_kpr"
#pfile["pretty_graph"] = "Immune.pretty_graph_2_pMHC"
pfile["plotdata"] = "plotdata_kpr"

#################################################################################
# mutation rates
dictionary_mutation={}

# Rates for nodes to add
dictionary_mutation['random_gene()']=0.01
dictionary_mutation['random_gene(\'TF\')']=0.01
dictionary_mutation['random_gene(\'Kinase\')']=0.01
dictionary_mutation['random_gene(\'Ligand\')']=0.01
dictionary_mutation['random_gene(\'Receptor\')']=0.01


dictionary_mutation['random_Interaction(\'TFHill\')']=0.00
dictionary_mutation['random_Interaction(\'PPI\')']=0.002
dictionary_mutation['random_Interaction(\'Phosphorylation\')']=0.002

# Rates for nodes to remove
dictionary_mutation['remove_Interaction(\'TFHill\')']=0.00
dictionary_mutation['remove_Interaction(\'PPI\')']=0.005
dictionary_mutation['remove_Interaction(\'CorePromoter\')']=0.0
dictionary_mutation['remove_Interaction(\'Phosphorylation\')']=0.005

# Rates to change parameters for a node
dictionary_mutation['mutate_Node(\'Species\')']=0.1
dictionary_mutation['mutate_Node(\'TFHill\')']=0.0
dictionary_mutation['mutate_Node(\'CorePromoter\')']=0.
dictionary_mutation['mutate_Node(\'TModule\')']=0.
dictionary_mutation['mutate_Node(\'PPI\')']=0.1
dictionary_mutation['mutate_Node(\'Phosphorylation\')']=0.1

#rates to change output tags.  See list_types_output array below
dictionary_mutation['random_add_output()']=0.0
dictionary_mutation['random_remove_output()']=0.0
dictionary_mutation['random_change_output()']=0.0


#############################################################################
# parameters in various modules, created as one dict, so that can then be passed as argument

# Needed in deriv2.py to create C program from network
prmt = {}
prmt['nstep'] =30000         #number of time steps
prmt['ncelltot']=1            #number of cells in an organism
prmt['nneighbor'] = 3 # must be >0, whatever geometry requires, even for ncelltot=1
prmt['ntries'] = 10    # number of initial conditions tried in C programs
prmt['dt'] = 0.05     # time step


# Generic parameters, transmitted to C as list or dictionary.
# dict encoded in C as: #define KEY value (KEY converted to CAPS)
# list encoded in C as: static double free_prmt[] = {}
# The dict is more readable, but no way to test in C if given key supplied.  So
# always include in C: #define NFREE_PRMT int. if(NFREE_PRMT) then use free_prmt[n]
# prmt['free_prmt'] = { 'step_egf_off':20 }  # beware of int vs double type
# prmt['free_prmt'] = [1,2]



# Needed in evolution_gill to define evol algorithm and create initial network
prmt['npopulation'] =50
prmt['ngeneration'] =10#301
prmt['tgeneration']=1.0       #initial generation time (for gillespie), reset during evolution
prmt['noutput']=1    # to define initial network
prmt['ninput']=1
prmt['freq_stat'] = 5     # print stats every freq_stat generations
prmt['frac_mutate'] = 0.5 #fraction of networks to mutate
prmt['redo'] = 1   # rerun the networks that do not change to compute fitness for different IC

# used in run_evolution,
prmt['nseed'] = 2#5  # number of times entire evol procedure repeated, see main program.
prmt['firstseed'] = 0  #first seed


prmt['multipro_level']=1
prmt['pareto']=0
prmt['npareto_functions'] = 2
prmt['rshare']= 0.0
prmt['plot']=0
prmt['langevin_noise']=0

## prmt['restart']['kgeneration'] tells phievo to save a complete generation
## at a given frequency in a restart_file. The algorithm can relaunched at
## backuped generation by turning  prmt['restart']['activated'] = True and
## setting the generation and the seed. If the two latters are not set, the
## algorithm restart from the highest generation  in the highest seed.
prmt['restart'] = {}
prmt['restart']['activated'] = False #indicate if you want to restart or not
prmt['restart']['freq'] = 50  # save population every freq generations
#prmt['restart']['seed'] =  0 # the directory of the population you want to restart from
#prmt['restart']['kgeneration'] = 50  # restart from After this generation number
#prmt['restart']['same_seed'] = True # Backup the random generator to the same seed

list_unremovable=['Input','Output']

# control the types of output species, default 'Species' set in classes_eds2.    This list targets
# effects of dictionary_mutation[random*output] events to certain types, when we let output tags
# move around
#
list_types_output=['Species']


# Two optional functions to add input species or output genes, with IO index starting from 0.
# Will overwrite default variants in evol_gillespie if supplied below

def init_network(seed=-1):
    if seed == -1:
        seed = int(random.random()*510)
    g = random.Random(seed)
    L = mutation.Mutable_Network(g)

    # the ligand (the pMHC on the antigen presenting cell)
    parameters=[['Ligand']]
    parameters.append(['Complexable'])
    parameters.append(['Input',0])
    Lig = L.new_Species(parameters)

    # the receptor (on the T cell)
    parameters = [['Receptor']]
    parameters.append(['Phosphorylable'])
    parameters.append(['Complexable'])
    R = L.new_Species(parameters)

    # a kinase.
    parameters = [['Kinase']]
    parameters.append(['Phosphorylable'])
    parameters.append(['Phospho', 0])
    K1 = L.new_Species(parameters)

    # a phosphatase.
    parameters = [['Phosphatase']]
    parameters.append(['Phosphorylable'])
    parameters.append(['Phospho', 0])
    P1 = L.new_Species(parameters)

    # the complex generated: it is a kinase that is phosphorylable
    parameters = [['Kinase']]
    parameters.append(['Phosphorylable'])
    parameters.append(['Phospho', 0])
    parameters.append(['pMHC']) # label pMHC to indicate it is in the cascade ???
    kappa = 1E-4
    ppi, C = L.new_PPI(Lig, R, kappa, kappa, parameters)

    # the downstream reporter molecule
    parameters = [['Degradable', 0.5]]
    parameters.append(['Phosphorylable'])
    M1 = L.new_Species(parameters)

    # Add phosphorylation of M1 by C
    M2, phospho = L.new_Phosphorylation(C, M1, 2.0, 0.5, 1.0, 3)
    M2.add_type(['Output',0])  # Sets the molecule to be read as output

    L.write_id()

    # checks consecutive numbering for IO species.
    L.verify_IO_numbers()
    return L


def fitness_treatment(population):
    """Function to change the fitness of the networks"""
    pass
    #for nnetwork in range(population.npopulation):
    #    population.genus[nnetwork].fitness=eval(population.genus[nnetwork].data_evolution[0])
