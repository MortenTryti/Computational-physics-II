import sys
import jax
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# The following will add the path to the ../src directory, for any given laptop running the code
# Assuming the structure of the folders are the same as Daniel initially built (i.e /src is the parent of /simulation script etc.)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from qs import quantum_state
import config

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

delta_t_values = np.array([1e-8 , 1e-5 , 1e-3 ,  1 , 1e3, 1e5 , 1e7])
alpha_values = np.array([ 0.2 , 0.3, 0.4, 0.5, 0.6,  0.8 , 1.0])
energy_ana = np.zeros((len(alpha_values), len(delta_t_values)))
energy_jax = np.zeros((len(alpha_values), len(delta_t_values)))


if config.mcmc_alg != "mh":
    raise ValueError("This script only supports Metropolis-Hastings sampler")

def setup_and_train(backend, delta_t , alpha_value = config.alpha):
    
    system = quantum_state.QS(
        backend=backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=alpha_value,
        beta=config.beta,
        time_step=delta_t,
        diffusion_coeff=config.diffusion_coeff
    )
    
    # Adjust parameters based on sample_size if necessary
    
    # Setup and training process
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
    system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

    results, _, _ = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

    
    
    return results.energy

for i, delta_t in enumerate(delta_t_values):
    for j, alpha in enumerate(alpha_values):
        # You might need to adjust the call to setup_and_train based on whether it's 'ana' or 'jax' backend
        energy_ana[j, i] = setup_and_train('numpy', delta_t, alpha)
        energy_jax[j, i] = setup_and_train('jax', delta_t, alpha)




np.savetxt(f"data_analysis/delta_t_values_{config.particle_type}_{config.nparticles}.dat", delta_t_values)
np.savetxt(f"data_analysis/alpha_values_{config.particle_type}_{config.nparticles}.dat", alpha_values)
np.savetxt(f"data_analysis/energy_ana_{config.particle_type}_{config.nparticles}.dat", energy_ana)
np.savetxt(f"data_analysis/energy_jax_{config.particle_type}_{config.nparticles}.dat", energy_jax)

