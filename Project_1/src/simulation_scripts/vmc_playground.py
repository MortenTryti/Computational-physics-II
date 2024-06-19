import sys
import os
import time



# The following will add the path to the ../src directory, for any given laptop running the code
# Assuming the structure of the folders are the same as Daniel initially built (i.e /src is the parent of /simulation script etc.)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)


import jax


from qs import quantum_state
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""
All the parameters you want to change are contained in the file config.py

"""

# start the timer
start_time = time.time()

print("alpha = ", config.alpha)
# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
system = quantum_state.QS(
    backend=config.backend,
    log=True,
    logger_level="INFO",
    seed=config.seed,
    alpha=config.alpha,
    beta=config.beta,
    radius = config.radius,
    time_step=config.time_step,
    diffusion_coeff=config.diffusion_coeff
)


# set up the wave function with some of its properties 
system.set_wf(
    config.wf_type,
    config.nparticles,
    config.dim,
)
 

# choose the hamiltonian
system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)

# choose the sampler algorithm and scale
system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)


# choose the optimizer, learning rate, and other properties depending on the optimizer
system.set_optimizer(
    optimizer=config.optimizer,
    eta=config.eta,
)

print("System initialization: Complete..")

# train the system, meaning we find the optimal variational parameters for the wave function
alphas ,energies , cycles = system.train(
    max_iter=config.training_cycles,
    batch_size=config.batch_size,
    seed=config.seed,
)

# now we get the results or do whatever we want with them
results , sampled_positions , _  = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

end_time = time.time()
execution_time = end_time - start_time

sampled_positions = np.array(sampled_positions).reshape(-1, config.nparticles, config.dim)
sampled_positions = np.linalg.norm(sampled_positions, axis=2)

# display the results
print("Metrics: ", results)
print("Result Energy: ", results.energy)
print(f"Acceptance rate: {results.accept_rate}")

print(f"Execution time: {execution_time} seconds")



#plot alphas vs cycles

np.savetxt(f"data_analysis/alphas_{config.particle_type}_{config.nparticles}.dat", alphas)
np.savetxt(f"data_analysis/energies_{config.particle_type}_{config.nparticles}.dat", energies)
np.savetxt(f"data_analysis/cycles_{config.particle_type}_{config.nparticles}.dat", cycles)
np.savetxt(f"data_analysis/sampled_positions_{config.particle_type}_{config.nparticles}_{config.nsamples}.dat", sampled_positions)


