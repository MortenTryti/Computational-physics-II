import sys
import jax
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append("/mnt/c/Users/annar/OneDrive/Desktop/FYS4411/Repo/src")

from qs import quantum_state
import config

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

sample_values = np.array([2**12 , 2**13 , 2**14 , 2**15 , 2**16 , 2**17, 2**18])
times_ana = []
times_jax = []

def setup_and_train(backend, n_sample):
    start_time = time.perf_counter()
    
    system = quantum_state.QS(
        backend=backend,
        log=True,
        logger_level="INFO",
        seed=config.seed,
        alpha=config.alpha,
        beta=config.beta,
        time_step=config.time_step,
        diffusion_coeff=config.diffusion_coeff
    )
    
    # Adjust parameters based on sample_size if necessary
    
    # Setup and training process
    system.set_wf(config.wf_type, config.nparticles, config.dim)
    system.set_hamiltonian(type_=config.hamiltonian, int_type=config.interaction, omega=1.0)
    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_optimizer(optimizer=config.optimizer, eta=config.eta)
    system.train(max_iter=config.training_cycles, batch_size=config.batch_size, seed=config.seed)

    results, _, _ = system.sample(n_sample, nchains=config.nchains, seed=config.seed)

    
    end_time = time.perf_counter()
    return end_time - start_time

for n_samples in sample_values:
    times_ana.append(setup_and_train("numpy", n_samples))
    times_jax.append(setup_and_train("jax", n_samples))

print("Execution times for numpy:", times_ana)
print("Execution times for jax:", times_jax)

times_difference = np.array(times_ana) - np.array(times_jax)


np.savetxt(f"data_analysis/sample_values{config.particle_type}_{config.nparticles}.dat", sample_values)
np.savetxt(f"data_analysis/times_ana{config.particle_type}_{config.nparticles}.dat", times_ana)
np.savetxt(f"data_analysis/times_jax{config.particle_type}_{config.nparticles}.dat", times_jax)
np.savetxt(f"data_analysis/times_difference{config.particle_type}_{config.nparticles}.dat", times_difference)


