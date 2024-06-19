import jax.numpy as jnp
import numpy as np
from jax import jit, ops
from jax import lax
import jax
import jax.random as random
from qs.utils import State
from qs.utils import advance_PRNG_state
from .sampler import Sampler
from qs.models.vmc_fermions import VMC_fermions
from qs.utils.parameter import Parameter
from jax import vmap


class Metropolis_fermions(Sampler):
    def __init__(
        self,
        alg_inst,
        hamiltonian,
        rng,
        scale,
        n_particles,
        dim,
        seed,
        log,
        logger=None,
        logger_level="INFO",
        backend="Numpy",
    ):
        # Initialize the VMC instance
        # Initialize Metropolis-specific variables
        self.step_method = self.step
        self._seed = seed
        self._N = n_particles//2
        self._dim = dim
        
        super().__init__(alg_inst, hamiltonian, log, rng, scale, logger, backend)

        self.alg_inst = alg_inst

        self.n_up , self.n_down = self.alg_inst.generate_quantum_states(self._N*2, self._dim)
        self.n_up = np.array(self.n_up)
        self.n_down = np.array(self.n_down)

    

    def step(self, wf_squared, state, seed):
        """One step of the random walk Metropolis algorithm."""
        """
        initial_positions_up = state.positions[:self._N]
        initial_positions_down = state.positions[self._N:]

        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)
        # Generate a proposal move

        proposed_positions_tot_up = rng.normal(loc=initial_positions_up , scale=self.scale)
        proposed_positions_tot_down = rng.normal(loc=initial_positions_down , scale=self.scale)

        initial_positions_up = np.array(initial_positions_up)
        initial_positions_down = np.array(initial_positions_down)
        proposed_positions_tot_up = np.array(proposed_positions_tot_up)
        proposed_positions_tot_down = np.array(proposed_positions_tot_down)

        D_up = np.array(state.D_up)
        D_down = np.array(state.D_down)
        D_inv_up = np.array(state.D_inv_up)
        D_inv_down = np.array(state.D_inv_down)

        n_accepted_up = state.n_accepted_up
        n_accepted_down = state.n_accepted_down

        """
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        initial_positions = state.positions
        proposed_positions_tot = rng.normal(loc=initial_positions, scale=self.scale)



        initial_positions = np.array(initial_positions)
        proposed_positions_tot = np.array(proposed_positions_tot)

        
        n_accepted = state.n_accepted

        #IT WAS ONLY SELF._N BEFORE
        for i in range(self._N*2):

            initial_positions_ = np.array(initial_positions)    
            """
            
            proposed_positions_up = initial_positions_up.copy()
            proposed_positions_down = initial_positions_down.copy()
            

            proposed_positions_up[i , :] = proposed_positions_tot_up[i , :]
            proposed_positions_down[i , :] = proposed_positions_tot_down[i , :]

            prop_line_up = proposed_positions_up[i , :]
            prop_line_down = proposed_positions_down[i , :]
            # Calculate log probability densities for current and proposed positions

            wf_mapped_up = self.single_particle_all_quantum_numbers(prop_line_up, self.n_up)
            wf_mapped_down = self.single_particle_all_quantum_numbers(prop_line_down, self.n_down)
            
            # Calculate acceptance probability in log domain
            accept_prob_up = self.backend.sum(wf_mapped_up * D_inv_up[i, :])
            accept_prob_down = self.backend.sum(wf_mapped_down*D_inv_down[i, :])

            
            # Decide on acceptance
            accept_up = np.full(self._N, False, dtype=bool)
            accept_down = np.full(self._N, False, dtype=bool)


            accept_up[i] = rng.random() < accept_prob_up
            accept_down[i] = rng.random() < accept_prob_down
            
            
            #breakpoint()
            new_positions_up = self.accept_func(
                accept=accept_up,
                initial_positions=initial_positions_up,
                proposed_positions=proposed_positions_up,
            )

            new_positions_down = self.accept_func(
                accept=accept_down,
                initial_positions=initial_positions_down,
                proposed_positions=proposed_positions_down,
            )

            
            
            if accept_up[i]:
                initial_positions_up = new_positions_up
                D_up = self.update_D(D_up , prop_line_up , self.n_up , i)
                D_inv_up = jnp.linalg.inv( D_up)#self.alg_inst.update_inverse(D_up , D_inv_up , i)
                n_accepted_up += 1

                #print("Accepted up")
                

            if accept_down[i]:
                
                initial_positions_down = new_positions_down
                D_down = self.update_D(D_down , prop_line_down , self.n_down , i)
                D_inv_down = jnp.linalg.inv(D_down) #self.alg_inst.update_inverse(D_down , D_inv_down , i)
                n_accepted_down += 1

                #print("Accepted down")
            """

            proposed_positions = initial_positions.copy()
            proposed_positions[i , :] = proposed_positions_tot[i , :]
                

            prob_current = wf_squared(initial_positions)
            prob_proposed = wf_squared(proposed_positions)

            accept_prob = prob_proposed / prob_current



            accept = np.full(self._N*2, False, dtype=bool)
            

            accept[i] = rng.random() < accept_prob

            new_positions = self.accept_func(
                accept=accept,
                initial_positions=initial_positions,
                proposed_positions=proposed_positions,
            )
            
            if accept[i]:
                initial_positions = new_positions
                n_accepted += 1
            

            #breakpoint()
        """"
        new_positions_tot = np.vstack((initial_positions_up, initial_positions_down))

        """

        new_positions_tot = initial_positions
        # Create new state by updating state variables.


        state.delta += 1
        """
        state.n_accepted_up = n_accepted_up
        state.n_accepted_down = n_accepted_down
        """
        state.n_accepted = n_accepted #n_accepted_up + n_accepted_down
        state.positions = new_positions_tot

        """
        state.D_up = D_up
        state.D_down = D_down
        state.D_inv_up = D_inv_up
        state.D_inv_down = D_inv_down
        """
        
    def accept_func(
        self,
        accept,
        initial_positions,
        proposed_positions,
    ):
        
        accept = accept.reshape(-1, 1)
        # accept is a boolean array, so you can use it to index directly
        new_positions = np.where(accept, proposed_positions, initial_positions)
        

        return new_positions
    

    def single_particle_all_quantum_numbers(self, ri, n):

        def apply_single_particle_wavefunction(n_row):
            
            return self.alg_inst.single_particle_wavefunction(ri, n_row)
        

        # Use vmap to apply the function across each row of n
        vmapped_wf = vmap(apply_single_particle_wavefunction, in_axes=0)

        # Execute vmap across all particles

        wf_mapped = vmapped_wf(n)

        return wf_mapped
        

    
    def update_D(self, D_old, prop_line, n, index):
       
        new_column = self.single_particle_all_quantum_numbers(prop_line, n)

        # Update D matrix with the new values at the specified column index
        #D_new = D_old.at[:, index].set(new_column.squeeze())

        D_new = D_old.copy()


        D_new[: , index] = new_column.squeeze()

        return D_new


