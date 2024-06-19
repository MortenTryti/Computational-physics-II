import jax 
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import device_get, ops
import itertools
from jax import lax
from jax import vmap

from qs.utils import (
    Parameter,
)  # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State
import pdb
from simulation_scripts import config


class VMC_fermions:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None,
        seed=None,
        logger_level="INFO",
        backend="numpy",
        alpha=None,
        beta=None,
        radius=None,
    ):
        self._configure_backend(backend)
        self.params = Parameter()
        self.log = log
        self._seed = seed
        self._logger = logger
        self._N = nparticles

        self._dim = dim
        self.radius = config.radius
        self.beta = beta
        self.rng = random.PRNGKey(self._seed)  # Initialize RNG with the provided seed

        if alpha:
            self._initialize_variational_params(alpha)
        else:
            self._initialize_variational_params()  # initialize the variational parameters (ALPHA)

        if beta:
            self.beta = beta


        self.n_up , self.n_down = self.generate_quantum_states(self._N, self._dim)

        self._max_degree = config.max_degree

        self.state = 0  # take a look at the qs.utils State class
        self._initialize_vars(nparticles, dim, log, logger, logger_level)

        if self._N % 2 != 0:
            raise ValueError("The number of particles must be even")
        

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self._logger.info(msg)



    def _configure_backend(self, backend):
        """
        Here we configure the backend, for example, numpy or jax
        You can use this to change the linear algebra methods or do just in time compilation.
        Note however that depending on how you build your code, you might not need this.
        """
        if backend == "numpy":
            raise ValueError("Numpy backend not supported for Fermions")
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg

            self._jit_functions()
        else:
            raise ValueError(f"Backend {self.backend} not supported")
        

    def _jit_functions(self):
        """
        Note there are other ways to jit functions. this is just one example.
        However, you should be careful with how you jit functions.
        They have to be pure functions, meaning they cannot have side effects (modify some state variable values outside its local environment)
        Take a close look at "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html"
        """

        functions_to_jit = [
            "wf_closure",
            "single_particle_wavefunction_closure",
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
        ]

        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self
    

    

    def generate_quantum_states(self, N_particles, n_dim):
        
        N = N_particles // 2 
        # Generate a reasonable range of quantum numbers for each dimension
        max_quantum_number = int(jnp.ceil(N ** (1/n_dim)))


        states = list(itertools.product(range(max_quantum_number), repeat=n_dim))
        
        # Sort states by the sum of their components (which corresponds to energy levels in an isotropic harmonic oscillator)
        states.sort(key=sum)

        states = states[:N]


        states_array = np.array(states)

        # I am filling up each level with 2 electrons before to move to the next level ##,
        # even though they have the same energy
        spin_up_states = np.copy(states_array)
        spin_down_states = np.copy(states_array)

        return spin_up_states ,spin_down_states



    def hermite_poly(self, max_degree, x, active_degree):

        def scan_fun(carry, i):
            H_n_minus_2, H_n_minus_1 = carry
            H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
            return (H_n_minus_1, H_n), H_n

        H_n_minus_2 = jnp.ones_like(x)
        H_n_minus_1 = 2 * x

        _, scanned_results = lax.scan(scan_fun, (H_n_minus_2, H_n_minus_1), jnp.arange(2, max_degree + 1))

        results = jnp.concatenate([jnp.array([H_n_minus_2, H_n_minus_1]), scanned_results])

        selected_result = results[active_degree]
        return selected_result
    




    def single_particle_wavefunction(self, r, n):
        """
        Helper for the single particle wave function

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha

        return self.single_particle_wavefunction_closure(r, n , alpha)

    
    def single_particle_wavefunction_closure(self, r, n, alpha):
        """
        
        In the log domain
        
        """
        scaled_r =jnp.sqrt(2 * alpha) * r
        n_ = n

        # Map over each quantum number
        hermite_values = vmap(self.hermite_poly, in_axes=(None, 0, 0))(self._max_degree, scaled_r, n)
        
        prod_hermite = jnp.prod(hermite_values)

        wf = prod_hermite * jnp.exp(-alpha * jnp.sum(r ** 2))

        
        return wf.squeeze()


    def pairwise_distances(self , r, epsilon):

        diff = r[:, None, :] - r[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + epsilon)

        return dist

    def jastrow_factor_f(self, dist, a, epsilon):

        safe_dist = dist + epsilon  # Prevent division by zero
        jastrow_f = jnp.where(dist <= a, 0.0, 1 - a / safe_dist)
        jastrow_f = jastrow_f.at[jnp.diag_indices_from(jastrow_f)].set(1.0)

        return jastrow_f
    


    def wf(self, r):
        """
        Helper for the wave function

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha

        return self.wf_closure(r, alpha)

    #it was wf_closure(self, r, n_up, n_down, alpha)
    def wf_closure(self , r,  alpha):

        N = self._N// 2

        r_up = r[:N]  # Contains the first half of the particles
        r_down = r[N:]  # Contains the second half

        n_up = self.n_up
        n_down = self.n_down


        outer_vmap = vmap(lambda single_r, n, alpha: vmap(lambda single_n: self.single_particle_wavefunction_closure(single_r, single_n, alpha))(n), in_axes=(0, None, None))

        D_up = outer_vmap(r_up, n_up, alpha).reshape( N , N ).T
        D_down = outer_vmap(r_down, n_down, alpha).reshape( N , N ).T #now they are how you would expect them to be

        slater_det_up = jnp.linalg.det(D_up)
        slater_det_down =jnp.linalg.det(D_down)

        epsilon = 1e-14

        distances = self.pairwise_distances(r, epsilon)
        jastrow_f = self.jastrow_factor_f(distances, self.radius, epsilon)

        f = jnp.prod(jastrow_f )
        
        wf = slater_det_up * slater_det_down * f

        

        return wf #,  D_up, D_down
        

    """
    def wf_training(self, r,  alpha):
        
        D_up , D_down = self.wf_closure(r, self.n_up , self.n_down , alpha )
        
        slater_det_up = jnp.linalg.det(D_up)
        slater_det_down =jnp.linalg.det(D_down)

        
        wf = slater_det_up * slater_det_down

        return wf
    """  

    def prob(self, r ):
            """
            Helper for the probability density

            OBS: We strongly recommend you work with the wavefunction in log domain.
            """
            alpha = self.params.get("alpha")  # Using Parameter.get to access alpha
            return self.prob_closure(r, alpha)
        
    
    def prob_closure(self, r,  alpha):
        """
        Return a function that computes |Ψ(alpha, r)|^2

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        #
        return (self.wf_closure(r , alpha))**2 # Since we're not in the log domain


#it was grad_wf(self, r , n)
    def grad_wf(self, r ):
            """
            Helper for the gradient of the wavefunction with respect to r

            OBS: We strongly recommend you work with the wavefunction in log domain.
            """
            alpha = self.params.get("alpha")  # Using Parameter.get to access alpha

            return self.grad_wf_closure(r,  alpha)


        #it was grad_wf_closure(self, r, n_up, n_down, alpha)
    def grad_wf_closure(self, r,  alpha):

        """
        Computes the gradient of the wavefunction with respect to r, but with jax.grad
        output is of shape (N_particles , n_dim)
        
        # Define a function to compute gradient per particle
        def grad_single_particle(ri, ni ,alpha):
            #taking the norm of the gradient
            return (jax.grad(self.single_particle_wavefunction_closure, argnums=0)(ri, ni, alpha)) 
        
        # Vectorize the gradient computation across all particles
        
        outer_vmap = vmap(lambda single_r, n, alpha: vmap(lambda single_n: grad_single_particle(single_r, single_n, alpha))(n), in_axes=(0, None, None))

        
        grad = outer_vmap(r, n , alpha)

        """
        

        grad = jax.grad(self.wf_closure , argnums=0)(r,  alpha) / self.wf_closure(r,  alpha)

        return grad
    



    def grads(self, r):
        """
        Helper for the gradient of the wavefunction with respect to the variational parameters

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """
        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha

        return self.grads_closure(r, alpha)


    def grads_closure(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters with JAX grad using Vmap.
        """

        # Define the gradient function for a single instance
        def single_grad(pos, var):

            grad = jax.grad(lambda a: (self.wf_closure(pos, a)))(var)

            return grad / self.wf_closure(pos, var)
        # Vectorize the gradient computation over the batch dimension
        batched_grad = jax.vmap(single_grad, (0, None), 0)

        # Compute gradients for the entire batch
        grad_alpha = batched_grad(r, alpha) 


        return self.backend.squeeze(grad_alpha) 
    

#it was laplacian(self, r , n )
    def laplacian(self, r  ):
        """
        Return a function that computes the laplacian of the wavefunction ∇^2 Ψ(r)

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        alpha = self.params.get("alpha")  # Using Parameter.get to access alpha
        #NEEDS TO BE CHANGED FOR THE NON BF CASE
        return self.laplacian_closure(r,alpha)

    #it was laplacian_closure(self, r, n_up, n_down, alpha)
    def laplacian_closure(self, r, alpha):
        """
        Computes the Laplacian of the wavefunction for each particle using JAX automatic differentiation.
        r: Position array of shape (n_particles, n_dimensions)
        n: Quantum numbers array for each particle
        alpha: Parameter of the wavefunction
        Outputs laplacian of shape (N_particles ,)
        

        # Define a function that computes the wavefunction and its Laplacian for a single particle
        def laplacian_single_particle(ri, ni, alpha):
            hessian_psi = (jax.hessian(self.single_particle_wavefunction_closure, argnums=0)(ri, ni, alpha))
            laplacian_single = jnp.trace(hessian_psi#.reshape(self._dim * (self._N//2) , self._dim * (self._N//2)))
 
            
            return laplacian_single

        # Vectorize the computation across all particles


        outer_vmap = vmap(lambda single_r, n, alpha: vmap(lambda single_n: laplacian_single_particle(single_r, single_n, alpha))(n), in_axes=(0, None, None))

        laplacian = outer_vmap(r, n , alpha) 

        """
        r_ = r
        hessian_psi =jax.hessian(self.wf_closure, argnums=0)(r,  alpha)
        laplacian = jnp.trace(hessian_psi.reshape(self._dim * self._N , self._dim * self._N)) / self.wf_closure(r,  alpha)


        
        return laplacian
    
    """
    def reconstruct_array(self, B, col_i, i):
        # B is the array with the i-th column removed
        # col_i is the i-th column (as a 2D array of shape (m, 1))
        # i is the index at which the column was originally located
        
        # Get the number of columns in B
        num_cols = B.shape[1]
        
        # Check if the column was originally the last one
        if i == num_cols:
            # If col_i was the last column, just concatenate it at the end
            return jnp.hstack((B, col_i))
        else:
            # Otherwise, split B into the left and right parts and insert col_i in between
            left_part = B[:, :i]  # Columns before the i-th
            right_part = B[:, i:] # Columns from the i-th onward (originally i+1 onward in the full array)
            return jnp.hstack((left_part, col_i, right_part))


    def update_inverse(self,  D_new , D_inv_old , i ):
        
    
        
        i represents the index of the particle that has been updated
        
        
        #Getting the i-th column of D_new and of D_inv_old
        d_i = D_new[: , i ].reshape(-1 ,1) 
        d_inv_i = D_inv_old[: , i].reshape(-1 ,1)

        #Extract all the column except the i-th 
        D_inv_no_i = jnp.hstack((D_inv_old[:, :i], D_inv_old[:, i+1:])) 

        #Calulate the ratio of the move
        R = jnp.sum( d_i * d_inv_i)


        #Here I multiply the d_i with all the columns of D_inv_no_i and then sum along the rows
        S = jnp.sum( d_i * D_inv_no_i , axis= 0).reshape(1,-1)


        #Update all the column j != i 
        D_inv_no_i_new = D_inv_no_i  - S / R * d_inv_i

        #Update the i-th column of the inverse

        d_inv_i_new = d_inv_i / R

        #stack together the column in order to rebuild the all matrix

        D_inv_new = self.reconstruct_array(D_inv_no_i_new , d_inv_i_new ,i)

        return D_inv_new


    """
        

    def _initialize_vars(self, nparticles, dim, log, logger, logger_level):
        """ Initializing the parameters in the VMC instance """
        assert isinstance(nparticles, int), "nparticles must be an integer"
        assert isinstance(dim, int), "dim must be an integer"
        self._N = nparticles
        self._dim = dim
        self._log = log if log else False

        if logger:
            self._logger = logger
        else:
            import logging

            self._logger = logging.getLogger(__name__)

        self._logger_level = logger_level

        # Generate initial positions randomly
        # Note: We split the RNG key to ensure subsequent uses of RNG don't reuse the same state.
        key, subkey = random.split(self.rng)
        initial_positions =  random.normal(
            subkey, (nparticles, dim)
        ) #self.initialize_positions(subkey , nparticles, dim , self.radius) # Using JAX for random numbers
        # Initialize logp, assuming a starting value or computation
        a = self.params.get("alpha")  # Using Parameter.get to access alpha
        initial_logp = 0  # self.prob_closure(initial_positions , a)  # Now I use the log of the modulus of wave function, can be changed
        """
        self._D_down , self._D_up = self.wf(initial_positions)
        """

        self._D_down = jnp.ones((self._N//2 , self._N//2))
        self._D_up = jnp.ones((self._N//2 , self._N//2))

        self._D_down_inv = jnp.linalg.inv(self._D_down)
        self._D_up_inv = jnp.linalg.inv(self._D_up)
        
        self.state = State(
            positions=initial_positions, logp=initial_logp, n_accepted=0, delta=0 ,
              D_up = self._D_up , D_down = self._D_down , D_inv_up= self._D_up_inv , D_inv_down= self._D_down_inv
        )



    def _initialize_variational_params(self, alpha=None):
        # Initialize variational parameters in the correct range with the correct shape
        # Take a look at the qs.utils.Parameter class. You may or may not use it depending on how you implement your code.
        # Here, we initialize the variational parameter 'alpha'.
        if alpha:
            initial_params = {"alpha": jnp.array([alpha])}
            print("alpha is ", alpha)
        else:
            initial_params = {
                "alpha": jnp.array([0.5])
            }  # Example initial value for alpha ( 1 paramter)
            print("alpha is ", 0.5)
        self.params = Parameter(
            initial_params
        )  # I still do not understand what should be the alpha dim
        pass
