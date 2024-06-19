def wf_closure_train(self, r, alpha):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector

        return: should return Ψ(alpha, r)

        OBS: We strongly recommend you work with the wavefunction in log domain.

        """
        g = -alpha *self.backend.sum(
            self.beta**2 * (r[:, 0] ** 2) + self.backend.sum(r[:, 1:] ** 2, axis=1)
        )

        wf = self.backend.sum(g) 

        return wf



def wf_closure_int(self, r, alpha):
        """

        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (1,1) array so that alpha is just a number but in the array form

        return: should return a an array of the wavefunction for each particle ( N, )

        OBS: We strongly recommend you work with the wavefunction in log domain.
        """

        g = -alpha * (
            self.beta**2 * (r[:, 0] ** 2) + self.backend.sum(r[:, 1:] ** 2, axis=1)
        )  # Sum over the coordinates x^2 + y^2 + z^2 for each particle
        # Calculate pairwise distances.
        distances = self.la.norm(self.state.r_dist, axis=-1)
        # Compute f using the masked distances
        f = jnp.log(
            jnp.where(distances < self.radius, 0, 1 - self.radius / distances)
            + jnp.eye(r.shape[0])
        )
        f_term = self.backend.sum(f, axis=1)
        wf = g + f_term

        return wf



def uprime(self, rij):
        return self.radius / (rij**2 - self.radius * rij)

def u_double_prime(self, rij):
    return (
        self.radius * (self.radius - 2 * rij) / ((rij**2 - self.radius * rij) ** 2)
    )

def anal_laplacian_closure_int(self, r, alpha):
    """Something something soon easter boys - this should take osme arguments and return some value?"""

    k_indx = np.where(self.state.positions == r)[0][0]
    k_distances = self.la.norm(self.state.r_dist + 1e-8, axis=-1)[k_indx]
    self.k_distances = jnp.delete(k_distances, k_indx, axis=0)
    r_dist = self.state.positions - r + 1e-8
    r_dist = jnp.delete(r_dist, k_indx, axis=0)
    x, y, z = r
    self.grad_phi = -2 * alpha * jnp.array([x, y, self.beta * z])
    self.grad_phi_square = 4 * alpha**2 * (
        x**2 + y**2 + self.beta**2 * z**2
    ) - 2 * alpha * (self.beta + 2)

    self.first_term = self.grad_phi_square

    self.second_term_sum = jnp.sum(
        (r_dist.T / self.k_distances * self.uprime(self.k_distances)).T, axis=0
    )  # sum downwards, to keep it as a vector

    self.second_term = 2 * jnp.dot(self.grad_phi, self.second_term_sum)
    self.third_term = jnp.dot(self.second_term_sum, self.second_term_sum)
    self.fourth_term = jnp.sum(
        self.u_double_prime(self.k_distances)
        + 2 / self.k_distances * self.uprime(self.k_distances)
    )

    self.anal_laplacian = (
        self.first_term + self.second_term + self.third_term + self.fourth_term
    )

    return self.anal_laplacian


def int_energy(self, r):

        kinetic_energy =  0

        for i in range(self._N):
            kinetic_energy +=  self.alg_int.laplacian(r[i])

        kinetic_energy = -0.5 * kinetic_energy

        potential_energy  = 0.5 * self.backend.sum(self.beta**2 * r[:, 0]**2 + self.backend.sum(r[:, 1:]**2, axis=1))

        int_energy = kinetic_energy + potential_energy

        return int_energy


"""
    

class EllipticOscillator(HarmonicOscillator):
    def __init__(self, alg_int, nparticles, dim, log, logger, seed, logger_level, int_type, backend, beta):
        super().__init__(alg_int, nparticles, dim, log, logger, seed, logger_level, int_type, backend)

        self.beta = beta  # Store the ellipticity parameter

    def potential_energy(self, r):
        Calculates the potential energy
        
        pe = 0.5 * self.backend.sum(self.beta**2 * r[:, 0]**2 + self.backend.sum(r[:, 1:]**2, axis=1))
        int_energy = 0

        if self._int_type == "Coulomb":
            r_copy = r.copy()
            r_dist = self.la.norm(r_copy[None, ...] - r_copy[:, None, :], axis=-1)
            r_dist = self.backend.where(r_dist < config.radius, 0, r_dist)     
            int_energy = self.backend.sum(
                self.backend.triu(1 / r_dist, k=1)
            )   # Calculates the upper triangular of the distance matrix (to not do a double sum)
        else:
            pass

            
        return pe + int_energy
    
    def local_energy(self, wf, r):
        ###TODO Impliment local energy for EO

        Local energy of the system
        Calculates the local energy of a system with positions `r` and wavefunction `wf`.
        `wf` is assumed to be the log of the wave function.
        
        # Adjust the potential energy calculation for the elliptic oscillator
        # Assuming r is structured as [nparticles, dim], and the first column is x, second is y, and the third is z.
        pe = self.potential_energy(r)
        ke = self.kinetic_energy(r)
        
        # Correct calculation of local energy
        local_energy =  ke + pe

        return local_energy


"""
    