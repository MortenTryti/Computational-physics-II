import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config

# data for training
def training_plot(particle_type, nparticles):

    cycles = np.loadtxt(f"data_analysis/cycles_{particle_type}_{nparticles}.dat")
    alphas = np.loadtxt(f"data_analysis/alphas_{particle_type}_{nparticles}.dat")
    energies = np.loadtxt(f"data_analysis/energies_{particle_type}_{nparticles}.dat")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot for Alpha vs Cycles
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, alphas, label=f"Alpha of {particle_type}", color='blue', linestyle='-', marker='o', markersize=4)
    plt.xlabel("Cycles")
    plt.ylabel("Alpha")
    plt.title(f"Alpha vs Cycles of {particle_type} with {nparticles} particles")
    plt.legend()
    plt.savefig("figures/alpha_vs_cycles.pdf")
    plt.close()

    # Plot for Energy vs Cycles
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, energies, label=f"Energy of {particle_type}", color='green', linestyle='-', marker='o', markersize=4)
    plt.xlabel("Cycles")
    plt.ylabel("Energy")
    plt.title(f"Energy vs Cycles of {particle_type} with {nparticles} particles")
    plt.legend()
    plt.savefig("figures/energy_vs_cycles.pdf")
    plt.close()


def bootstrap_plots( particle_type , nparticles):

    n_boot_values = np.loadtxt(f"data_analysis/n_boot_values_{particle_type}_{nparticles}.dat")
    variances_bo = np.loadtxt(f"data_analysis/variances_bo_{particle_type}_{nparticles}.dat")
    variances_boot = np.loadtxt(f"data_analysis/variances_boot_{particle_type}_{nparticles}.dat")
    block_sizes = np.loadtxt(f"data_analysis/block_sizes_{particle_type}_{nparticles}.dat")
    variances_bl = np.loadtxt(f"data_analysis/variances_bl_{particle_type}_{nparticles}.dat")
    variances_block = np.loadtxt(f"data_analysis/variances_block_{particle_type}_{nparticles}.dat")


    # Set Seaborn style
    sns.set(style="whitegrid", context="talk", palette="colorblind")

    # Create a figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Bootstrapped Variance vs Normal Variance on the first subplot
    ax[0].plot(n_boot_values, variances_bo, label="Normal Variance", marker='o', linestyle='-', linewidth=2)
    ax[0].plot(n_boot_values, variances_boot, label="Bootstrapped Variance", marker='x', linestyle='--', linewidth=2)
    ax[0].set_xlabel("Number of Bootstraps")
    ax[0].set_ylabel("Variance")
    ax[0].set_title(f"Variance Comparison: Bootstrapped vs Normal of {particle_type} with {nparticles} particles")
    ax[0].legend()

    # Plot 2: Blocking Variance vs Normal Variance on the second subplot
    ax[1].plot(block_sizes, variances_bl, label="Normal Variance", marker='o', linestyle='-', linewidth=2)
    ax[1].plot(block_sizes, variances_block, label="Blocking Variance", marker='x', linestyle='--', linewidth=2)
    ax[1].set_xlabel("Block Size")
    ax[1].set_ylabel("Variance")
    ax[1].set_title(f"Variance Comparison: Blocking vs Normal of {particle_type} with {nparticles} particles")
    ax[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("figures/variance_comparisons.pdf")
    plt.close()

    # Compare bootstrapping and blocking
    plt.figure(figsize=(12, 8))
    plt.plot(n_boot_values, variances_boot, label="Variance with bootstrap", marker='o', linestyle='-', linewidth=2)
    plt.plot(block_sizes, variances_block, label="Variance with Blocking", marker='x', linestyle='--', linewidth=2)
    plt.xlabel("Number of Bootstraps / Block Sizes")
    plt.ylabel("Variance")
    plt.title(f"Variance Comparison: Bootstrapping vs Blocking of {particle_type} with {nparticles} particles")
    plt.legend()
    plt.savefig("figures/boot_vs_blocking.pdf")


def plot_energy_vs_particles():

    energies_bosons = np.loadtxt("data_analysis/bosons_energies.dat")
    energies_fermions = np.loadtxt("data_analysis/fermion_energies.dat")
    n_particles = np.loadtxt("data_analysis/n_particles.dat")


    # Set Seaborn style
    sns.set(style="whitegrid", palette="muted")

    # Create the plot
    plt.figure(figsize=(10, 6))  # Optionally increase figure size for better readability
    plt.plot(n_particles, energies_fermions, 'o-', label="Fermions", linewidth=2, markersize=8)
    plt.plot(n_particles, energies_bosons, 's-', label="Bosons", linewidth=2, markersize=8)
    plt.xlabel("Number of particles")
    plt.ylabel("Energy")
    plt.title("Energy vs Number of Particles")
    plt.legend(title="Particle Type")
    plt.grid(True)  # Ensure the grid is enabled

    # Save the figure
    plt.savefig("figures/energy_vs_particles.pdf")


def plot_heatmap(particle_type, nparticles):


    delta_t_values = np.loadtxt(f"data_analysis/delta_t_values_{particle_type}_{nparticles}.dat")
    alpha_values = np.loadtxt(f"data_analysis/alpha_values_{particle_type}_{nparticles}.dat")
    energy_ana = np.loadtxt(f"data_analysis/energy_ana_{particle_type}_{nparticles}.dat")
    energy_jax = np.loadtxt(f"data_analysis/energy_jax_{particle_type}_{nparticles}.dat")

    # Assuming your energy_ana and energy_jax matrices are populated as described above
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # Two plots side by side

    # Setting the extent for imshow, ensuring the x-axis values (delta_t) are represented in log scale
    extent = [np.log10(delta_t_values.min()), np.log10(delta_t_values.max()), alpha_values.min(), alpha_values.max()]

    # Heatmap for analytical backend
    pos = ax[0].imshow(energy_ana, cmap='viridis', aspect='auto', origin='lower', extent=extent)
    ax[0].set_title(f'Energy (Ana) vs. Log(Delta_t) and Alpha of {particle_type} with {nparticles} particles')
    ax[0].set_xlabel('Log(Delta_t)')
    ax[0].set_ylabel('Alpha')
    fig.colorbar(pos, ax=ax[0])

    # Setting the x-axis to show the actual log10 values of delta_t
    ax[0].set_xticks(np.log10(delta_t_values))
    ax[0].set_xticklabels([f'{val:f}' for val in delta_t_values])

    # Heatmap for JAX backend
    pos = ax[1].imshow(energy_jax, cmap='viridis', aspect='auto', origin='lower', extent=extent)
    ax[1].set_title(f'Energy (Jax) vs. Log(Delta_t) and Alpha of {particle_type} with {nparticles} particles')
    ax[1].set_xlabel('Log(Delta_t)')
    # We only set the ylabel for the first subplot

    # Setting the x-axis to show the actual log10 values of delta_t for the second plot as well
    ax[1].set_xticks(np.log10(delta_t_values))
    ax[1].set_xticklabels([f'{val:f}' for val in delta_t_values])

    fig.colorbar(pos, ax=ax[1])
    plt.tight_layout()
    plt.savefig("figures/delta_t_heatmaps.pdf")



def plot_energy_stability(particle_type, nparticles):

    sample_sizes = np.loadtxt(f"data_analysis/samples_{particle_type}_{nparticles}.dat")
    alpha_values = np.loadtxt(f"data_analysis/alpha_values_stab_{particle_type}_{nparticles}.dat")
    energy_matrix = np.loadtxt(f"data_analysis/energy_matrix_{particle_type}_{nparticles}.dat")
    variance_matrix = np.loadtxt(f"data_analysis/variance_matrix_{particle_type}_{nparticles}.dat")


    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plotting Energy
    
    colors = sns.color_palette("magma", len(sample_sizes))  # Choose a color palette
    for i, size in enumerate(sample_sizes):
        ax[0].plot(alpha_values, energy_matrix[i], "o-", label=f"n_samples = {size}", color=colors[i])
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Energy")
    ax[0].set_title(f"Energy as a function of alpha in JAX calculation of {particle_type} with {nparticles} particles")
    ax[0].legend()

    # Plotting Variance
    for i, size in enumerate(sample_sizes):
        ax[1].plot(alpha_values, variance_matrix[i], "o-", label=f"n_samples = {size}", color=colors[i])
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Variance")
    ax[1].set_title(f"Variance as a function of alpha in JAX calculation of {particle_type} with {nparticles} particles")
    ax[1].legend()

    # Adjust layout to ensure titles and labels are visible
    plt.tight_layout()

    # Save the figure
    fig.savefig("figures/energy_stability.pdf")

def plot_energy_vs_alpha(particle_type, nparticles):


    alpha_values = np.loadtxt(f"data_analysis/alpha_values_plot_{particle_type}_{nparticles}.dat")
    energies = np.loadtxt(f"data_analysis/energies_{particle_type}_{nparticles}.dat")
    variances = np.loadtxt(f"data_analysis/variances_{particle_type}_{nparticles}.dat")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plotting Energy
    ax[0].plot(alpha_values, energies, "o-", label="Energy", color='royalblue', markersize=8, linewidth=2)
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Energy")
    ax[0].set_title(f"Energy as a function of alpha in JAX calculation of {particle_type} with {nparticles} particles")
    ax[0].legend()

    # Plotting Variance
    ax[1].plot(alpha_values, variances, "o-", label="Variance", color='crimson', markersize=8, linewidth=2)
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Variance")
    ax[1].set_title(f"Variance as a function of alpha in JAX calculation of {particle_type} with {nparticles} particles")
    ax[1].legend()

    # Adjust layout to ensure no overlapping
    plt.tight_layout()

    # Save the figure
    fig.savefig("figures/energy_alpha.pdf")


def plot_time_measures(particle_type, nparticles):

    sample_values = np.loadtxt(f"data_analysis/sample_values_{particle_type}_{nparticles}.dat")
    times_ana = np.loadtxt(f"data_analysis/times_ana_{particle_type}_{nparticles}.dat")
    times_jax = np.loadtxt(f"data_analysis/times_jax_{particle_type}_{nparticles}.dat")
    times_difference = np.loadtxt(f"data_analysis/times_difference_{particle_type}_{nparticles}.dat")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # First plot: Execution times for Numpy and Jax
    colors = sns.color_palette("tab10")  # Selecting a color palette
    ax[0].plot(np.log2(sample_values), times_ana, label="Numpy", color=colors[0], marker='o', linestyle='-', linewidth=2)
    ax[0].plot(np.log2(sample_values), times_jax, label="Jax", color=colors[1], marker='s', linestyle='-', linewidth=2)
    ax[0].set_xlabel("Log2(Number of samples)")
    ax[0].set_ylabel("Execution time [s]")
    ax[0].set_title(f"Execution time for different number of samples of {particle_type} with {nparticles} particles")
    ax[0].legend()

    # Second plot: Difference in execution times
    ax[1].plot(np.log2(sample_values), times_difference, label="Difference (Numpy - Jax)", color='red', marker='^', linestyle='-', linewidth=2)
    ax[1].set_xlabel("Log2(Number of samples)")
    ax[1].set_ylabel("Difference in execution time [s]")
    ax[1].set_title(f"Difference in execution time between Numpy and Jax of {particle_type} with {nparticles} particles")
    ax[1].legend()

    # Adjust layout to ensure there is no content clipping
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/execution_time_comparison.pdf")
    

def plot_one_body():

    r1 = np.loadtxt("data_analysis/r1.dat")
    rho = np.loadtxt("data_analysis/rho.dat")
    rho_noInt = np.loadtxt("data_analysis/rho_noInt.dat")

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set a larger figure size for better readability
    sns.lineplot(r1, rho, label="Interactions", linestyle='-', linewidth=2.5, marker='o', markersize=8)
    sns.lineplot(r1, rho_noInt, label="No Interactions", linestyle='-', linewidth=2.5, marker='x', markersize=8)

    # Adding plot labels and title
    plt.xlabel("$r_1 [-]$", fontsize=14)
    plt.ylabel(r"$\rho(r_1) [-]$", fontsize=14)
    plt.title("Radial Distribution With and Without Interactions", fontsize=16)

    # Adding legend with better placement
    plt.legend(title="Condition", title_fontsize='13', fontsize='12')

    # Saving the plot
    plt.savefig("figures/onebody_comp.pdf")



def position_plot(nparticles,  particle_type , nsamples):

    sampled_positions = np.loadtxt(f"data_analysis/sampled_positions_{particle_type}_{nparticles}_{nsamples}.dat")


   
    # Assuming each column is a particle and each row is a sample, create a DataFrame
    df = pd.DataFrame(sampled_positions, columns=[f'Particle {i+1}' for i in range(sampled_positions.shape[1])])
    df['Sample'] = df.index % 100  # Recreate the sample index if needed, modify '100' based on actual samples per chain

    # Melting the DataFrame to use Seaborn easily
    df_melted = df.melt(id_vars=['Sample'], var_name='Particle', value_name='Distance')

    # Set up the FacetGrid
    g = sns.FacetGrid(df_melted, col="Particle", col_wrap=3, height=4, aspect=1.5, hue='Sample', palette='viridis')
    g.map(plt.scatter, 'Sample', 'Distance', alpha=0.6, s=20)  # Scatter plot for each particle's position over time

    # Adding a line plot to connect the positions, to show the movement over samples
    g.map(plt.plot, 'Sample', 'Distance', alpha=0.3)

    # Enhance the plot
    g.set_titles("{col_name}")
    g.add_legend(title="Sample Index")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Particle Distances Over Time for {nparticles} {particle_type}  with {nsamples} samples ')  # Overall title
    
    # Save the figure
    plt.savefig("figures/position_plot.pdf")



#training_plot(config.particle_type, config.nparticles)
bootstrap_plots(config.particle_type, config.nparticles)
#plot_energy_vs_particles()
#plot_heatmap(config.particle_type, config.nparticles)
#plot_energy_stability(config.particle_type, config.nparticles)
#plot_energy_vs_alpha(config.particle_type, config.nparticles)
#plot_time_measures(config.particle_type, config.nparticles)
#plot_one_body()
#position_plot(config.nparticles, config.particle_type, config.nsamples)








    



