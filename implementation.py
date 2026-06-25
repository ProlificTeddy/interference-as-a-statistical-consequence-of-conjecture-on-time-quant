import numpy as np
import matplotlib.pyplot as plt
import torch

def simulate_particle_scattering(num_particles, slit_width, screen_distance, time_quant, screen_resolution):
    """
    Simulate particle scattering on a screen with a slit using discrete time dynamics.
    
    Args:
        num_particles (int): Number of particles to simulate.
        slit_width (float): Width of the slit.
        screen_distance (float): Distance from the slit to the screen.
        time_quant (float): Fundamental time quantization step.
        screen_resolution (int): Number of points on the screen to measure intensity.
    
    Returns:
        np.ndarray: Intensity distribution on the screen.
    """
    # Initialize particle positions and velocities
    particle_positions = torch.zeros(num_particles)
    particle_velocities = torch.randn(num_particles) * 0.1  # Small random initial velocities
    
    # Define slit boundaries
    slit_min = -slit_width / 2
    slit_max = slit_width / 2
    
    # Time evolution
    time_steps = int(screen_distance / time_quant)
    for _ in range(time_steps):
        particle_positions += particle_velocities * time_quant
        
        # Reflect particles that hit the slit boundaries
        outside_slit = (particle_positions < slit_min) | (particle_positions > slit_max)
        particle_velocities[outside_slit] *= -1
    
    # Project particles onto the screen
    screen_positions = particle_positions.numpy()
    screen_bins = np.linspace(-slit_width, slit_width, screen_resolution)
    intensity, _ = np.histogram(screen_positions, bins=screen_bins)
    
    return intensity

if __name__ == '__main__':
    # Parameters
    num_particles = 10000
    slit_width = 1.0
    screen_distance = 10.0
    time_quant = 0.01
    screen_resolution = 500
    
    # Simulate particle scattering
    intensity = simulate_particle_scattering(num_particles, slit_width, screen_distance, time_quant, screen_resolution)
    
    # Plot the resulting interference pattern
    screen_bins = np.linspace(-slit_width, slit_width, screen_resolution)
    screen_centers = (screen_bins[:-1] + screen_bins[1:]) / 2
    plt.plot(screen_centers, intensity, label="Interference Pattern")
    plt.xlabel("Screen Position")
    plt.ylabel("Intensity")
    plt.title("Interference Pattern from Discrete Time Dynamics")
    plt.legend()
    plt.show()