"""
1D Water Injection Example

This example demonstrates a simple 1D water injection problem
where water is injected at one end of a porous domain.

The domain starts at hydrostatic equilibrium with single-phase
liquid water. Water is injected at constant rate, causing
pressure to build up and propagate through the domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import POROUS modules
from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager, PhaseState
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator
from porous.io.output import VTKWriter, CSVWriter


def run_1d_injection():
    """
    Run 1D injection simulation.

    Returns:
        Dictionary of simulation results
    """
    print("=" * 60)
    print("POROUS: 1D Water Injection Example")
    print("=" * 60)

    # =====================
    # Domain Setup
    # =====================
    length = 100.0  # Domain length [m]
    num_cells = 50  # Number of cells
    area = 1.0      # Cross-sectional area [m²]

    print(f"\nDomain: {length} m, {num_cells} cells")

    # Create mesh
    mesh = create_mesh_1d(length, num_cells, area)
    print(f"Created 1D mesh with {mesh.num_cells} cells and {mesh.num_connections} connections")

    # =====================
    # Rock Properties
    # =====================
    porosity = 0.2
    permeability = 1e-13  # 100 mD in m²

    rock = RockProperties(
        name="sandstone",
        porosity=porosity,
        permeability=np.array([permeability, permeability, permeability]),
        density=2650.0,
        specific_heat=1000.0,
        thermal_conductivity=2.0,
        rel_perm_model="corey",
        rel_perm_params={"slr": 0.1, "sgr": 0.05, "nl": 2.0, "ng": 2.0},
        cap_pressure_model="van_genuchten",
        cap_pressure_params={"slr": 0.1, "alpha": 1e-4, "m": 0.45, "pmax": 1e7}
    )

    rock_properties = [rock] * mesh.num_cells
    print(f"Rock: porosity={porosity}, permeability={permeability:.2e} m²")

    # =====================
    # Initial Conditions
    # =====================
    initial_pressure = 1e6   # 10 bar
    initial_temperature = 293.15  # 20°C

    # Create state manager with constitutive models
    rel_perm = CoreyRelPerm(slr=0.1, sgr=0.05, nl=2.0, ng=2.0)
    cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-4, m=0.45)

    state_manager = StateManager(
        mesh.num_cells,
        default_rel_perm=rel_perm,
        default_cap_pressure=cap_pressure
    )

    # Initialize with uniform conditions (single-phase liquid)
    state_manager.initialize_uniform(
        pressure=initial_pressure,
        temperature=initial_temperature,
        liquid_saturation=1.0
    )

    print(f"Initial conditions: P={initial_pressure/1e5:.1f} bar, T={initial_temperature-273.15:.1f}°C")

    # =====================
    # Source Terms (Injection)
    # =====================
    injection_rate = 0.01  # 0.01 kg/s water injection
    num_eq = 2  # Water mass, gas mass

    source_terms = np.zeros((mesh.num_cells, num_eq))
    source_terms[0, 0] = injection_rate  # Inject water in first cell

    print(f"Injection rate: {injection_rate} kg/s in cell 0")

    # =====================
    # Simulation Setup
    # =====================
    simulator = Simulator(
        mesh=mesh,
        state_manager=state_manager,
        rock_properties=rock_properties,
        include_energy=False
    )

    # Simulation time
    end_time = 3600.0 * 24  # 1 day
    dt_initial = 10.0

    simulator.set_simulation_time(end_time, dt_initial)
    simulator.set_source_terms(source_terms)

    print(f"\nSimulation: end_time={end_time/3600:.1f} hours")

    # =====================
    # Run Simulation
    # =====================
    print("\nRunning simulation...")
    success = simulator.run(verbose=True)

    if not success:
        print("Simulation failed!")
        return None

    # =====================
    # Get Results
    # =====================
    results = simulator.get_results()

    print("\nSimulation completed successfully!")
    print(f"Number of time steps: {len(results['times'])}")
    print(f"Final time: {results['times'][-1]/3600:.2f} hours")

    # Print final state summary
    final_pressures = results['pressures'][-1]
    print(f"\nFinal pressure range: {final_pressures.min()/1e5:.2f} - {final_pressures.max()/1e5:.2f} bar")

    return results


def plot_results(results: dict, output_dir: str = './output'):
    """
    Plot simulation results.

    Args:
        results: Dictionary from simulator.get_results()
        output_dir: Directory for output plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    times = results['times']
    pressures = results['pressures']
    saturations = results['saturations']

    num_cells = pressures.shape[1]
    x = np.linspace(0, 100, num_cells)  # Assuming 100m domain

    # Plot pressure profiles at different times
    fig, ax = plt.subplots(figsize=(10, 6))

    # Select times to plot
    plot_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    plot_indices = [i for i in plot_indices if i < len(times)]

    for idx in plot_indices:
        t_hours = times[idx] / 3600
        ax.plot(x, pressures[idx] / 1e5, label=f't = {t_hours:.1f} h')

    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_title('1D Water Injection - Pressure Profiles')
    ax.legend()
    ax.grid(True)

    plt.savefig(Path(output_dir) / 'pressure_profiles.png', dpi=150)
    plt.close()

    # Plot pressure vs time at different locations
    fig, ax = plt.subplots(figsize=(10, 6))

    cells_to_plot = [0, num_cells//4, num_cells//2, 3*num_cells//4, num_cells-1]
    for cell in cells_to_plot:
        ax.plot(times/3600, pressures[:, cell]/1e5, label=f'x = {x[cell]:.0f} m')

    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Pressure [bar]')
    ax.set_title('1D Water Injection - Pressure vs Time')
    ax.legend()
    ax.grid(True)

    plt.savefig(Path(output_dir) / 'pressure_vs_time.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


if __name__ == '__main__':
    # Run simulation
    results = run_1d_injection()

    if results is not None:
        # Plot results
        try:
            plot_results(results)
        except ImportError:
            print("\nMatplotlib not available for plotting.")
            print("Results are available in the 'results' dictionary.")
