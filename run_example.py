#!/usr/bin/env python
"""
POROUS - Main Example Runner

Runs the 1D water injection example to demonstrate the simulator.

Usage:
    python run_example.py [--plot] [--output-dir OUTPUT_DIR]

Options:
    --plot          Generate plots of results
    --output-dir    Directory for output files (default: ./output)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator
from porous.io.output import VTKWriter


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='POROUS - Porous Media Multiphase Flow Simulator'
    )
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of results')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--vtk', action='store_true',
                       help='Write VTK output files')
    args = parser.parse_args()

    print("=" * 60)
    print("POROUS - Porous Media Multiphase Flow Simulator")
    print("=" * 60)
    print()

    # =====================
    # Problem Setup
    # =====================
    print("Setting up 1D water injection problem...")

    # Domain
    length = 100.0  # m
    num_cells = 50
    area = 1.0  # m²

    # Create mesh
    mesh = create_mesh_1d(length, num_cells, area)
    print(f"  Mesh: {num_cells} cells, {length} m length")

    # Rock properties
    rock = RockProperties(
        name="sandstone",
        porosity=0.2,
        permeability=np.array([1e-13, 1e-13, 1e-13]),  # 100 mD
        density=2650.0,
        specific_heat=1000.0,
        thermal_conductivity=2.0
    )
    rock_properties = [rock] * mesh.num_cells
    print(f"  Rock: porosity=0.2, permeability=100 mD")

    # State manager
    rel_perm = CoreyRelPerm(slr=0.1, sgr=0.05, nl=2.0, ng=2.0)
    cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-4, m=0.45)

    state_manager = StateManager(
        mesh.num_cells,
        default_rel_perm=rel_perm,
        default_cap_pressure=cap_pressure
    )

    # Initial conditions
    initial_pressure = 1e6  # 10 bar
    initial_temperature = 293.15  # 20°C
    state_manager.initialize_uniform(initial_pressure, initial_temperature, 1.0)
    print(f"  Initial: P={initial_pressure/1e5:.0f} bar, T={initial_temperature-273.15:.0f}°C")

    # Make the last cell a large-volume boundary (constant pressure reservoir)
    mesh.cells[-1].volume *= 1e8  # Very large volume acts as pressure boundary

    # Source terms (injection in first cell)
    num_eq = 2
    source_terms = np.zeros((mesh.num_cells, num_eq))
    source_terms[0, 0] = 0.001  # 0.001 kg/s water injection (smaller rate)
    print(f"  Injection: 0.001 kg/s in cell 0 (constant P at outlet)")

    # =====================
    # Simulation
    # =====================
    print()
    print("Running simulation...")

    simulator = Simulator(
        mesh=mesh,
        state_manager=state_manager,
        rock_properties=rock_properties,
        include_energy=False
    )

    end_time = 3600.0 * 6  # 6 hours
    dt_initial = 0.1  # Smaller initial time step

    simulator.set_simulation_time(end_time, dt_initial)
    simulator.set_source_terms(source_terms)

    # VTK output
    if args.vtk:
        vtk_writer = VTKWriter(args.output_dir)
        print(f"  VTK output enabled: {args.output_dir}")

    success = simulator.run(verbose=True)

    if not success:
        print("\nSimulation FAILED!")
        return 1

    print("\nSimulation completed successfully!")

    # =====================
    # Results
    # =====================
    results = simulator.get_results()

    print()
    print("Results Summary:")
    print(f"  Time steps: {len(results['times'])}")
    print(f"  Final time: {results['times'][-1]/3600:.2f} hours")

    final_P = results['pressures'][-1]
    print(f"  Final pressure range: {final_P.min()/1e5:.2f} - {final_P.max()/1e5:.2f} bar")

    # Write VTK if requested
    if args.vtk:
        print()
        print("Writing VTK output...")
        vtk_writer.write(mesh, state_manager, results['times'][-1])
        print(f"  VTK file written to {args.output_dir}")

    # Plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            print()
            print("Generating plots...")

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            times = results['times']
            pressures = results['pressures']
            x = np.linspace(0, length, num_cells)

            # Pressure profiles
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, t_idx in enumerate([0, len(times)//3, 2*len(times)//3, -1]):
                t_hours = times[t_idx] / 3600
                ax.plot(x, pressures[t_idx]/1e5, label=f't = {t_hours:.1f} h')
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Pressure [bar]')
            ax.set_title('1D Water Injection - Pressure Profiles')
            ax.legend()
            ax.grid(True)
            plt.savefig(output_dir / 'pressure_profiles.png', dpi=150)
            plt.close()

            # Pressure vs time
            fig, ax = plt.subplots(figsize=(10, 6))
            cells = [0, num_cells//4, num_cells//2, num_cells-1]
            for cell in cells:
                ax.plot(times/3600, pressures[:, cell]/1e5,
                       label=f'x = {x[cell]:.0f} m')
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('Pressure [bar]')
            ax.set_title('1D Water Injection - Pressure vs Time')
            ax.legend()
            ax.grid(True)
            plt.savefig(output_dir / 'pressure_vs_time.png', dpi=150)
            plt.close()

            print(f"  Plots saved to {output_dir}/")

        except ImportError:
            print("  Matplotlib not available, skipping plots")

    print()
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
