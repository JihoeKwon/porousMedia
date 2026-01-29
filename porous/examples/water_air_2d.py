"""
2D Water-Air Injection Example (Five-Spot Pattern)

This example demonstrates two-phase isothermal flow in a 2D porous medium.
Water is injected at one corner and produced at the opposite corner,
similar to a quarter five-spot pattern used in oil recovery.

Physical Setup:
- 2D domain (100m x 100m)
- Initially saturated with air (Sw = 0.2 = residual water)
- Water injection at corner (0, 0)
- Fixed pressure production at corner (L, L)
- Isothermal conditions (20°C)

Expected Results:
- Diagonal water saturation front propagation
- Pressure distribution from injector to producer
- Breakthrough curve at producer
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from porous.core.mesh import create_mesh_2d
from porous.core.properties import RockProperties
from porous.core.state import StateManager, FluidSystem, PhaseState
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator


def run_2d_water_air_example(plot_results: bool = True, verbose: bool = True):
    """
    Run the 2D water-air injection example.

    Args:
        plot_results: Whether to plot results
        verbose: Whether to print progress

    Returns:
        Dictionary of results
    """
    print("=" * 60)
    print("2D Water-Air Injection Example (Five-Spot Pattern)")
    print("=" * 60)

    # =========================================
    # Domain Parameters
    # =========================================
    lx = 100.0          # Domain length in x [m]
    ly = 100.0          # Domain length in y [m]
    nx = 20             # Number of cells in x
    ny = 20             # Number of cells in y
    thickness = 10.0    # Thickness [m]

    # =========================================
    # Rock Properties
    # =========================================
    porosity = 0.2
    permeability = 1e-13  # 100 mD [m²]

    # Relative permeability (Corey model)
    slr = 0.2   # Residual water saturation
    sgr = 0.2   # Residual gas (air) saturation
    nl = 2.0    # Water exponent
    ng = 2.0    # Gas exponent

    # =========================================
    # Initial and Boundary Conditions
    # =========================================
    initial_pressure = 1e7      # 100 bar [Pa]
    initial_temperature = 293.15  # 20°C [K]
    initial_sw = slr            # Initial water saturation = residual

    outlet_pressure = 9e6       # 90 bar [Pa] (fixed at producer)
    inlet_water_rate = 0.1      # 100 g/s water injection [kg/s]

    # =========================================
    # Simulation Time
    # =========================================
    end_time = 3600 * 24 * 5    # 5 days [s]
    dt_initial = 10.0           # Initial time step [s]

    # =========================================
    # Create Mesh
    # =========================================
    print("\n[1] Creating 2D mesh...")
    mesh = create_mesh_2d(lx=lx, ly=ly, nx=nx, ny=ny, thickness=thickness)
    print(f"    Grid: {nx} x {ny} = {mesh.num_cells} cells")
    print(f"    Connections: {mesh.num_connections}")
    print(f"    Cell size: {lx/nx:.2f} m x {ly/ny:.2f} m")

    # Producer at corner (nx-1, ny-1) - large volume for fixed pressure BC
    producer_cell = nx * ny - 1  # Last cell (top-right corner)
    mesh.cells[producer_cell].volume *= 1e8

    # =========================================
    # Rock Properties
    # =========================================
    print("\n[2] Setting rock properties...")
    rock = RockProperties(
        porosity=porosity,
        permeability=np.array([permeability, permeability, permeability])
    )
    rocks = [rock] * mesh.num_cells
    print(f"    Porosity: {porosity}")
    print(f"    Permeability: {permeability/1e-15:.1f} mD")

    # =========================================
    # Relative Permeability and Capillary Pressure
    # =========================================
    print("\n[3] Setting constitutive models...")
    rel_perm = CoreyRelPerm(slr=slr, sgr=sgr, nl=nl, ng=ng)
    cap_pressure = VanGenuchtenCapillary(slr=slr, alpha=1e-4, m=0.45)
    print(f"    Corey model: slr={slr}, sgr={sgr}, nl={nl}, ng={ng}")

    # =========================================
    # State Manager (Water-Air System)
    # =========================================
    print("\n[4] Initializing states (Water-Air isothermal)...")
    state_manager = StateManager(
        mesh.num_cells,
        default_rel_perm=rel_perm,
        default_cap_pressure=cap_pressure,
        fluid_system=FluidSystem.WATER_AIR
    )

    # Initialize with residual water saturation (mostly air)
    state_manager.initialize_uniform(
        pressure=initial_pressure,
        temperature=initial_temperature,
        liquid_saturation=initial_sw
    )

    # Set producer to lower pressure
    state_manager.states[producer_cell].pressure = outlet_pressure
    state_manager.states[producer_cell].update_secondary_variables()

    print(f"    Initial pressure: {initial_pressure/1e5:.1f} bar")
    print(f"    Producer pressure: {outlet_pressure/1e5:.1f} bar")
    print(f"    Initial Sw: {initial_sw}")

    # =========================================
    # Source Terms
    # =========================================
    print("\n[5] Setting source terms...")
    num_eq = 2  # Water mass, Air mass
    source = np.zeros((mesh.num_cells, num_eq))

    # Water injection at first cell (corner 0,0)
    injector_cell = 0
    source[injector_cell, 0] = inlet_water_rate   # Water mass rate [kg/s]
    source[injector_cell, 1] = 0.0                # No air injection

    print(f"    Injector cell: {injector_cell} (corner 0,0)")
    print(f"    Producer cell: {producer_cell} (corner {nx-1},{ny-1})")
    print(f"    Water injection rate: {inlet_water_rate*1000:.1f} g/s")

    # =========================================
    # Run Simulation
    # =========================================
    print("\n[6] Running simulation...")
    print(f"    End time: {end_time/86400:.1f} days")
    print(f"    Initial dt: {dt_initial:.1f} s")

    simulator = Simulator(mesh, state_manager, rocks)
    simulator.set_simulation_time(end_time=end_time, dt_initial=dt_initial)
    simulator.set_source_terms(source)

    try:
        simulator.run(verbose=verbose)
        print("\n[7] Simulation completed successfully!")
    except Exception as e:
        print(f"\n[!] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # =========================================
    # Extract Results
    # =========================================
    results = simulator.get_results()

    # Cell centers for plotting
    x = np.array([mesh.cells[i].center[0] for i in range(mesh.num_cells)])
    y = np.array([mesh.cells[i].center[1] for i in range(mesh.num_cells)])

    # Final state
    pressures = state_manager.get_pressures()
    saturations = state_manager.get_liquid_saturations()

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Pressure range: {pressures.min()/1e5:.2f} - {pressures.max()/1e5:.2f} bar")
    print(f"Water saturation range: {saturations.min():.3f} - {saturations.max():.3f}")

    # Check breakthrough
    producer_sw = saturations[producer_cell]
    print(f"Producer water saturation: {producer_sw:.3f}")
    if producer_sw > slr + 0.01:
        print("  -> Water breakthrough has occurred!")
    else:
        print("  -> No breakthrough yet")

    # =========================================
    # Plot Results
    # =========================================
    if plot_results:
        try:
            import matplotlib.pyplot as plt

            # Create output directory
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "output")
            os.makedirs(output_dir, exist_ok=True)

            # Reshape data for 2D plotting
            sat_2d = saturations.reshape((ny, nx))
            pres_2d = (pressures / 1e5).reshape((ny, nx))  # bar

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Saturation contour
            ax1 = axes[0]
            X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))
            cs1 = ax1.contourf(X, Y, sat_2d, levels=20, cmap='Blues')
            ax1.contour(X, Y, sat_2d, levels=[0.3, 0.4, 0.5, 0.6], colors='black', linewidths=0.5)
            plt.colorbar(cs1, ax=ax1, label='Water Saturation [-]')
            ax1.plot(0, 0, 'r^', markersize=15, label='Injector')
            ax1.plot(lx, ly, 'go', markersize=15, label='Producer')
            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.set_title('Water Saturation Distribution')
            ax1.legend()
            ax1.set_aspect('equal')

            # Pressure contour
            ax2 = axes[1]
            cs2 = ax2.contourf(X, Y, pres_2d, levels=20, cmap='RdYlBu_r')
            ax2.contour(X, Y, pres_2d, levels=10, colors='black', linewidths=0.5)
            plt.colorbar(cs2, ax=ax2, label='Pressure [bar]')
            ax2.plot(0, 0, 'r^', markersize=15, label='Injector')
            ax2.plot(lx, ly, 'go', markersize=15, label='Producer')
            ax2.set_xlabel('X [m]')
            ax2.set_ylabel('Y [m]')
            ax2.set_title('Pressure Distribution')
            ax2.legend()
            ax2.set_aspect('equal')

            plt.tight_layout()

            output_file = os.path.join(output_dir, "water_air_2d.png")
            plt.savefig(output_file, dpi=150)
            print(f"\nPlot saved to: {output_file}")

            plt.show()

        except ImportError:
            print("\nMatplotlib not available. Skipping plots.")

    return {
        'x': x,
        'y': y,
        'nx': nx,
        'ny': ny,
        'lx': lx,
        'ly': ly,
        'pressure': pressures,
        'saturation': saturations,
        'times': results.get('times', []),
        'pressure_history': results.get('pressures', []),
        'saturation_history': results.get('saturations', [])
    }


def create_2d_animation(results: dict, output_path: str,
                        slr: float = 0.2, sgr: float = 0.2, fps: int = 10):
    """
    Create animation of 2D saturation front propagation.

    Args:
        results: Dictionary from run_2d_water_air_example
        output_path: Path for output GIF file
        slr: Residual liquid saturation
        sgr: Residual gas saturation
        fps: Frames per second
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    nx = results['nx']
    ny = results['ny']
    lx = results['lx']
    ly = results['ly']
    times = results['times']
    sat_history = results['saturation_history']
    pres_history = results['pressure_history']

    if len(times) == 0 or len(sat_history) == 0:
        print("No time history data available for animation.")
        return

    print(f"\nCreating 2D animation with {len(times)} frames...")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    X, Y = np.meshgrid(np.linspace(0, lx, nx), np.linspace(0, ly, ny))

    # Initial plots
    sat_2d = sat_history[0].reshape((ny, nx))
    pres_2d = (pres_history[0] / 1e5).reshape((ny, nx))

    # Saturation plot
    cs1 = ax1.contourf(X, Y, sat_2d, levels=np.linspace(slr, 1-sgr, 20), cmap='Blues')
    cb1 = plt.colorbar(cs1, ax=ax1, label='Water Saturation [-]')
    ax1.plot(0, 0, 'r^', markersize=12, label='Injector')
    ax1.plot(lx, ly, 'go', markersize=12, label='Producer')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    title1 = ax1.set_title('Water Saturation')

    # Pressure plot
    cs2 = ax2.contourf(X, Y, pres_2d, levels=20, cmap='RdYlBu_r')
    cb2 = plt.colorbar(cs2, ax=ax2, label='Pressure [bar]')
    ax2.plot(0, 0, 'r^', markersize=12, label='Injector')
    ax2.plot(lx, ly, 'go', markersize=12, label='Producer')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')
    title2 = ax2.set_title('Pressure')

    plt.tight_layout()

    def animate(frame):
        ax1.clear()
        ax2.clear()

        sat_2d = sat_history[frame].reshape((ny, nx))
        pres_2d = (pres_history[frame] / 1e5).reshape((ny, nx))
        t = times[frame]

        # Format time
        if t < 3600:
            time_str = f't = {t:.1f} s'
        elif t < 86400:
            time_str = f't = {t/3600:.1f} hours'
        else:
            time_str = f't = {t/86400:.1f} days'

        # Saturation
        cs1 = ax1.contourf(X, Y, sat_2d, levels=np.linspace(slr, 0.95, 20), cmap='Blues')
        ax1.contour(X, Y, sat_2d, levels=[0.3, 0.4, 0.5, 0.6], colors='black', linewidths=0.5)
        ax1.plot(0, 0, 'r^', markersize=12)
        ax1.plot(lx, ly, 'go', markersize=12)
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title(f'Water Saturation ({time_str})')
        ax1.set_aspect('equal')

        # Pressure
        cs2 = ax2.contourf(X, Y, pres_2d, levels=20, cmap='RdYlBu_r')
        ax2.contour(X, Y, pres_2d, levels=10, colors='black', linewidths=0.5)
        ax2.plot(0, 0, 'r^', markersize=12)
        ax2.plot(lx, ly, 'go', markersize=12)
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_title(f'Pressure ({time_str})')
        ax2.set_aspect('equal')

        return []

    # Select frames (subsample if too many)
    max_frames = 100
    if len(times) > max_frames:
        frame_indices = np.linspace(0, len(times)-1, max_frames, dtype=int)
    else:
        frame_indices = range(len(times))

    anim = animation.FuncAnimation(
        fig, animate,
        frames=frame_indices, interval=1000//fps
    )

    # Save animation
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Animation saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='2D Water-Air Injection Example')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    parser.add_argument('--animation', action='store_true', help='Create animation')
    args = parser.parse_args()

    results = run_2d_water_air_example(
        plot_results=not args.no_plot,
        verbose=not args.quiet
    )

    if results is not None:
        # Create animation if requested
        if args.animation:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "output")
            os.makedirs(output_dir, exist_ok=True)
            anim_path = os.path.join(output_dir, "water_air_2d_animation.gif")
            create_2d_animation(results, anim_path)
