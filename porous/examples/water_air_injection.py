"""
1D Water-Air Injection Example (Buckley-Leverett Type)

This example demonstrates two-phase isothermal flow in porous media.
Water is injected into an air-saturated porous medium, creating a
displacement front similar to the Buckley-Leverett problem.

Physical Setup:
- 1D domain (100m length)
- Initially saturated with air (Sw = 0.2 = residual water)
- Water injection at inlet (x=0)
- Fixed pressure at outlet (x=L)
- Isothermal conditions (20°C)

Expected Results:
- Water saturation front propagates from inlet to outlet
- Shock front formation due to nonlinear fractional flow
- Front velocity depends on relative permeability curves
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from porous.core.mesh import create_mesh_1d
from porous.core.properties import RockProperties
from porous.core.state import StateManager, FluidSystem, PhaseState
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator


def run_water_air_example(plot_results: bool = True, verbose: bool = True):
    """
    Run the 1D water-air injection example.

    Args:
        plot_results: Whether to plot results
        verbose: Whether to print progress

    Returns:
        Dictionary of results
    """
    print("=" * 60)
    print("1D Water-Air Injection Example (Buckley-Leverett Type)")
    print("=" * 60)

    # =========================================
    # Domain Parameters
    # =========================================
    length = 100.0      # Domain length [m]
    num_cells = 100     # Number of cells (high resolution for shock)
    area = 1.0          # Cross-sectional area [m²]

    # =========================================
    # Rock Properties
    # =========================================
    porosity = 0.2
    permeability = 1e-13  # 100 mD [m²]

    # Relative permeability (Corey model)
    # For Buckley-Leverett: quadratic curves typical
    slr = 0.2   # Residual water saturation
    sgr = 0.2   # Residual gas (air) saturation
    nl = 2.0    # Water exponent
    ng = 2.0    # Gas exponent

    # =========================================
    # Initial and Boundary Conditions
    # =========================================
    initial_pressure = 1e6      # 10 bar [Pa]
    initial_temperature = 293.15  # 20°C [K]
    initial_sw = slr            # Initial water saturation = residual

    outlet_pressure = 1e6       # 10 bar [Pa] (fixed)
    inlet_water_rate = 1e-2     # 10 g/s water injection [kg/s]

    # =========================================
    # Simulation Time
    # =========================================
    # PVI = Pore Volume Injected
    # For 100m domain, A=1m², porosity=0.2: PV = 100 * 1 * 0.2 = 20 m³
    # At 0.1 g/s = 1e-4 kg/s ≈ 1e-7 m³/s (for water density 1000 kg/m³)
    # 1 PVI ≈ 20 / 1e-7 = 2e8 s ≈ 6.3 years
    # For faster simulation, use higher injection rate or shorter time

    # Use shorter time for demonstration
    end_time = 3600 * 24 * 10  # 10 days [s]
    dt_initial = 10.0          # Initial time step [s]

    # =========================================
    # Create Mesh
    # =========================================
    print("\n[1] Creating mesh...")
    mesh = create_mesh_1d(length=length, num_cells=num_cells, area=area)
    print(f"    Cells: {mesh.num_cells}")
    print(f"    Connections: {mesh.num_connections}")
    print(f"    Cell size: {length/num_cells:.2f} m")

    # Outlet boundary condition (large volume cell)
    mesh.cells[-1].volume *= 1e8

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
    print(f"    Initial pressure: {initial_pressure/1e5:.1f} bar")
    print(f"    Initial temperature: {initial_temperature - 273.15:.1f} °C")
    print(f"    Initial Sw: {initial_sw}")

    # =========================================
    # Source Terms
    # =========================================
    print("\n[5] Setting source terms...")
    num_eq = 2  # Water mass, Air mass
    source = np.zeros((mesh.num_cells, num_eq))

    # Water injection at first cell (positive = injection)
    source[0, 0] = inlet_water_rate   # Water mass rate [kg/s]
    source[0, 1] = 0.0                # No air injection

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

    # Final state
    pressures = state_manager.get_pressures()
    saturations = state_manager.get_liquid_saturations()

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Pressure range: {pressures.min()/1e5:.2f} - {pressures.max()/1e5:.2f} bar")
    print(f"Water saturation range: {saturations.min():.3f} - {saturations.max():.3f}")

    # Find front position (where Sw transitions)
    sw_mid = (saturations.max() + saturations.min()) / 2
    front_idx = np.argmin(np.abs(saturations - sw_mid))
    front_position = x[front_idx]
    print(f"Front position (approx): {front_position:.1f} m")

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

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Saturation profile
            ax1 = axes[0]
            ax1.plot(x, saturations, 'b-', linewidth=2, label='Water Saturation')
            ax1.axhline(y=slr, color='r', linestyle='--', label=f'Swr = {slr}')
            ax1.axhline(y=1-sgr, color='g', linestyle='--', label=f'1-Sgr = {1-sgr}')
            ax1.set_xlabel('Distance [m]')
            ax1.set_ylabel('Water Saturation [-]')
            ax1.set_title('Water Saturation Profile (Buckley-Leverett Type)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, length])
            ax1.set_ylim([0, 1])

            # Pressure profile
            ax2 = axes[1]
            ax2.plot(x, pressures/1e5, 'k-', linewidth=2)
            ax2.set_xlabel('Distance [m]')
            ax2.set_ylabel('Pressure [bar]')
            ax2.set_title('Pressure Profile')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([0, length])

            plt.tight_layout()

            output_file = os.path.join(output_dir, "water_air_saturation.png")
            plt.savefig(output_file, dpi=150)
            print(f"\nPlot saved to: {output_file}")

            plt.show()

        except ImportError:
            print("\nMatplotlib not available. Skipping plots.")

    return {
        'x': x,
        'pressure': pressures,
        'saturation': saturations,
        'times': results.get('times', []),
        'pressure_history': results.get('pressures', []),
        'saturation_history': results.get('saturations', [])
    }


def compute_buckley_leverett_analytical(
    slr: float = 0.2,
    sgr: float = 0.2,
    nl: float = 2.0,
    ng: float = 2.0,
    mu_w: float = 0.001,
    mu_g: float = 1.8e-5,
    num_points: int = 100
):
    """
    Compute analytical Buckley-Leverett solution for comparison.

    Args:
        slr: Residual liquid saturation
        sgr: Residual gas saturation
        nl, ng: Corey exponents
        mu_w, mu_g: Phase viscosities [Pa·s]
        num_points: Number of saturation points

    Returns:
        Dictionary with fractional flow curve and shock saturation
    """
    # Normalized saturation
    sw_range = np.linspace(slr + 0.001, 1 - sgr - 0.001, num_points)

    def effective_saturation(sw):
        return (sw - slr) / (1 - slr - sgr)

    def rel_perm_water(sw):
        se = effective_saturation(sw)
        return np.maximum(0, se ** nl)

    def rel_perm_gas(sw):
        se = effective_saturation(sw)
        return np.maximum(0, (1 - se) ** ng)

    # Fractional flow
    krw = rel_perm_water(sw_range)
    krg = rel_perm_gas(sw_range)

    # fw = 1 / (1 + (krg/mu_g) / (krw/mu_w))
    mobility_ratio = (krg / mu_g) / (krw / mu_w + 1e-20)
    fw = 1.0 / (1.0 + mobility_ratio)

    # Derivative dfw/dSw (numerical)
    dfw = np.gradient(fw, sw_range)

    # Find shock saturation using Welge tangent construction
    # The shock connects (Swf, fw(Swf)) to (Swr, 0)
    # Tangent condition: fw(Swf) / (Swf - Swr) = dfw/dSw at Swf

    # Find where tangent from origin intersects the curve
    tangent_slopes = fw / (sw_range - slr + 1e-10)

    # Find intersection (where tangent slope = derivative)
    diff = tangent_slopes - dfw
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        shock_idx = sign_changes[0]
        sw_shock = sw_range[shock_idx]
        fw_shock = fw[shock_idx]
    else:
        sw_shock = slr
        fw_shock = 0.0

    return {
        'sw': sw_range,
        'fw': fw,
        'dfw': dfw,
        'sw_shock': sw_shock,
        'fw_shock': fw_shock,
        'krw': krw,
        'krg': krg
    }


def create_animation(results: dict, output_path: str, length: float = 100.0,
                     slr: float = 0.2, sgr: float = 0.2, fps: int = 10):
    """
    Create animation of saturation front propagation.

    Args:
        results: Dictionary from run_water_air_example
        output_path: Path for output GIF/MP4 file
        length: Domain length [m]
        slr: Residual liquid saturation
        sgr: Residual gas saturation
        fps: Frames per second
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x = results['x']
    times = results['times']
    sat_history = results['saturation_history']
    pres_history = results['pressure_history']

    if len(times) == 0 or len(sat_history) == 0:
        print("No time history data available for animation.")
        return

    print(f"\nCreating animation with {len(times)} frames...")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Initialize plots
    line_sat, = ax1.plot([], [], 'b-', linewidth=2, label='Water Saturation')
    ax1.axhline(y=slr, color='r', linestyle='--', alpha=0.5, label=f'Swr = {slr}')
    ax1.axhline(y=1-sgr, color='g', linestyle='--', alpha=0.5, label=f'1-Sgr = {1-sgr}')
    ax1.set_xlabel('Distance [m]')
    ax1.set_ylabel('Water Saturation [-]')
    ax1.set_xlim([0, length])
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    line_pres, = ax2.plot([], [], 'k-', linewidth=2)
    ax2.set_xlabel('Distance [m]')
    ax2.set_ylabel('Pressure [bar]')
    ax2.set_xlim([0, length])
    ax2.grid(True, alpha=0.3)

    # Find pressure range for y-axis
    p_min = min(p.min() for p in pres_history) / 1e5
    p_max = max(p.max() for p in pres_history) / 1e5
    ax2.set_ylim([p_min * 0.95, p_max * 1.05])

    plt.tight_layout()

    def init():
        line_sat.set_data([], [])
        line_pres.set_data([], [])
        time_text.set_text('')
        return line_sat, line_pres, time_text

    def animate(frame):
        sat = sat_history[frame]
        pres = pres_history[frame] / 1e5  # Convert to bar
        t = times[frame]

        line_sat.set_data(x, sat)
        line_pres.set_data(x, pres)

        # Format time
        if t < 3600:
            time_str = f't = {t:.1f} s'
        elif t < 86400:
            time_str = f't = {t/3600:.2f} hours'
        else:
            time_str = f't = {t/86400:.2f} days'

        time_text.set_text(time_str)
        ax1.set_title(f'Water-Air Displacement (Buckley-Leverett Type)')

        return line_sat, line_pres, time_text

    # Select frames (subsample if too many)
    max_frames = 200
    if len(times) > max_frames:
        frame_indices = np.linspace(0, len(times)-1, max_frames, dtype=int)
    else:
        frame_indices = range(len(times))

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=frame_indices, interval=1000//fps, blit=True
    )

    # Save animation
    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        except:
            # Fallback to GIF if FFmpeg not available
            output_path = output_path.rsplit('.', 1)[0] + '.gif'
            writer = animation.PillowWriter(fps=fps)

    anim.save(output_path, writer=writer)
    print(f"Animation saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='1D Water-Air Injection Example')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    parser.add_argument('--animation', action='store_true', help='Create animation')
    args = parser.parse_args()

    results = run_water_air_example(
        plot_results=not args.no_plot,
        verbose=not args.quiet
    )

    if results is not None:
        print("\n[Analysis] Computing Buckley-Leverett analytical solution...")
        bl = compute_buckley_leverett_analytical()
        print(f"    Theoretical shock saturation: Swf = {bl['sw_shock']:.3f}")
        print(f"    Theoretical fractional flow at shock: fw = {bl['fw_shock']:.3f}")

        # Create animation if requested
        if args.animation:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "output")
            os.makedirs(output_dir, exist_ok=True)
            anim_path = os.path.join(output_dir, "water_air_animation.gif")
            create_animation(results, anim_path)
