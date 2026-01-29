"""
2D Heterogeneous Porous Media - Water Injection

Domain: 40cm x 60cm
- Random porosity (0.2 ~ 0.4)
- Top: 10 periodic injection ports (2cm each)
- Bottom: Periodic outlet ports (2cm each)
- Sides: No-flow boundaries
- Initial: Air saturated
- Injection: Water
- Simulation time: 10 seconds
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from porous.core.mesh import create_mesh_2d
from porous.core.properties import RockProperties
from porous.core.state import StateManager, FluidSystem
from porous.physics.relative_perm import CoreyRelPerm
from porous.physics.capillary import VanGenuchtenCapillary
from porous.solver.newton import Simulator


def run_heterogeneous_example():
    print("=" * 60)
    print("2D Heterogeneous Porous Media - Water Injection")
    print("=" * 60)

    # =========================================
    # Domain Parameters (cm -> m)
    # =========================================
    lx = 0.40  # 40 cm
    ly = 0.60  # 60 cm
    nx = 20    # 2cm resolution in x
    ny = 30    # 2cm resolution in y
    thickness = 0.01  # 1 cm thickness

    dx = lx / nx  # 0.01 m = 1 cm
    dy = ly / ny  # 0.01 m = 1 cm

    print(f"\n[1] Domain: {lx*100:.0f}cm x {ly*100:.0f}cm")
    print(f"    Grid: {nx} x {ny} = {nx*ny} cells")
    print(f"    Cell size: {dx*100:.1f}cm x {dy*100:.1f}cm")

    # =========================================
    # Create Mesh
    # =========================================
    mesh = create_mesh_2d(lx=lx, ly=ly, nx=nx, ny=ny, thickness=thickness)

    # =========================================
    # Random Porosity Field (0.2 ~ 0.4)
    # =========================================
    print("\n[2] Creating heterogeneous porosity field...")
    np.random.seed(42)  # For reproducibility

    # Create smooth random field using multiple scales
    porosity_field = np.zeros((ny, nx))

    # Base random field
    porosity_base = 0.3 + 0.1 * (np.random.rand(ny, nx) - 0.5) * 2

    # Add larger scale variations (correlation)
    from scipy.ndimage import gaussian_filter
    porosity_smooth = gaussian_filter(np.random.rand(ny, nx), sigma=3)
    porosity_smooth = 0.2 + 0.2 * (porosity_smooth - porosity_smooth.min()) / (porosity_smooth.max() - porosity_smooth.min() + 1e-10)

    # Combine
    porosity_field = 0.5 * porosity_base + 0.5 * porosity_smooth
    porosity_field = np.clip(porosity_field, 0.2, 0.4)

    print(f"    Porosity range: {porosity_field.min():.3f} ~ {porosity_field.max():.3f}")

    # =========================================
    # Rock Properties (heterogeneous)
    # =========================================
    print("\n[3] Setting heterogeneous rock properties...")

    # Permeability correlates with porosity (Kozeny-Carman like)
    # k ~ phi^3 / (1-phi)^2
    permeability_field = np.zeros((ny, nx))
    k_ref = 1e-10  # Reference permeability for phi=0.3

    for j in range(ny):
        for i in range(nx):
            phi = porosity_field[j, i]
            # Simplified correlation
            permeability_field[j, i] = k_ref * (phi / 0.3) ** 3

    rocks = []
    for j in range(ny):
        for i in range(nx):
            cell_idx = i + j * nx
            phi = porosity_field[j, i]
            k = permeability_field[j, i]
            rock = RockProperties(
                porosity=phi,
                permeability=np.array([k, k, k])
            )
            rocks.append(rock)

    print(f"    Permeability range: {permeability_field.min()*1e12:.2f} ~ {permeability_field.max()*1e12:.2f} x10^-12 m²")

    # =========================================
    # Boundary Conditions
    # =========================================
    print("\n[4] Setting boundary conditions...")

    # Injection ports at top (y = ly)
    # 10 ports, each 2cm wide, periodically spaced
    # Total width = 40cm, 10 ports x 2cm = 20cm injection, 20cm wall
    # Pattern: 2cm wall, 2cm port, 2cm wall, 2cm port, ...

    port_width = 2  # cells (2cm)
    wall_width = 2  # cells (2cm)
    period = port_width + wall_width  # 4 cells

    injection_cells = []
    outlet_cells = []

    # Top row (j = ny-1): injection ports
    j_top = ny - 1
    for i in range(nx):
        cell_idx = i + j_top * nx
        # Check if this cell is in an injection port
        pos_in_period = i % period
        if pos_in_period < port_width:
            injection_cells.append(cell_idx)

    # Bottom row (j = 0): outlet ports (same pattern, offset by half period)
    j_bottom = 0
    for i in range(nx):
        cell_idx = i + j_bottom * nx
        # Offset pattern for outlets
        pos_in_period = (i + period // 2) % period
        if pos_in_period < port_width:
            outlet_cells.append(cell_idx)
            # Large volume for fixed pressure BC
            mesh.cells[cell_idx].volume *= 1e6

    print(f"    Injection ports (top): {len(injection_cells)} cells")
    print(f"    Outlet ports (bottom): {len(outlet_cells)} cells")

    # =========================================
    # State Manager
    # =========================================
    print("\n[5] Initializing states...")

    rel_perm = CoreyRelPerm(slr=0.1, sgr=0.1, nl=2.0, ng=2.0)
    cap_pressure = VanGenuchtenCapillary(slr=0.1, alpha=1e-3, m=0.5)

    state_manager = StateManager(
        mesh.num_cells,
        default_rel_perm=rel_perm,
        default_cap_pressure=cap_pressure,
        fluid_system=FluidSystem.WATER_AIR
    )

    # Initial condition: air saturated (Sw = residual)
    initial_pressure = 1e5  # 1 bar
    initial_sw = 0.1  # Residual water saturation

    state_manager.initialize_uniform(
        pressure=initial_pressure,
        temperature=293.15,
        liquid_saturation=initial_sw
    )

    # Set outlet cells to fixed lower pressure
    outlet_pressure = 0.95e5  # 0.95 bar
    for cell_idx in outlet_cells:
        state_manager.states[cell_idx].pressure = outlet_pressure
        state_manager.states[cell_idx].update_secondary_variables()

    print(f"    Initial Sw: {initial_sw}")
    print(f"    Initial P: {initial_pressure/1e5:.2f} bar")
    print(f"    Outlet P: {outlet_pressure/1e5:.2f} bar")

    # =========================================
    # Source Terms (Water Injection)
    # =========================================
    print("\n[6] Setting injection source terms...")

    num_eq = 2
    source = np.zeros((mesh.num_cells, num_eq))

    # Injection rate per port cell
    # Total injection: ~10 mL/s = 0.01 kg/s (for water)
    total_injection_rate = 0.01  # kg/s
    rate_per_cell = total_injection_rate / len(injection_cells)

    for cell_idx in injection_cells:
        source[cell_idx, 0] = rate_per_cell  # Water injection

    print(f"    Total injection rate: {total_injection_rate*1000:.1f} g/s")
    print(f"    Rate per cell: {rate_per_cell*1000:.3f} g/s")

    # =========================================
    # Run Simulation
    # =========================================
    print("\n[7] Running simulation...")

    end_time = 3.0  # 3 seconds (reduced for faster completion)
    dt_initial = 0.0005  # 0.5 ms

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "output")
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "heterogeneous_log.csv")

    sim = Simulator(mesh, state_manager, rocks, log_file=log_file)
    sim.set_simulation_time(end_time=end_time, dt_initial=dt_initial)
    sim.set_source_terms(source)

    success = sim.run(verbose=True)

    if not success:
        print("Simulation failed!")
        return None

    print(f"\n[8] Simulation completed!")

    # =========================================
    # Results
    # =========================================
    times = np.array(sim.times)
    sat_history = [s.copy() for s in sim.saturations]
    pres_history = [p.copy() for p in sim.pressures]

    final_sat = sat_history[-1]
    print(f"    Final Sw range: {final_sat.min():.3f} ~ {final_sat.max():.3f}")

    # =========================================
    # Create Visualizations
    # =========================================
    print("\n[9] Creating visualizations...")

    X, Y = np.meshgrid(
        np.linspace(0, lx*100, nx),  # cm
        np.linspace(0, ly*100, ny)   # cm
    )

    # 1. Initial Porosity Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 10))
    cs1 = ax1.contourf(X, Y, porosity_field, levels=20, cmap='viridis')
    plt.colorbar(cs1, ax=ax1, label='Porosity [-]')
    ax1.set_xlabel('X [cm]')
    ax1.set_ylabel('Y [cm]')
    ax1.set_title('Initial Porosity Distribution')
    ax1.set_aspect('equal')

    # Mark injection/outlet ports
    for cell_idx in injection_cells:
        i = cell_idx % nx
        j = cell_idx // nx
        ax1.plot((i+0.5)*dx*100, (j+0.5)*dy*100, 'rv', markersize=3)
    for cell_idx in outlet_cells:
        i = cell_idx % nx
        j = cell_idx // nx
        ax1.plot((i+0.5)*dx*100, (j+0.5)*dy*100, 'b^', markersize=3)

    fig1.savefig(os.path.join(output_dir, 'heterogeneous_porosity.png'), dpi=150)
    print(f"    Saved: heterogeneous_porosity.png")
    plt.close(fig1)

    # 2. Final Saturation Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    sat_2d = final_sat.reshape((ny, nx))
    cs2 = ax2.contourf(X, Y, sat_2d, levels=np.linspace(0.1, 0.9, 20), cmap='Blues')
    plt.colorbar(cs2, ax=ax2, label='Water Saturation [-]')
    ax2.set_xlabel('X [cm]')
    ax2.set_ylabel('Y [cm]')
    ax2.set_title(f'Water Saturation at t = {end_time:.1f} s')
    ax2.set_aspect('equal')
    fig2.savefig(os.path.join(output_dir, 'heterogeneous_saturation_final.png'), dpi=150)
    print(f"    Saved: heterogeneous_saturation_final.png")
    plt.close(fig2)

    # 3. Animation of Saturation Evolution
    print("    Creating animation...")

    fig3, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Select frames
    n_frames = min(50, len(times))
    frame_idx = np.linspace(0, len(times)-1, n_frames, dtype=int)

    def animate(frame):
        for ax in axes:
            ax.clear()

        idx = frame_idx[frame]
        t = times[idx]

        # Saturation
        sat_2d = sat_history[idx].reshape((ny, nx))
        cs1 = axes[0].contourf(X, Y, sat_2d, levels=np.linspace(0.1, 0.9, 20), cmap='Blues')
        axes[0].set_xlabel('X [cm]')
        axes[0].set_ylabel('Y [cm]')
        axes[0].set_title(f'Water Saturation (t = {t:.2f} s)')
        axes[0].set_aspect('equal')

        # Porosity (static reference)
        cs2 = axes[1].contourf(X, Y, porosity_field, levels=20, cmap='viridis')
        axes[1].contour(X, Y, sat_2d, levels=[0.3, 0.5, 0.7], colors='white', linewidths=1)
        axes[1].set_xlabel('X [cm]')
        axes[1].set_ylabel('Y [cm]')
        axes[1].set_title('Porosity with Saturation Contours')
        axes[1].set_aspect('equal')

        return []

    anim = animation.FuncAnimation(fig3, animate, frames=n_frames, interval=100)
    anim_path = os.path.join(output_dir, 'heterogeneous_animation.gif')
    anim.save(anim_path, writer=animation.PillowWriter(fps=10))
    print(f"    Saved: heterogeneous_animation.gif")
    plt.close(fig3)

    # 4. Velocity field (approximate from pressure gradient)
    print("    Computing velocity field...")

    fig4, ax4 = plt.subplots(figsize=(8, 10))

    # Compute approximate velocity from pressure gradient
    pres_2d = pres_history[-1].reshape((ny, nx))

    # Gradient (dP/dx, dP/dy)
    dPdx = np.gradient(pres_2d, dx, axis=1)
    dPdy = np.gradient(pres_2d, dy, axis=0)

    # Darcy velocity: v = -k/mu * grad(P)
    mu_water = 0.001  # Pa.s
    vx = -permeability_field / mu_water * dPdx
    vy = -permeability_field / mu_water * dPdy

    # Velocity magnitude
    v_mag = np.sqrt(vx**2 + vy**2)

    # Plot
    cs4 = ax4.contourf(X, Y, v_mag * 100, levels=20, cmap='hot')  # cm/s
    plt.colorbar(cs4, ax=ax4, label='Velocity magnitude [cm/s]')

    # Streamlines
    skip = 2
    ax4.streamplot(X[::skip, ::skip], Y[::skip, ::skip],
                   vx[::skip, ::skip]*100, vy[::skip, ::skip]*100,
                   color='white', linewidth=0.5, density=1.5)

    ax4.set_xlabel('X [cm]')
    ax4.set_ylabel('Y [cm]')
    ax4.set_title('Velocity Field (Darcy)')
    ax4.set_aspect('equal')

    fig4.savefig(os.path.join(output_dir, 'heterogeneous_velocity.png'), dpi=150)
    print(f"    Saved: heterogeneous_velocity.png")
    plt.close(fig4)

    print("\n" + "=" * 60)
    print("All outputs saved to:", output_dir)
    print("=" * 60)

    return {
        'times': times,
        'saturation_history': sat_history,
        'pressure_history': pres_history,
        'porosity_field': porosity_field,
        'permeability_field': permeability_field,
        'nx': nx, 'ny': ny, 'lx': lx, 'ly': ly
    }


if __name__ == "__main__":
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Installing scipy for gaussian filter...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])

    results = run_heterogeneous_example()
