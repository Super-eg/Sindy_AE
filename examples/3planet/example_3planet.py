"""Three-body gravitational simulation — Broucke-Hadjidemetriou-Henon (BHH) orbit.

The BHH family orbits are relative periodic. In an inertial frame, they trace out
beautiful quasi-periodic rosette (precession) patterns.

Run:
    python3 bhh_orbit.py

Outputs:
    bhh_three_body.npz
    (A matplotlib window will also pop up showing the trajectory)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── physical parameters ───────────────────────────────────────────────────────
G  = 1.0
m1 = 1.0
m2 = 1.0
m3 = 1.0

# ── time step for saved output ────────────────────────────────────────────────
DT_OUT = 0.01


# ── RHS ───────────────────────────────────────────────────────────────────────
def _rhs(t, s):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = s
    r12_3 = ((x2-x1)**2 + (y2-y1)**2) ** 1.5
    r13_3 = ((x3-x1)**2 + (y3-y1)**2) ** 1.5
    r23_3 = ((x3-x2)**2 + (y3-y2)**2) ** 1.5
    return [vx1, vy1, vx2, vy2, vx3, vy3,
            G*m2*(x2-x1)/r12_3 + G*m3*(x3-x1)/r13_3,
            G*m2*(y2-y1)/r12_3 + G*m3*(y3-y1)/r13_3,
            G*m1*(x1-x2)/r12_3 + G*m3*(x3-x2)/r23_3,
            G*m1*(y1-y2)/r12_3 + G*m3*(y3-y2)/r23_3,
            G*m1*(x1-x3)/r13_3 + G*m2*(x2-x3)/r23_3,
            G*m1*(y1-y3)/r13_3 + G*m2*(y2-y3)/r23_3]


# ── simulation ────────────────────────────────────────────────────────────────
def simulate(r1_0, r2_0, r3_0, v1_0, v2_0, v3_0, T_TOTAL):
    """Integrate with scipy DOP853 (adaptive step, high precision)."""
    s0 = np.concatenate([r1_0, r2_0, r3_0, v1_0, v2_0, v3_0])
    n_steps = int(round(T_TOTAL / DT_OUT)) + 1
    t_eval = np.linspace(0., T_TOTAL, n_steps)

    sol = solve_ivp(_rhs, [0., T_TOTAL], s0,
                    method='DOP853', t_eval=t_eval,
                    rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t, s = sol.t, sol.y
    return dict(t=t,
                x1=s[0], y1=s[1], x2=s[2], y2=s[3],
                x3=s[4], y3=s[5],
                vx1=s[6], vy1=s[7], vx2=s[8], vy2=s[9],
                vx3=s[10], vy3=s[11])


# ── BHH Family ICs (Inertial frame, center of mass at origin) ───────────────
# Internal Period T ≈ 9.1993 (G=1, m1=m2=m3=1)
T_PERIOD = 9.1993

_R2_0 = np.array([-1.21708465, 0.0])
_V2_0 = np.array([ 0.0, -0.89338778])

_R1_0 = np.array([ 1.10854233, 0.0])
_V1_0 = np.array([ 0.0, -0.28857029])

_R3_0 = np.array([ 0.10854233, 0.0])
_V3_0 = np.array([ 0.0,  1.18195807])


if __name__ == '__main__':
    # Increase N_PERIODS to clearly see the quasi-periodic precession
    N_PERIODS = 10
    T_TOTAL   = N_PERIODS * T_PERIOD

    print(f"G={G}  m1={m1}  m2={m2}  m3={m3}")
    print(f"Broucke-Hadjidemetriou-Henon (BHH) Family  (Internal T_period ≈ {T_PERIOD})")
    print(f"Simulating {N_PERIODS} internal periods  →  T_total={T_TOTAL:.4f}  (DT_out={DT_OUT})")

    data = simulate(_R1_0, _R2_0, _R3_0, _V1_0, _V2_0, _V3_0, T_TOTAL)
    
    # Save the data for SINDy
    np.savez('bhh_three_body.npz', **data)

    n = len(data['t'])
    print(f"Saved → bhh_three_body.npz  ({n} steps)")
    
    # ── plotting ───────────────────────────────────────────────────────────────
    print("Plotting trajectory...")
    plt.figure(figsize=(8, 8))
    
    # Plot trajectories
    plt.plot(data['x1'], data['y1'], label='m1', color='crimson', lw=0.8, alpha=0.8)
    plt.plot(data['x2'], data['y2'], label='m2', color='royalblue', lw=0.8, alpha=0.8)
    plt.plot(data['x3'], data['y3'], label='m3', color='forestgreen', lw=0.8, alpha=0.8)
    
    # Mark starting positions
    plt.scatter([_R1_0[0], _R2_0[0], _R3_0[0]], 
                [_R1_0[1], _R2_0[1], _R3_0[1]], 
                color=['crimson', 'royalblue', 'forestgreen'], 
                marker='o', s=40, zorder=5, label='Start Positions')

    plt.title("Broucke-Hadjidemetriou-Henon Family Orbit\n(Inertial Frame / Quasi-Periodic)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')  # Ensure the aspect ratio is 1:1
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()