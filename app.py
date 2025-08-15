import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

EPS0 = 8.854187817e-12  # F/m

st.set_page_config(page_title="MEMS-Lab: Real-Time Interactive MEMS Simulator", layout="wide")

st.title("MEMS-Lab: Real-Time Interactive MEMS Simulator")
st.caption("No datasets. No ML. Pure physics. Built for FI1937 (Anna University, 2021 Regulation).")

tabs = st.tabs([
    "Parallel-Plate Pull-In",
    "Comb-Drive Resonator",
    "Capacitive Accelerometer",
    "Optical MEMS Mirror"
])

# ---------- Tab 1: Parallel-Plate Pull-In ----------
with tabs[0]:
    st.header("Parallel-Plate Electrostatic Actuator: Pull-In Instability")
    col1, col2 = st.columns([1,1])

    with col1:
        A = st.number_input("Plate Area A (μm²)", value=10000.0, min_value=100.0, step=100.0) * 1e-12  # m²
        g0 = st.number_input("Initial Gap g₀ (μm)", value=2.0, min_value=0.2, step=0.1) * 1e-6        # m
        k = st.number_input("Spring Constant k (N/m)", value=1.0, min_value=0.01, step=0.05)         # N/m
        eps_r = st.number_input("Relative Permittivity εr", value=1.0, min_value=1.0, step=0.1)
        V = st.slider("Applied Voltage V (V)", min_value=0.0, max_value=200.0, value=20.0, step=1.0)

    with col2:
        # Pull-in analytics
        V_pi = math.sqrt((8.0*k*(g0**3)) / (27.0*EPS0*eps_r*A))
        x_pi = g0/3.0
        st.markdown(f"**Pull-In Displacement:** xₚᵢ = g₀/3 = {x_pi*1e6:.3f} μm")
        st.markdown(f"**Pull-In Voltage:** Vₚᵢ = {V_pi:.3f} V")

        # Force curves
        x = np.linspace(0.0, min(0.95*g0, 0.999*g0), 600)
        F_spring = k * x
        # add a tiny epsilon to avoid division by zero near g0
        F_elec = 0.5 * EPS0 * eps_r * A * (V**2) / (np.maximum(g0 - x, 1e-12)**2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=1e6*x, y=F_spring, name="Spring Force kx", mode="lines"))
        fig.add_trace(go.Scatter(x=1e6*x, y=F_elec, name="Electrostatic Force", mode="lines"))
        fig.update_layout(
            xaxis_title="Displacement x (μm)",
            yaxis_title="Force (N)",
            title="Force Balance: Intersections are Equilibria (Rightmost one becomes unstable near pull-in)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Find intersections (equilibria)
        diff = F_spring - F_elec
        sign_change = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
        if len(sign_change) == 0:
            st.warning("No force-balance intersection found → **actuator is unstable** (likely pulled-in).")
        else:
            xs = []
            for idx in sign_change:
                # linear interpolation for intersection
                x0, x1 = x[idx], x[idx+1]
                y0, y1 = diff[idx], diff[idx+1]
                xi = x0 - y0 * (x1 - x0) / (y1 - y0)
                xs.append(xi)
            st.success(f"Estimated equilibrium positions (μm): {', '.join([f'{v*1e6:.3f}' for v in xs])}")

# ---------- Tab 2: Comb-Drive Resonator ----------
with tabs[1]:
    st.header("Electrostatic Comb-Drive Resonator: Time Response & FRF")
    col1, col2 = st.columns([1,1])

    with col1:
        N = st.number_input("Number of finger pairs N", value=80, min_value=1, step=1)
        h_um = st.number_input("Finger Height h (μm)", value=10.0, min_value=2.0, step=1.0)
        g_um = st.number_input("Finger Gap g (μm)", value=2.0, min_value=0.5, step=0.1)
        m = st.number_input("Effective Mass m (μg)", value=2.0, min_value=0.1, step=0.1) * 1e-9   # kg
        k = st.number_input("Spring Constant k (N/m)", value=0.5, min_value=0.01, step=0.01)
        zeta = st.slider("Damping Ratio ζ", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
        Vdc = st.slider("Vdc (V)", 0.0, 60.0, 20.0, 1.0)
        Vac = st.slider("Vac (V)", 0.0, 20.0, 5.0, 0.5)
        f_drive = st.slider("Drive Frequency f (kHz)", 0.1, 500.0, 10.0, 0.1)
        t_end_ms = st.slider("Simulation Time (ms)", 1.0, 200.0, 50.0, 1.0)

    with col2:
        h = h_um * 1e-6
        g = g_um * 1e-6
        c = 2.0 * zeta * math.sqrt(k*m)

        def V_t(t):
            return Vdc + Vac*math.cos(2.0*math.pi*f_drive*1e3*t)

        # dC/dx ≈ 2 N ε0 h / g  (two facing sidewalls per finger pair)
        dCdx = 2.0 * N * EPS0 * h / g

        def F_e(t):
            V = V_t(t)
            return 0.5 * (V**2) * dCdx  # electrostatic force along x

        def ode(t, y):
            x, xdot = y
            return [xdot, (F_e(t) - c*xdot - k*x)/m]

        t_span = (0.0, t_end_ms/1000.0)
        t_eval = np.linspace(t_span[0], t_span[1], 4000)
        sol = solve_ivp(ode, t_span, y0=[0.0, 0.0], t_eval=t_eval, rtol=1e-7, atol=1e-9)
        x = sol.y[0]
        t_ms = sol.t * 1e3

        wn = math.sqrt(k/m)                # rad/s
        fn = wn/(2.0*math.pi)              # Hz
        Q = 1.0/(2.0*zeta)
        st.markdown(f"**Natural Frequency:** fₙ = {fn/1e3:.3f} kHz  |  **Q ≈** {Q:.1f}")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t_ms, y=1e6*x, mode="lines", name="x(t)"))
        fig1.update_layout(xaxis_title="Time (ms)", yaxis_title="Displacement (μm)",
                           title="Time-Domain Response")
        st.plotly_chart(fig1, use_container_width=True)

        # Frequency response (small-signal around Vdc): drive ~ 2*Vdc*Vac*cos(ωt)
        F0 = 2.0 * Vdc * Vac * dCdx
        freqs = np.linspace(0.1e3, 500e3, 1000)  # Hz
        w = 2.0*np.pi*freqs
        # |X(jw)| = F0 / sqrt((k - m w^2)^2 + (c w)^2)
        Xmag = F0 / np.sqrt((k - m*w**2)**2 + (c*w)**2)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freqs/1e3, y=1e6*Xmag, mode="lines", name="|X|"))
        fig2.add_vline(x=fn/1e3, line_dash="dash", annotation_text="fₙ", annotation_position="top")
        fig2.update_layout(xaxis_title="Frequency (kHz)", yaxis_title="Amplitude (μm per N-equivalent)",
                           title="Small-Signal Frequency Response (around Vdc)")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Tab 3: Capacitive Accelerometer ----------
with tabs[2]:
    st.header("Capacitive Accelerometer: Mechanics + Simple Charge Readout")
    col1, col2 = st.columns([1,1])

    with col1:
        m = st.number_input("Proof Mass m (μg)", value=5.0, min_value=0.1, step=0.1)*1e-9
        k = st.number_input("Spring Constant k (N/m)", value=0.8, min_value=0.01, step=0.01)
        zeta = st.slider("Damping Ratio ζ", 0.005, 0.25, 0.03, 0.001)
        A = st.number_input("Electrode Area A (μm²)", value=40000.0, min_value=1000.0, step=500.0)*1e-12
        g0 = st.number_input("Initial Gap g₀ (μm)", value=2.0, min_value=0.5, step=0.1)*1e-6
        Vbias = st.slider("Sense Bias Vbias (V)", 0.0, 10.0, 3.3, 0.1)
        Cf = st.number_input("Feedback Cap Cf (pF)", value=2.0, min_value=0.1, step=0.1)*1e-12
        stim = st.selectbox("Input Acceleration a(t)", ["Step (±g)", "Sine"], index=0)
        g_mult = st.slider("Step Level (±g)", 0.1, 5.0, 1.0, 0.1)
        f_sine = st.slider("Sine Frequency (Hz)", 1.0, 2000.0, 100.0, 1.0)
        t_end_ms = st.slider("Simulation Time (ms)", 50.0, 1000.0, 300.0, 10.0)

    with col2:
        c = 2.0*zeta*math.sqrt(k*m)
        def a_in(t):
            if stim == "Step (±g)":
                return g_mult*9.80665 if t >= 0.5e-3 else 0.0
            else:
                return (0.5*9.80665)*math.sin(2.0*math.pi*f_sine*t)

        def ode(t, y):
            x, xdot = y
            return [xdot, (-m*a_in(t) - c*xdot - k*x)/m]  # no electrostatic actuation in this tab

        t_span = (0.0, t_end_ms/1000.0)
        t_eval = np.linspace(t_span[0], t_span[1], 4000)
        sol = solve_ivp(ode, t_span, [0.0, 0.0], t_eval=t_eval, rtol=1e-7, atol=1e-9)
        t = sol.t
        x = sol.y[0]

        # Differential capacitance (two caps in anti-series): ΔC ≈ ε0 A (1/(g0-x) - 1/(g0+x))
        # small-x linear approx: ΔC ≈ 2 ε0 A x / g0^2
        deltaC_exact = EPS0*A*((1.0/np.maximum(g0 - x, 1e-12)) - (1.0/np.maximum(g0 + x, 1e-12)))
        deltaC_lin = 2.0*EPS0*A*x/(g0**2)
        # Simple charge amp model: Vout ≈ (Vbias * ΔC)/Cf
        Vout_exact = Vbias*deltaC_exact/np.maximum(Cf, 1e-18)
        Vout_lin = Vbias*deltaC_lin/np.maximum(Cf, 1e-18)

        wn = math.sqrt(k/m); fn = wn/(2.0*math.pi)
        st.markdown(f"**fₙ ≈ {fn:.1f} Hz**, **ωₙ ≈ {wn:.1f} rad/s**, **Q ≈ {1/(2*zeta):.1f}**")

        figx = go.Figure()
        figx.add_trace(go.Scatter(x=1e3*t, y=1e6*x, mode="lines", name="x(t)"))
        figx.update_layout(xaxis_title="Time (ms)", yaxis_title="Displacement (μm)", title="Proof-Mass Displacement")
        st.plotly_chart(figx, use_container_width=True)

        figv = go.Figure()
        figv.add_trace(go.Scatter(x=1e3*t, y=Vout_exact, mode="lines", name="Vout (exact ΔC)"))
        figv.add_trace(go.Scatter(x=1e3*t, y=Vout_lin, mode="lines", name="Vout (linearized)", line=dict(dash="dash")))
        figv.update_layout(xaxis_title="Time (ms)", yaxis_title="Vout (V)", title="Charge-Amplifier Output")
        st.plotly_chart(figv, use_container_width=True)

# ---------- Tab 4: Optical MEMS Mirror ----------
with tabs[3]:
    st.header("Optical MEMS: Torsional Mirror (Small-Angle Model)")
    col1, col2 = st.columns([1,1])

    with col1:
        Ae = st.number_input("Electrode Area Aₑ (μm²)", value=20000.0, min_value=1000.0, step=500.0)*1e-12
        g0 = st.number_input("Electrode Gap g₀ (μm)", value=2.0, min_value=0.5, step=0.1)*1e-6
        d = st.number_input("Electrode Lever Arm d (μm)", value=50.0, min_value=5.0, step=1.0)*1e-6
        Ktheta = st.number_input("Torsional Stiffness Kθ (nN·m/rad)", value=50.0, min_value=1.0, step=1.0)*1e-9
        V = st.slider("Voltage V (V)", 0.0, 120.0, 20.0, 1.0)
        L_beam = st.number_input("Optical Path Length (mm)", value=50.0, min_value=5.0, step=1.0)*1e-3
        n_env = st.number_input("Refractive Index (environment)", value=1.0, min_value=1.0, step=0.01)

    with col2:
        # Very simple small-angle, small-deflection torque approx:
        # τ_e ≈ (ε0 * A_e * V^2 * d) / (2 * g0^2)
        tau_e = (EPS0 * Ae * (V**2) * d) / (2.0 * g0**2)
        theta = tau_e / Ktheta  # rad
        scan_angle = 2.0*theta  # reflected beam deflection ≈ 2θ
        st.markdown(f"**Mirror Angle θ:** {theta*180/math.pi:.3f}°  |  **Beam Deflection ≈ 2θ:** {scan_angle*180/math.pi:.3f}°")

        # Draw incoming and reflected rays
        # Incoming along +x, mirror at origin
        # Reflected ray deflects by 2θ
        x_line = np.linspace(0, L_beam, 200)
        y_in = np.zeros_like(x_line)
        y_out = np.tan(scan_angle) * x_line

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=1e3*x_line, y=y_in*1e3, mode="lines", name="Incoming Ray"))
        fig.add_trace(go.Scatter(x=1e3*x_line, y=y_out*1e3, mode="lines", name="Reflected Ray"))
        # mirror line (rotated by θ around origin)
        mirror_len = L_beam*0.05
        xm = np.array([-mirror_len, mirror_len])
        ym = np.tan(theta) * xm
        fig.add_trace(go.Scatter(x=1e3*xm, y=1e3*ym, mode="lines", name="Mirror"))
        fig.update_layout(xaxis_title="Distance (mm)", yaxis_title="Height (mm)",
                          title="Optical Steering (small-angle model)")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
**Notes (for viva):**
- **Pull-In:** At equilibrium, \(k x = \\tfrac{1}{2}\\,\\varepsilon_0\\varepsilon_r A V^2/(g_0-x)^2\). Pull-in at \(x=g_0/3\), \(V_{PI}=\\sqrt{\\tfrac{8 k g_0^3}{27\\varepsilon_0\\varepsilon_r A}}\).
- **Comb Drive:** \(F_x \\approx \\tfrac{1}{2}V^2\\,\\tfrac{dC}{dx},\\; \\tfrac{dC}{dx}\\approx 2N\\varepsilon_0 h/g\\). With \(V=V_{dc}+V_{ac}\\cos\\omega t\), the dominant drive near \\(\\omega\\) is \(\\approx 2V_{dc}V_{ac}\\tfrac{dC}{dx}\\cos\\omega t\).
- **Accelerometer:** \(m\\ddot{x}+c\\dot{x}+kx=-m a(t)\\). Differential capacitance \(\\Delta C\\approx 2\\varepsilon_0 A x/g_0^2\\) (small-x). Charge amp \(V_{out}\\approx V_{bias}\\Delta C/C_f\\).
- **Optical Mirror:** Small-angle torsion: \(\\tau_e \\approx \\tfrac{\\varepsilon_0 A_e V^2 d}{2 g_0^2},\\; \\theta=\\tau_e/K_\\theta,\\; \\text{beam deflection}\\approx 2\\theta\\).
""")
