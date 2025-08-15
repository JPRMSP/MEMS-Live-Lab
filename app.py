import json
import io
import math
import numpy as np
import streamlit as st
from dataclasses import asdict, dataclass
import plotly.graph_objs as go

st.set_page_config(page_title="MEMS Live Lab", page_icon="🧪", layout="wide")

# ---------- Utilities ----------
EPS0 = 8.854187817e-12  # F/m

def nice_plot(x, ys, names, xlab, ylab):
    fig = go.Figure()
    for y, name in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def rk4(f, y0, t):
    y = np.zeros((len(t), len(np.atleast_1d(y0))))
    y[0] = y0
    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i]+0.5*h, y[i]+0.5*h*k1)
        k3 = f(t[i]+0.5*h, y[i]+0.5*h*k2)
        k4 = f(t[i]+h, y[i]+h*k3)
        y[i+1] = y[i] + (h/6.0)*(k1+2*k2+2*k3+k4)
    return y

def download_text_button(label, text, file_name):
    st.download_button(label, text.encode("utf-8"), file_name=file_name, mime="text/plain")

# ---------- Sidebar ----------
st.sidebar.title("MEMS Live Lab")
st.sidebar.caption("No datasets. No ML. Pure physics.")
st.sidebar.write("Pick a module from the tabs →")
with st.sidebar.expander("📤 Export current design"):
    st.session_state.setdefault("designs", {})
    name = st.text_input("Design name", "demo_design")
    if st.button("Save design snapshot"):
        st.session_state["designs"][name] = st.session_state.get("snapshot", {})
        st.success(f"Saved snapshot '{name}'.")
    if st.session_state["designs"]:
        options = list(st.session_state["designs"].keys())
        choice = st.selectbox("Download which?", options)
        if st.button("Download JSON"):
            download_text_button("Download JSON now",
                                 json.dumps(st.session_state["designs"][choice], indent=2),
                                 f"{choice}.json")

st.title("🧪 MEMS Live Lab")
st.caption("Interactive calculators & simulators for Anna University FI1937 (MEMS).")

tabs = st.tabs([
    "Electrostatic Pull-In", "Comb-Drive Designer", "Capacitive Accelerometer",
    "Thermal Actuator", "Thin Plate Bending", "Optical MEMS Attenuator", "Process Planner"
])

# ---------- 1) Electrostatic Pull-In ----------
with tabs[0]:
    st.header("Electrostatic Pull-In Visualizer")
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input("Spring stiffness k (N/m)", 0.1, 1e6, 10_000.0, step=100.0, format="%.3f")
        g0_um = st.number_input("Initial gap g₀ (µm)", 0.1, 100.0, 2.0, step=0.1, format="%.3f")
        A_um2 = st.number_input("Plate area A (µm²)", 1.0, 1e12, 200*200.0, step=100.0, format="%.3f")
        epsr = st.number_input("Dielectric εr", 1.0, 12.0, 1.0, step=0.1, format="%.2f")
    with col2:
        V = st.slider("Applied voltage V (V)", 0.0, 300.0, 0.0, 1.0)
        show_curves = st.multiselect("Show", ["Force–disp", "Effective k", "Energy"], default=["Force–disp","Effective k"])
        st.info("Pull-in occurs near x ≈ g₀/3 where the electrostatic negative stiffness equals mechanical stiffness.")

    g0 = g0_um * 1e-6
    A = A_um2 * 1e-12
    eps = epsr * EPS0

    # Pull-in voltage (parallel-plate)
    Vpi = math.sqrt((8.0 * k * g0**3) / (27.0 * eps * A))
    st.metric("Estimated Pull-In Voltage Vπ (V)", f"{Vpi:.2f}")

    x = np.linspace(0, min(0.99*g0, 0.99*g0), 600)
    Fe = 0.5 * eps * A * (V**2) / (g0 - x)**2  # electrostatic attractive force
    Fk = k * x
    keff = k - (eps * A * V**2) / (g0 - x)**3  # effective stiffness
    Ue = -0.5 * eps * A * (V**2) / (g0 - x)   # electrostatic energy (relative)
    Um = 0.5 * k * x**2

    if "Force–disp" in show_curves:
        fig = nice_plot(x*1e6, [Fk, Fe], ["Mechanical (kx)", "Electrostatic"], "Displacement x (µm)", "Force (N)")
        st.plotly_chart(fig, use_container_width=True)
    if "Effective k" in show_curves:
        fig2 = nice_plot(x*1e6, [keff], ["k_eff"], "Displacement x (µm)", "Effective stiffness (N/m)")
        st.plotly_chart(fig2, use_container_width=True)
    if "Energy" in show_curves:
        fig3 = nice_plot(x*1e6, [Um, Ue, Um+Ue], ["U_mech", "U_elec", "Total"], "Displacement x (µm)", "Energy (J, relative)")
        st.plotly_chart(fig3, use_container_width=True)

    st.session_state["snapshot"] = {
        "electrostatic": dict(k=k, g0_um=g0_um, A_um2=A_um2, epsr=epsr, V=V, Vpi=Vpi)
    }

# ---------- 2) Comb-Drive ----------
with tabs[1]:
    st.header("Comb-Drive Lateral Actuator Designer")
    c1, c2 = st.columns(2)
    with c1:
        N = st.number_input("Number of finger pairs N", 1, 2000, 200)
        h_um = st.number_input("Finger thickness h (µm)", 1.0, 200.0, 10.0, step=0.5)
        g_um = st.number_input("Gap g (µm)", 0.2, 20.0, 2.0, step=0.1)
        Vc = st.slider("Voltage V (V)", 0.0, 300.0, 30.0, 1.0)
    with c2:
        overlap_um = st.number_input("Overlap length L_overlap (µm)", 1.0, 1000.0, 50.0, step=1.0)
        k_lateral = st.number_input("Lateral spring k (N/m)", 10.0, 1e6, 2000.0, step=10.0)
        travel_um = st.number_input("Simulated travel range (µm)", 1.0, 20.0, 5.0, step=0.5)

    h = h_um*1e-6
    g = g_um*1e-6

    # Force (approx): F = N * ε0 * h * V^2 / (2g)
    F = N * EPS0 * h * Vc**2 / (2.0 * g)
    xeq = F / k_lateral  # equilibrium displacement
    st.metric("Lateral force (µN)", f"{F*1e6:.3f}")
    st.metric("Equilibrium displacement (µm)", f"{xeq*1e6:.3f}")

    Vscan = np.linspace(0, 300.0, 500)
    Fscan = N * EPS0 * h * Vscan**2 / (2.0 * g)
    fig = nice_plot(Vscan, [Fscan*1e6], ["Force"], "Voltage (V)", "Force (µN)")
    st.plotly_chart(fig, use_container_width=True)

    x = np.linspace(0, travel_um*1e-6, 400)
    U = 0.5*k_lateral*x**2 - F*x
    fig2 = nice_plot(x*1e6, [U], ["Potential"], "Displacement (µm)", "Energy (J, relative)")
    st.plotly_chart(fig2, use_container_width=True)

    st.session_state["snapshot"].update({
        "comb_drive": dict(N=N, h_um=h_um, g_um=g_um, V=Vc, k_lateral=k_lateral, force_N=F, xeq_um=xeq*1e6)
    })

# ---------- 3) Capacitive Accelerometer ----------
with tabs[2]:
    st.header("Capacitive Accelerometer — m-c-k + charge amp + noise")
    col1, col2, col3 = st.columns(3)
    with col1:
        m = st.number_input("Proof mass m (mg)", 0.01, 1000.0, 10.0, step=0.01)/1000.0
        fn = st.number_input("Natural freq fₙ (Hz)", 1.0, 5000.0, 400.0, step=1.0)
        Q = st.number_input("Q factor", 0.1, 200.0, 20.0, step=0.1)
    with col2:
        g0_um_acc = st.number_input("Gap g₀ (µm)", 0.2, 10.0, 2.0, step=0.1)
        A_um2_acc = st.number_input("Electrode area A (µm²)", 100.0, 1e9, 200*200.0, step=100.0)
        Vbias = st.number_input("Bias voltage Vbias (V)", 0.0, 50.0, 3.3, step=0.1)
    with col3:
        Cf = st.number_input("Feedback capacitor C_f (pF)", 0.01, 10000.0, 10.0, step=0.01)*1e-12
        noise_R = st.number_input("Equivalent resistor for noise (kΩ)", 0.1, 10000.0, 10.0)*1e3
        sim_time = st.number_input("Sim time (ms)", 1.0, 2000.0, 200.0)/1000.0

    k = (2*np.pi*fn)**2 * m
    c = (2*np.pi*fn*m) / Q

    wave = st.selectbox("Excitation", ["Step: 1g", "Sine: 0.5g @ 50Hz", "Random: 0.2g RMS"])
    fs = 5000
    t = np.linspace(0, sim_time, int(fs*sim_time)+1)

    gacc = 9.80665
    if wave == "Step: 1g":
        a = np.ones_like(t)*gacc
    elif wave == "Sine: 0.5g @ 50Hz":
        a = 0.5*gacc*np.sin(2*np.pi*50*t)
    else:
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 0.2*gacc, size=t.shape)

    # m x'' + c x' + k x = m a(t)
    def f(_t, y):
        x, v = y
        anow = np.interp(_t, t, a)
        dv = (m*anow - c*v - k*x)/m
        return np.array([v, dv])
    y = rk4(f, np.array([0.0, 0.0]), t)
    x = y[:,0]

    # Differential capacitance (parallel-plate, half-bridge)
    g0_acc = g0_um_acc*1e-6
    Aacc = A_um2_acc*1e-12
    C1 = EPS0*Aacc/(g0_acc - x)
    C2 = EPS0*Aacc/(g0_acc + x)
    dC = C1 - C2  # differential
    # Ideal charge amp: Vout ≈ (Vbias/Cf) * dC
    vout_ideal = (Vbias/Cf) * dC

    # Add simple noise model (Johnson + kT/C white)
    kB = 1.380649e-23
    T = 300.0
    en_R = math.sqrt(4*kB*T*noise_R)      # V/√Hz
    en_kTC = math.sqrt(kB*T/Cf)           # V (per sample, crude)
    bw = fs/2
    vrms_R = en_R*math.sqrt(bw)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, vrms_R, size=vout_ideal.shape) + rng.normal(0.0, en_kTC, size=vout_ideal.shape)
    vout = vout_ideal + noise

    # SNR estimate
    sig_rms = np.sqrt(np.mean((vout_ideal - np.mean(vout_ideal))**2))
    noise_rms = np.sqrt(np.mean((vout - vout_ideal)**2))
    snr_db = 20*np.log10(max(sig_rms,1e-18)/max(noise_rms,1e-18))
    st.metric("Estimated SNR (dB)", f"{snr_db:.1f}")

    cA, cB = st.columns(2)
    with cA:
        figx = nice_plot(t*1e3, [x*1e6], ["x"], "Time (ms)", "Displacement (µm)")
        st.plotly_chart(figx, use_container_width=True)
        figc = nice_plot(t*1e3, [dC*1e15], ["ΔC"], "Time (ms)", "ΔC (fF)")
        st.plotly_chart(figc, use_container_width=True)
    with cB:
        figv = nice_plot(t*1e3, [vout, vout_ideal], ["Vout (noisy)", "Vout (ideal)"], "Time (ms)", "Voltage (V)")
        st.plotly_chart(figv, use_container_width=True)

    st.session_state["snapshot"].update({
        "accelerometer": dict(m_kg=m, k_Npm=k, c_Ns=m, fn=fn, Q=Q, g0_um=g0_um_acc,
                              A_um2=A_um2_acc, Vbias=Vbias, Cf_F=Cf, SNR_dB=float(snr_db))
    })

# ---------- 4) Thermal Actuator ----------
with tabs[3]:
    st.header("Thermal Actuator (RC Thermal Model)")
    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input("Beam length L (µm)", 10.0, 10000.0, 500.0, step=10.0)*1e-6
        w = st.number_input("Beam width w (µm)", 1.0, 200.0, 10.0, step=0.5)*1e-6
        t_th = st.number_input("Thickness t (µm)", 1.0, 100.0, 10.0, step=0.5)*1e-6
        k_th = st.number_input("Thermal conductivity k (W/mK)", 1.0, 200.0, 130.0, step=1.0)
    with col2:
        rho = st.number_input("Density ρ (kg/m³)", 1000.0, 9000.0, 2329.0, step=1.0)
        cp = st.number_input("Specific heat c (J/kgK)", 100.0, 2000.0, 700.0, step=1.0)
        alpha = st.number_input("CTE α (ppm/K)", 0.1, 30.0, 2.6, step=0.1)*1e-6
        I = st.number_input("Drive current I (mA)", 0.0, 100.0, 20.0, step=0.5)/1000.0
        R_el = st.number_input("Electrical resistance R (Ω)", 1.0, 10000.0, 500.0, step=1.0)

    A_cs = w*t_th
    V_beam = A_cs*L
    Rth = L/(k_th*A_cs + 1e-30)  # avoid zero
    Cth = rho*cp*V_beam
    tau = Rth*Cth
    P = I**2 * R_el
    dT_inf = P*Rth  # steady-state ΔT
    # Tip expansion ~ α * ΔT * L (single beam crude)
    tip = alpha * dT_inf * L

    st.metric("Thermal time constant τ (ms)", f"{tau*1e3:.2f}")
    st.metric("Steady ΔT (K)", f"{dT_inf:.2f}")
    st.metric("Tip thermal expansion (nm)", f"{tip*1e9:.2f}")

    tth = np.linspace(0, 5*max(tau,1e-6), 600)
    dT_t = dT_inf*(1 - np.exp(-tth/ max(tau,1e-12)))
    tip_t = alpha*L*dT_t
    figT = nice_plot(tth*1e3, [dT_t], ["ΔT"], "Time (ms)", "Temperature rise (K)")
    figTip = nice_plot(tth*1e3, [tip_t*1e9], ["Tip disp"], "Time (ms)", "Tip displacement (nm)")
    st.plotly_chart(figT, use_container_width=True)
    st.plotly_chart(figTip, use_container_width=True)

    st.session_state["snapshot"].update({
        "thermal": dict(Rth=Rth, Cth=Cth, tau_s=tau, dT_inf=dT_inf, tip_m=tip)
    })

# ---------- 5) Thin Plate Bending ----------
with tabs[4]:
    st.header("Thin Plate Bending (clamped, uniform load)")
    st.caption("Central deflection w₀ ≈ (q a⁴)/(64 D) for square plate, D = E t³/(12(1-ν²)).")
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("Plate side a (mm)", 0.1, 100.0, 5.0, step=0.1)*1e-3
        tplt = st.number_input("Thickness t (µm)", 1.0, 200.0, 10.0, step=0.5)*1e-6
        E = st.number_input("Young’s modulus E (GPa)", 1.0, 300.0, 160.0, step=1.0)*1e9
        nu = st.number_input("Poisson’s ratio ν", 0.0, 0.49, 0.22, step=0.01)
    with c2:
        q = st.number_input("Uniform load q (kPa)", 0.0, 1000.0, 10.0, step=1.0)*1e3
    D = E*tplt**3/(12*(1-nu**2) + 1e-30)
    w0 = (q*a**4)/(64*D + 1e-30)
    st.metric("Central deflection w₀ (nm)", f"{w0*1e9:.2f}")

    x = np.linspace(-a/2, a/2, 120)
    y = np.linspace(-a/2, a/2, 120)
    X, Y = np.meshgrid(x, y)
    # simple shape function for clamped square (illustrative)
    W = w0*(1 - (2*X/a)**2)**2 * (1 - (2*Y/a)**2)**2
    fig = go.Figure(data=[go.Surface(x=X*1e3, y=Y*1e3, z=W*1e6)])
    fig.update_layout(scene=dict(
        xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="Deflection (µm)"
    ), margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["snapshot"].update({
        "plate": dict(a_m=a, t_m=tplt, E_Pa=E, nu=nu, q_Pa=q, w0_m=w0)
    })

# ---------- 6) Optical MEMS Attenuator ----------
with tabs[5]:
    st.header("Optical MEMS Variable Attenuator (toy model)")
    st.caption("Gaussian coupling with angular misalignment: η ≈ exp(-(θ/θ₀)²) · exp(-(Δ/Δ₀)²)")
    col1, col2 = st.columns(2)
    with col1:
        theta_mdeg = st.slider("Tilt θ (millidegrees)", 0.0, 50.0, 5.0, 0.1)
        theta0_mdeg = st.number_input("θ₀ (millidegrees)", 1.0, 200.0, 15.0, step=1.0)
        offset_um = st.slider("Lateral offset Δ (µm)", 0.0, 20.0, 2.0, 0.1)
        delta0_um = st.number_input("Δ₀ (µm)", 1.0, 50.0, 8.0, step=0.5)
    with col2:
        Pin_dBm = st.number_input("Input power (dBm)", -40.0, 20.0, 0.0, step=0.1)
        loss_other_dB = st.number_input("Other fixed losses (dB)", 0.0, 10.0, 1.0, step=0.1)

    eta = np.exp(-(theta_mdeg/theta0_mdeg)**2) * np.exp(-(offset_um/delta0_um)**2)
    att_dB = -10*np.log10(max(eta,1e-15))
    Pout_dBm = Pin_dBm - att_dB - loss_other_dB

    st.metric("Attenuation (dB)", f"{att_dB:.2f}")
    st.metric("Output power (dBm)", f"{Pout_dBm:.2f}")

    th = np.linspace(0, 50, 500)
    att_scan = -10*np.log10(np.exp(-(th/theta0_mdeg)**2) * np.exp(-(offset_um/delta0_um)**2))
    fig = nice_plot(th, [att_scan], ["Attenuation"], "Tilt θ (mdeg)", "Attenuation (dB)")
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["snapshot"].update({
        "optical_va": dict(theta_mdeg=theta_mdeg, theta0_mdeg=theta0_mdeg,
                           offset_um=offset_um, delta0_um=delta0_um,
                           Pin_dBm=Pin_dBm, Pout_dBm=Pout_dBm)
    })

# ---------- 7) Process Planner ----------
with tabs[6]:
    st.header("Process Planner — Bulk / Surface / LIGA")
    flow = st.selectbox("Select process", ["Bulk micromachining", "Surface micromachining", "LIGA"])
    goal = st.text_input("Target device (free text)", "Capacitive accelerometer")
    steps = []
    if flow == "Bulk micromachining":
        steps = [
            "Start: Si (100) wafer, thermal SiO₂ mask",
            "Photolithography: pattern etch windows",
            "Anisotropic wet etch (KOH/TMAH) to form cavities",
            "DRIE to define structures / release windows",
            "Oxide removal, metallization, wafer bonding—final release"
        ]
    elif flow == "Surface micromachining":
        steps = [
            "Start: Si wafer with thermal oxide",
            "Deposit sacrificial oxide (e.g., PSG) + pattern",
            "Deposit structural polysilicon + pattern",
            "Repeat sac/structural stacks as needed",
            "Release etch (HF vapor), critical point drying"
        ]
    else:
        steps = [
            "Seed layer on substrate",
            "Photoresist (thick) exposure with X-rays",
            "Develop high-aspect molds",
            "Electroform metal to fill molds",
            "Strip resist, finish & release"
        ]
    st.subheader("Suggested high-level flow")
    for i, sstep in enumerate(steps, 1):
        st.write(f"**{i}.** {sstep}")

    report = f"""MEMS Process Plan
Target: {goal}
Chosen flow: {flow}

Steps:
""" + "\n".join([f"{i}. {s}" for i, s in enumerate(steps, 1)])
    download_text_button("Download plan (.txt)", report, f"{flow.replace(' ','_').lower()}_plan.txt")

    st.session_state["snapshot"].update({
        "process_plan": dict(flow=flow, steps=steps, goal=goal)
    })

st.success("Tip: Use the sidebar to save & export design snapshots as JSON for your report.")
