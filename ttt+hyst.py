import numpy as np
import matplotlib.pyplot as plt

class SimulationConfig: 
    def __init__(self, ho_strategy='SNR', hysteresis_dB=3.0, ttt_seconds=0.3):
        self.calculate_interference = False 
        self.moving_user = False  
        
        # Dinamičko podešavanje strategije i parametara kroz petlju
        self.ho_strategy = ho_strategy
        self.H_dB = hysteresis_dB            
        self.ttt_seconds = ttt_seconds 
        
        self.use_hysteresis = True if hysteresis_dB > 0 else False
        self.alpha_off = 0.5      
        self.gamma_th_dB = 3.0     # Outage prag [dB]

        # Geometrija i orbita
        self.N = 35                 
        self.elev_min_deg = 10.0    
        self.Re = 6371e3            
        self.h = 900e3              
        self.r_orbit = self.Re + self.h
        self.konstanta_mi = 3.986e14

        # Vreme simulacije - dt postavljen na 0.1s radi preciznijeg hvatanja TTT koraka
        self.dt = 0.1          
        self.ttt_steps = int(self.ttt_seconds / self.dt)  
        
        # Link Budget
        self.fc = 2e9               
        self.c = 3e8                
        self.lambda_ = self.c / self.fc
        self.Pt_dBm = 35
        self.Gt_dBi = 30.0          
        self.Gr_max_dBi = 0.0      
        self.La = 0.5
        
        self.B = 10e6                
        self.N0_dBmHz = -174.0
        self.NF_dB = 4.0

        self.noise_dBm = self.N0_dBmHz + 10 * np.log10(self.B) + self.NF_dB
        self.noise_linear = 10 ** (self.noise_dBm / 10.0)  

    @property
    def omega(self):
        return np.sqrt(self.konstanta_mi / (self.r_orbit ** 3))

    @property
    def T_orbite(self): 
        return 2 * np.pi / self.omega


class User:
    def __init__(self, angle_rad=0.0, radius=6371e3):
        self.radius = radius 
        self.v_ms = 100 / 3.6   
        self.omega_user = self.v_ms / self.radius 
        self.angle = angle_rad
        self.x = radius * np.cos(angle_rad)
        self.y = radius * np.sin(angle_rad)
        self.ux = self.x / radius
        self.uy = self.y / radius
        self.position_history = {}


class Satellite:
    def __init__(self, sat_id, initial_theta):
        self.id = sat_id
        self.initial_theta = initial_theta
        self.metrics_history = {}     
        self.positions_history = {}   
        self.prev_sf_LoS = None 
        self.prev_sf_NLoS = None

    def get_3gpp_rural_parameters(self, elev_deg):
        if elev_deg <= 15.0:
            return 0.782, 1.14, 9.7, 18.2
        elif elev_deg <= 25.0:
            return 0.861, 0.92, 8.8, 15.9
        else:
            return 0.950, 0.72, 7.5, 0.0

    def update_position_and_metrics(self, t, config, user): 
        theta_orbit = self.initial_theta + config.omega * t
        xs = config.r_orbit * np.cos(theta_orbit)
        ys = config.r_orbit * np.sin(theta_orbit)
        
        self.positions_history[t] = (xs, ys)
        
        dx = xs - user.x
        dy = ys - user.y
        d = np.sqrt(dx**2 + dy**2)
        
        elev_rad = np.arcsin((dx * user.ux + dy * user.uy) / d)
        elev_deg = np.degrees(elev_rad)
        
        if elev_deg >= config.elev_min_deg:
            visible = True
            fspl_dB = 20 * np.log10(4 * np.pi * d * config.fc/config.c) 
            pr_los, sigma_los, sigma_nlos, cl_dB = self.get_3gpp_rural_parameters(elev_deg)
            
            novi_sum_los = np.random.normal(0, sigma_los)
            novi_sum_nlos = np.random.normal(0, sigma_nlos)
            
            rho = 0.98
            if self.prev_sf_LoS is None or self.prev_sf_NLoS is None:
                sf_los = novi_sum_los
                sf_nlos = novi_sum_nlos
            else:
                sf_los = rho * self.prev_sf_LoS + np.sqrt(1 - rho**2) * novi_sum_los
                sf_nlos = rho * self.prev_sf_NLoS + np.sqrt(1 - rho**2) * novi_sum_nlos
                
            self.prev_sf_LoS = sf_los
            self.prev_sf_NLoS = sf_nlos
            
            PL_LoS = sf_los + fspl_dB
            PL_NLoS = sf_nlos + cl_dB + fspl_dB
            PL_tot = pr_los * PL_LoS + (1.0 - pr_los) * PL_NLoS
            pr_dBm = config.Pt_dBm + config.Gt_dBi + config.Gr_max_dBi - PL_tot - config.La
        else:
            visible = False
            pr_dBm = -np.inf
            self.prev_sf_LoS = None
            self.prev_sf_NLoS = None
            
        self.metrics_history[t] = {
            'sinr_dB': -np.inf,
            'capacity': 0.0,
            'visible': visible,
            'pr_dBm': pr_dBm,
            'distance': d if visible else np.inf, 
            'vector_to_sat': (dx, dy) if visible else (0.0, 0.0), 
            'elevation_deg': elev_deg if visible else 0.0  
        }

    def calculate_sinr_and_interference(self, t, config):
        if not self.metrics_history[t]['visible']:
            return
        pr_linear = 10 ** (self.metrics_history[t]['pr_dBm'] / 10.0)
        sinr_linearno = pr_linear / config.noise_linear
        self.metrics_history[t]['sinr_dB'] = 10 * np.log10(sinr_linearno)


# OPTIMIZOVANA LOGIKA ZA DONOŠENJE ODLUKE BEZ RIZIKA OD ZAGUŠENJA MEMORIJE
def handover_decision(t, idx, current_serving_id, satellites, config, ttt_state):    
    current_sat = next((s for s in satellites if s.id == current_serving_id), None)
    
    # Ako je početak simulacije, outage, ili je satelit zašao - resetuj lokalno TTT stanje
    if idx == 0 or current_serving_id == 0 or current_sat is None or not current_sat.metrics_history[t]['visible']:
        ttt_state['target_id'] = 0
        ttt_state['timer'] = 0.0
        
        visible_sats = [s for s in satellites if s.metrics_history[t]['visible'] and s.metrics_history[t]['sinr_dB'] != -np.inf] 
        if not visible_sats:
            return 0 
        best_sat = max(visible_sats, key=lambda s: s.metrics_history[t]['sinr_dB']) 
        return best_sat.id

    current_sinr = current_sat.metrics_history[t]['sinr_dB']

    # --- STRATEGIJA 1: ČISTI SNR (SA ILI BEZ HISTEREZISA) ---
    if config.ho_strategy == 'SNR':
        best_sat_id = current_serving_id
        threshold = (current_sinr + config.H_dB) if config.use_hysteresis else current_sinr
        
        for sat in satellites:
            if sat.id == current_serving_id or not sat.metrics_history[t]['visible']:
                continue
            if sat.metrics_history[t]['sinr_dB'] > threshold:  
                threshold = sat.metrics_history[t]['sinr_dB']
                best_sat_id = sat.id
        return best_sat_id
              
    # --- STRATEGIJA 2: TTT + SNR ---
    elif config.ho_strategy == 'TTT+SNR':
        best_alt_id = current_serving_id
        threshold = (current_sinr + config.H_dB) if config.use_hysteresis else current_sinr
        
        for sat in satellites:
            if sat.id == current_serving_id or not sat.metrics_history[t]['visible']:
                continue
            if sat.metrics_history[t]['sinr_dB'] > threshold:
                threshold = sat.metrics_history[t]['sinr_dB']
                best_alt_id = sat.id

        # Ako je pronađen stabilniji i bolji alternativni satelit van margine
        if best_alt_id != current_serving_id:
            if best_alt_id == ttt_state['target_id']:
                ttt_state['timer'] += config.dt
            else:
                ttt_state['target_id'] = best_alt_id
                ttt_state['timer'] = config.dt
                
            if ttt_state['timer'] >= config.ttt_seconds:
                activated_id = ttt_state['target_id']
                ttt_state['target_id'] = 0
                ttt_state['timer'] = 0.0
                return activated_id
        else:
            ttt_state['target_id'] = 0
            ttt_state['timer'] = 0.0

    return current_serving_id 


def run_core_simulation(ho_strategy, hysteresis_value, ttt_value):
    # Lokalni rečnik za praćenje TTT stanja koji se izoluje unutar svakog pokretanja
    ttt_state = {'target_id': 0, 'timer': 0.0}

    cfg = SimulationConfig(ho_strategy=ho_strategy, hysteresis_dB=hysteresis_value, ttt_seconds=ttt_value)
    user = User(angle_rad=0.0, radius=cfg.Re)
    
    satellites = [] 
    for i in range(1, cfg.N + 1):
        theta0 = 2 * np.pi * (i - 1) / cfg.N
        satellites.append(Satellite(sat_id=i, initial_theta=theta0))
        
    time_steps = []
    current_time = 0.0 
    while current_time <= cfg.T_orbite * 0.5:
        time_steps.append(current_time)
        current_time += cfg.dt

    for t in time_steps:
        user.position_history[t] = (user.x, user.y)
        for sat in satellites:
            sat.update_position_and_metrics(t, cfg, user) 
            sat.calculate_sinr_and_interference(t, cfg)

    serving_history = {}  
    current_serving_id = 0
    
    for idx, t in enumerate(time_steps):
        current_serving_id = handover_decision(t, idx, current_serving_id, satellites, cfg, ttt_state)
        if current_serving_id == 0:
            serving_history[t] = {'sat_id': 0, 'sinr_dB': -np.inf}
        else:
            active_sat = next(s for s in satellites if s.id == current_serving_id) 
            serving_history[t] = {
                'sat_id': current_serving_id,
                'sinr_dB': active_sat.metrics_history[t]['sinr_dB']
            }

    handover_count = 0
    outage_points = 0
    
    for idx, t in enumerate(time_steps):
        if idx > 0:
            prev_t = time_steps[idx-1]
            if serving_history[t]['sat_id'] != serving_history[prev_t]['sat_id']:
                if serving_history[t]['sat_id'] != 0 and serving_history[prev_t]['sat_id'] != 0:
                    handover_count += 1
        if serving_history[t]['sinr_dB'] < cfg.gamma_th_dB:
            outage_points += 1

    outage_probability = (outage_points / len(time_steps)) * 100
    return handover_count, outage_probability


def generate_comparative_plots():
    # Opseg histerezisa na X osi (od 0 do 5 dB sa korakom 0.5)
    hysteresis_range = np.arange(0.0, 5.5, 0.5)
    
    results_snr_ho = []
    results_snr_outage = []
    
    results_ttt02_ho = []
    results_ttt02_outage = []
    
    results_ttt03_ho = []
    results_ttt03_outage = []
    
    results_ttt05_ho = []
    results_ttt05_outage = []

    print("Započinjem proračun svih scenarija...")

    for h in hysteresis_range:
        print(f"Izvršavam proračune za marginu H = {h:.1f} dB...")
        
        # 1. Čista SNR metoda (Označena kao 'Measurement-based' u analizi, TTT=0)
        ho_snr, out_snr = run_core_simulation('SNR', h, ttt_value=0.0)
        results_snr_ho.append(ho_snr)
        results_snr_outage.append(out_snr)
        
        # 2. TTT + SNR sa TTT = 0.2 s
        ho_02, out_02 = run_core_simulation('TTT+SNR', h, ttt_value=0.2)
        results_ttt02_ho.append(ho_02)
        results_ttt02_outage.append(out_02)
        
        # 3. TTT + SNR sa TTT = 0.3 s
        ho_03, out_03 = run_core_simulation('TTT+SNR', h, ttt_value=0.3)
        results_ttt03_ho.append(ho_03)
        results_ttt03_outage.append(out_03)
        
        # 4. TTT + SNR sa TTT = 0.5 s
        ho_05, out_05 = run_core_simulation('TTT+SNR', h, ttt_value=0.5)
        results_ttt05_ho.append(ho_05)
        results_ttt05_outage.append(out_05)

    print("\nSimulacije uspešno završene. Generišem uporedne grafikone...")

    # ==================== GRAFIK 1: BROJ HENDOVERA ====================
    plt.figure(figsize=(10, 6))
    plt.plot(hysteresis_range, results_snr_ho, 'o-', color='black', label='Measurement-based (Čist SNR)', linewidth=2)
    plt.plot(hysteresis_range, results_ttt02_ho, 's--', color='teal', label='TTT + SNR (TTT = 0.2s)')
    plt.plot(hysteresis_range, results_ttt03_ho, '^--', color='blue', label='TTT + SNR (TTT = 0.3s)')
    plt.plot(hysteresis_range, results_ttt05_ho, 'd--', color='crimson', label='TTT + SNR (TTT = 0.5s)')
    
    plt.xlabel('Margina Histerezisa H [dB]', fontsize=11)
    plt.ylabel('Ukupan broj izvršenih hendovera', fontsize=11)
    plt.title('Uporedni prikaz: Uticaj Histerezisa i TTT parametra na broj hendovera', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

    # ==================== GRAFIK 2: VERIVATNOĆA OTKAZA ====================
    plt.figure(figsize=(10, 6))
    plt.plot(hysteresis_range, results_snr_outage, 'o-', color='black', label='Measurement-based (Čist SNR)', linewidth=2)
    plt.plot(hysteresis_range, results_ttt02_outage, 's--', color='teal', label='TTT + SNR (TTT = 0.2s)')
    plt.plot(hysteresis_range, results_ttt03_outage, '^--', color='blue', label='TTT + SNR (TTT = 0.3s)')
    plt.plot(hysteresis_range, results_ttt05_outage, 'd--', color='crimson', label='TTT + SNR (TTT = 0.5s)')
    
    plt.xlabel('Margina Histerezisa H [dB]', fontsize=11)
    plt.ylabel('Verovatnoća otkaza sistema (Outage Probability) [%]', fontsize=11)
    plt.title('Uporedni prikaz: Uticaj Histerezisa i TTT parametra na verovatnoću otkaza', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.show()

if __name__ == '__main__':
    generate_comparative_plots()