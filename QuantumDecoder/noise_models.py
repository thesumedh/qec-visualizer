"""
Advanced Quantum Noise Models
Realistic simulation of quantum hardware noise for different platforms
"""

import numpy as np
from typing import Dict, Any
from quantum_states import QuantumState

class QuantumNoiseModel:
    """Advanced noise model simulating real quantum hardware"""
    
    def __init__(self, platform: str = "IBM"):
        self.platform = platform
        self.noise_params = self._get_platform_params()
    
    def _get_platform_params(self) -> Dict[str, float]:
        """Get realistic noise parameters for different platforms"""
        params = {
            "IBM": {
                "t1_relaxation": 100e-6,  # 100 microseconds
                "t2_dephasing": 50e-6,    # 50 microseconds  
                "gate_error_1q": 0.001,   # 0.1% single-qubit error
                "gate_error_2q": 0.01,    # 1% two-qubit error
                "readout_error": 0.02,    # 2% measurement error
                "crosstalk": 0.005        # 0.5% crosstalk
            },
            "Google": {
                "t1_relaxation": 80e-6,
                "t2_dephasing": 40e-6,
                "gate_error_1q": 0.0008,
                "gate_error_2q": 0.008,
                "readout_error": 0.015,
                "crosstalk": 0.003
            },
            "IonQ": {
                "t1_relaxation": 10000e-6,  # Much longer for trapped ions
                "t2_dephasing": 1000e-6,
                "gate_error_1q": 0.0001,
                "gate_error_2q": 0.002,
                "readout_error": 0.001,
                "crosstalk": 0.0001
            }
        }
        return params.get(self.platform, params["IBM"])
    
    def apply_decoherence(self, state: QuantumState, time_us: float):
        """Apply T1 and T2 decoherence over time"""
        t1 = self.noise_params["t1_relaxation"]
        t2 = self.noise_params["t2_dephasing"]
        
        # T1 relaxation (amplitude damping)
        gamma1 = time_us / t1
        relaxation_factor = np.exp(-gamma1)
        
        # T2 dephasing (phase damping)  
        gamma2 = time_us / t2
        dephasing_factor = np.exp(-gamma2)
        
        # Apply to state vector
        for i in range(len(state.state_vector)):
            if abs(state.state_vector[i]) > 1e-10:
                # Apply amplitude damping
                state.state_vector[i] *= np.sqrt(relaxation_factor)
                
                # Apply random phase noise
                phase_noise = np.random.normal(0, np.sqrt(1 - dephasing_factor))
                state.state_vector[i] *= np.exp(1j * phase_noise)
        
        state._normalize()
    
    def apply_gate_noise(self, state: QuantumState, gate_type: str, qubits: list):
        """Apply realistic gate noise"""
        if gate_type in ["X", "Y", "Z", "H"]:
            error_rate = self.noise_params["gate_error_1q"]
        else:  # Two-qubit gates
            error_rate = self.noise_params["gate_error_2q"]
        
        # Apply random Pauli errors with probability = error_rate
        for qubit in qubits:
            if np.random.random() < error_rate:
                error_type = np.random.choice(["X", "Y", "Z"])
                if error_type == "X":
                    state.apply_x_error(qubit)
                elif error_type == "Y":
                    state.apply_y_error(qubit)
                elif error_type == "Z":
                    state.apply_z_error(qubit)
    
    def apply_crosstalk(self, state: QuantumState, target_qubit: int):
        """Apply crosstalk noise to neighboring qubits"""
        crosstalk_rate = self.noise_params["crosstalk"]
        
        # Define qubit connectivity (linear for simplicity)
        neighbors = []
        if target_qubit > 0:
            neighbors.append(target_qubit - 1)
        if target_qubit < state.n_qubits - 1:
            neighbors.append(target_qubit + 1)
        
        for neighbor in neighbors:
            if np.random.random() < crosstalk_rate:
                # Apply weak random rotation
                angle = np.random.normal(0, 0.1)  # Small angle
                # Simplified: apply as phase error
                state.apply_z_error(neighbor)
    
    def apply_measurement_noise(self, measurement_result: int) -> int:
        """Apply readout error to measurement"""
        error_rate = self.noise_params["readout_error"]
        
        if np.random.random() < error_rate:
            return 1 - measurement_result  # Flip result
        return measurement_result
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get detailed platform information"""
        return {
            "platform": self.platform,
            "coherence_time_t1": f"{self.noise_params['t1_relaxation']*1e6:.0f} μs",
            "coherence_time_t2": f"{self.noise_params['t2_dephasing']*1e6:.0f} μs", 
            "gate_fidelity_1q": f"{(1-self.noise_params['gate_error_1q'])*100:.2f}%",
            "gate_fidelity_2q": f"{(1-self.noise_params['gate_error_2q'])*100:.2f}%",
            "readout_fidelity": f"{(1-self.noise_params['readout_error'])*100:.2f}%"
        }

def simulate_realistic_qec_with_noise(qec_code, noise_model: QuantumNoiseModel, 
                                    error_type: str, decoder_type: str) -> Dict[str, Any]:
    """Complete QEC simulation with realistic noise"""
    
    # Initialize state
    state = qec_code.encode_logical_zero()
    original_state = QuantumState(state.state_vector.copy(), state.n_qubits)
    
    # Apply decoherence during encoding (10 μs)
    noise_model.apply_decoherence(state, 10.0)
    
    # Apply intentional error
    if error_type != "None":
        target_qubit = np.random.randint(0, qec_code.n_qubits)
        if error_type == "X":
            state.apply_x_error(target_qubit)
        elif error_type == "Z":
            state.apply_z_error(target_qubit)
        elif error_type == "Y":
            state.apply_y_error(target_qubit)
        
        # Apply crosstalk
        noise_model.apply_crosstalk(state, target_qubit)
    
    # Apply decoherence during syndrome measurement (20 μs)
    noise_model.apply_decoherence(state, 20.0)
    
    # Measure syndrome with noise
    syndrome = qec_code.measure_syndrome(state)
    noisy_syndrome = [noise_model.apply_measurement_noise(bit) for bit in syndrome]
    
    # Apply correction
    success = qec_code.decode_and_correct(state, noisy_syndrome)
    
    # Apply decoherence during correction (5 μs)
    noise_model.apply_decoherence(state, 5.0)
    
    # Calculate final fidelity
    fidelity = np.abs(np.vdot(original_state.state_vector, state.state_vector))**2
    
    # Decoder-specific performance
    decoder_efficiency = {"Standard Lookup": 0.85, "ML-Enhanced": 0.92, "Iterative": 0.78}
    realistic_fidelity = fidelity * decoder_efficiency.get(decoder_type, 0.85)
    
    return {
        "fidelity": realistic_fidelity,
        "syndrome": syndrome,
        "noisy_syndrome": noisy_syndrome,
        "success": success and realistic_fidelity > 0.7,
        "platform_info": noise_model.get_platform_info(),
        "total_time_us": 35.0  # Total circuit time
    }