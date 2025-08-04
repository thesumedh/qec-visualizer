import numpy as np
from typing import List, Optional, Tuple

class QuantumState:
    """Represents a quantum state with operations for error correction."""
    
    def __init__(self, state_vector: np.ndarray, n_qubits: int):
        """
        Initialize quantum state.
        
        Args:
            state_vector: Complex amplitude vector
            n_qubits: Number of qubits
        """
        self.state_vector = state_vector.copy()
        self.n_qubits = n_qubits
        self._normalize()
    
    def _normalize(self):
        """Normalize the state vector."""
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-10:
            self.state_vector /= norm
        else:
            # Handle zero norm case - reset to |0⟩ state
            self.state_vector = np.zeros(2**self.n_qubits, dtype=complex)
            self.state_vector[0] = 1.0
    
    def apply_x_error(self, qubit_index: int):
        """Apply X (bit flip) error to specified qubit."""
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        # Create new state vector with X gate applied
        new_state = np.zeros_like(self.state_vector)
        
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                # Flip the bit at qubit_index position
                flipped_state = i ^ (1 << (self.n_qubits - 1 - qubit_index))
                new_state[flipped_state] = amplitude
        
        self.state_vector = new_state
    
    def apply_z_error(self, qubit_index: int):
        """Apply Z (phase flip) error to specified qubit."""
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        # Apply phase flip: multiply amplitude by -1 if qubit is |1⟩
        for i in range(len(self.state_vector)):
            # Check if the specified qubit is in |1⟩ state
            if (i >> (self.n_qubits - 1 - qubit_index)) & 1:
                self.state_vector[i] *= -1
    
    def apply_y_error(self, qubit_index: int):
        """Apply Y error (combination of X and Z) to specified qubit."""
        # Y = iXZ, properly implemented
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        # Y gate implementation: Y = i * X * Z
        new_state = np.zeros_like(self.state_vector)
        
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                # Apply X error (bit flip)
                flipped_state = i ^ (1 << (self.n_qubits - 1 - qubit_index))
                
                # Apply Z error (phase flip) - check if qubit was originally |1⟩
                phase_factor = -1 if (i >> (self.n_qubits - 1 - qubit_index)) & 1 else 1
                
                # Apply Y = iXZ
                new_state[flipped_state] = amplitude * phase_factor * 1j
        
        self.state_vector = new_state
    
    def measure_qubit(self, qubit_index: int) -> Tuple[int, 'QuantumState']:
        """
        Measure a single qubit and return result and post-measurement state.
        
        Returns:
            Tuple of (measurement_result, collapsed_state)
        """
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        # Calculate probabilities for |0⟩ and |1⟩
        prob_0 = 0
        prob_1 = 0
        
        for i, amplitude in enumerate(self.state_vector):
            prob = abs(amplitude)**2
            if (i >> (self.n_qubits - 1 - qubit_index)) & 1:
                prob_1 += prob
            else:
                prob_0 += prob
        
        # Simulate measurement
        measurement_result = np.random.choice([0, 1], p=[prob_0, prob_1])
        
        # Create collapsed state
        new_state = np.zeros_like(self.state_vector)
        norm_factor = 0
        
        for i, amplitude in enumerate(self.state_vector):
            qubit_value = (i >> (self.n_qubits - 1 - qubit_index)) & 1
            if qubit_value == measurement_result:
                new_state[i] = amplitude
                norm_factor += abs(amplitude)**2
        
        # Normalize collapsed state with better error handling
        if norm_factor > 1e-10:
            new_state /= np.sqrt(norm_factor)
        else:
            # Handle edge case where norm is too small
            new_state = np.zeros_like(self.state_vector)
            if len(new_state) > 0:
                new_state[0] = 1.0
        
        return measurement_result, QuantumState(new_state, self.n_qubits)
    
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution over computational basis states."""
        return np.abs(self.state_vector)**2
    
    def get_computational_basis_labels(self) -> List[str]:
        """Get labels for computational basis states."""
        return [format(i, f'0{self.n_qubits}b') for i in range(2**self.n_qubits)]
    
    def calculate_fidelity_with_logical_zero(self) -> float:
        """Calculate fidelity with the logical zero state."""
        # For simplicity, assume logical zero is |000...0⟩
        if len(self.state_vector) > 0:
            logical_zero_prob = abs(self.state_vector[0])**2
            return logical_zero_prob
        return 0.0
    
    def get_bloch_sphere_coordinates(self, qubit_index: int) -> Tuple[float, float, float]:
        """
        Get Bloch sphere coordinates for a single qubit (if separable).
        
        Returns:
            Tuple of (x, y, z) coordinates on Bloch sphere
        """
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        # Check for entanglement - simplified approach
        if self.n_qubits > 1:
            # For entangled states, return approximate coordinates
            pass
        
        # Calculate reduced density matrix for single qubit
        # This is a simplified calculation assuming the state can be factored
        
        # Calculate expectation values of Pauli operators
        sigma_x = 0
        sigma_y = 0
        sigma_z = 0
        
        for i, amplitude in enumerate(self.state_vector):
            prob = abs(amplitude)**2
            qubit_value = (i >> (self.n_qubits - 1 - qubit_index)) & 1
            
            if qubit_value == 0:
                sigma_z += prob
            else:
                sigma_z -= prob
        
        # For simplicity, set x and y components based on phase relationships
        # This is an approximation for visualization purposes
        sigma_x = 0  # Would need more complex calculation for exact value
        sigma_y = 0  # Would need more complex calculation for exact value
        
        return sigma_x, sigma_y, sigma_z
    
    def copy(self) -> 'QuantumState':
        """Create a copy of the quantum state."""
        return QuantumState(self.state_vector.copy(), self.n_qubits)
    
    def __str__(self) -> str:
        """String representation of the quantum state."""
        result = "Quantum State:\n"
        labels = self.get_computational_basis_labels()
        probs = self.get_probabilities()
        
        for i, (label, prob) in enumerate(zip(labels, probs)):
            if prob > 1e-6:  # Only show significant amplitudes
                amplitude = self.state_vector[i]
                result += f"|{label}⟩: {amplitude:.4f} (prob: {prob:.4f})\n"
        
        return result
