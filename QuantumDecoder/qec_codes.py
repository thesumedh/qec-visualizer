import numpy as np
from typing import List, Tuple, Optional
from quantum_states import QuantumState

class QECCode:
    """Base class for quantum error correction codes."""
    
    def __init__(self, n_qubits: int, k_qubits: int, distance: int):
        self.n_qubits = n_qubits  # Number of physical qubits
        self.k_qubits = k_qubits  # Number of logical qubits
        self.distance = distance   # Code distance
    
    def encode_logical_zero(self) -> QuantumState:
        """Encode the logical |0⟩ state."""
        raise NotImplementedError
    
    def encode_logical_one(self) -> QuantumState:
        """Encode the logical |1⟩ state."""
        raise NotImplementedError
    
    def measure_syndrome(self, state: QuantumState) -> List[int]:
        """Measure the error syndrome."""
        raise NotImplementedError
    
    def decode_and_correct(self, state: QuantumState, syndrome: List[int]) -> bool:
        """Apply error correction based on syndrome."""
        raise NotImplementedError

class ThreeQubitBitFlipCode(QECCode):
    """3-qubit bit flip quantum error correction code."""
    
    def __init__(self):
        super().__init__(n_qubits=3, k_qubits=1, distance=3)
        
        # Stabilizer generators for 3-qubit code
        # S1 = Z1⊗Z2⊗I, S2 = I⊗Z2⊗Z3
        self.stabilizers = [
            [1, 1, 0],  # Z1⊗Z2⊗I
            [0, 1, 1]   # I⊗Z2⊗Z3
        ]
    
    def encode_logical_zero(self) -> QuantumState:
        """Encode logical |0⟩ as |000⟩."""
        # Create |000⟩ state
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        state_vector[0] = 1.0  # |000⟩ corresponds to index 0
        return QuantumState(state_vector, self.n_qubits)
    
    def encode_logical_one(self) -> QuantumState:
        """Encode logical |1⟩ as |111⟩."""
        # Create |111⟩ state
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        state_vector[-1] = 1.0  # |111⟩ corresponds to index 7
        return QuantumState(state_vector, self.n_qubits)
    
    def measure_syndrome(self, state: QuantumState) -> List[int]:
        """Measure syndrome for 3-qubit bit flip code."""
        # Find the dominant state (highest probability)
        probs = np.abs(state.state_vector)**2
        dominant_idx = np.argmax(probs)
        
        # Convert to binary string (3 bits)
        binary_state = format(dominant_idx, '03b')
        
        # Calculate syndrome based on parity checks
        # S1: parity of qubits 0 and 1
        s1 = (int(binary_state[0]) + int(binary_state[1])) % 2
        
        # S2: parity of qubits 1 and 2  
        s2 = (int(binary_state[1]) + int(binary_state[2])) % 2
        
        return [s1, s2]
    
    def decode_and_correct(self, state: QuantumState, syndrome: List[int]) -> bool:
        """Apply correction based on syndrome measurement."""
        s1, s2 = syndrome
        
        # Syndrome lookup table for 3-qubit bit flip code
        if s1 == 0 and s2 == 0:
            # No error detected
            return True
        elif s1 == 1 and s2 == 0:
            # Error on qubit 0
            state.apply_x_error(0)
            return True
        elif s1 == 1 and s2 == 1:
            # Error on qubit 1
            state.apply_x_error(1)
            return True
        elif s1 == 0 and s2 == 1:
            # Error on qubit 2
            state.apply_x_error(2)
            return True
        else:
            # This shouldn't happen for valid syndromes
            return False

class FiveQubitCode(QECCode):
    """5-qubit quantum error correction code."""
    
    def __init__(self):
        super().__init__(n_qubits=5, k_qubits=1, distance=3)
        
        # Stabilizer generators for 5-qubit code
        self.stabilizers = [
            [1, 0, 0, 1, 1],  # X1⊗I⊗I⊗X4⊗X5
            [0, 1, 0, 1, 0],  # I⊗X2⊗I⊗X4⊗I
            [0, 0, 1, 0, 1],  # I⊗I⊗X3⊗I⊗X5
            [1, 1, 1, 1, 0]   # X1⊗X2⊗X3⊗X4⊗I
        ]
    
    def encode_logical_zero(self) -> QuantumState:
        """Encode logical |0⟩ for 5-qubit code."""
        # The logical |0⟩ state for 5-qubit code is:
        # (|00000⟩ + |10010⟩ + |01001⟩ + |10100⟩ + |01010⟩)/√16 + ...
        # For simplicity, we'll use a computational basis approximation
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Logical |0⟩ basis states (simplified)
        logical_zero_states = [
            0b00000,  # |00000⟩
            0b10010,  # |10010⟩
            0b01001,  # |01001⟩
            0b10100,  # |10100⟩
            0b01010   # |01010⟩
        ]
        
        norm_factor = 1.0 / np.sqrt(len(logical_zero_states))
        for state_idx in logical_zero_states:
            state_vector[state_idx] = norm_factor
        
        return QuantumState(state_vector, self.n_qubits)
    
    def encode_logical_one(self) -> QuantumState:
        """Encode logical |1⟩ for 5-qubit code."""
        # Apply logical X to logical |0⟩
        logical_zero = self.encode_logical_zero()
        # For this implementation, we'll flip all qubits as approximation
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Get the logical |0⟩ state and apply logical X
        for i, amp in enumerate(logical_zero.state_vector):
            if abs(amp) > 1e-10:
                # Flip all bits for logical X approximation
                flipped_state = i ^ ((1 << self.n_qubits) - 1)
                state_vector[flipped_state] = amp
        
        return QuantumState(state_vector, self.n_qubits)
    
    def measure_syndrome(self, state: QuantumState) -> List[int]:
        """Measure syndrome for 5-qubit code."""
        syndrome = []
        
        # For each stabilizer, calculate expectation value
        for stabilizer in self.stabilizers:
            expectation = self._calculate_stabilizer_expectation(state, stabilizer)
            syndrome.append(1 if expectation < 0 else 0)
        
        return syndrome
    
    def _calculate_stabilizer_expectation(self, state: QuantumState, stabilizer: List[int]) -> float:
        """Calculate expectation value of a stabilizer measurement."""
        probs = np.abs(state.state_vector)**2
        expectation = 0
        
        for i, prob in enumerate(probs):
            if prob > 1e-10:
                # Calculate parity based on stabilizer
                parity = 0
                binary = format(i, f'0{self.n_qubits}b')
                for j, qubit_in_stabilizer in enumerate(stabilizer):
                    if qubit_in_stabilizer == 1:
                        parity += int(binary[j])
                parity = parity % 2
                
                if parity == 1:
                    expectation -= prob
                else:
                    expectation += prob
        
        return expectation
    
    def decode_and_correct(self, state: QuantumState, syndrome: List[int]) -> bool:
        """Apply correction based on syndrome measurement for 5-qubit code."""
        syndrome_int = sum(bit * (2**i) for i, bit in enumerate(syndrome))
        
        # Syndrome lookup table for single qubit errors
        error_lookup = {
            0: None,     # No error
            1: 0,        # Error on qubit 0
            2: 1,        # Error on qubit 1
            4: 2,        # Error on qubit 2
            8: 3,        # Error on qubit 3
            3: 4,        # Error on qubit 4
            # Additional syndrome patterns for other error types
        }
        
        if syndrome_int in error_lookup:
            error_qubit = error_lookup[syndrome_int]
            if error_qubit is not None and error_qubit < state.n_qubits:
                state.apply_x_error(error_qubit)
            return True
        else:
            # Unknown syndrome - might be uncorrectable error
            return False
