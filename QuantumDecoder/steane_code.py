"""
Steane 7-qubit Code Implementation
The first quantum error correction code capable of correcting both bit-flip and phase-flip errors.
Perfect for educational demonstrations of CSS codes.
"""

import numpy as np
from typing import List, Tuple
from quantum_states import QuantumState
from qec_codes import QECCode

class SteaneCode(QECCode):
    """Steane [[7,1,3]] quantum error correction code - CSS code with transversal gates"""
    
    def __init__(self):
        super().__init__(n_qubits=7, k_qubits=1, distance=3)
        
        # Steane code stabilizer generators (CSS code structure)
        # X-type stabilizers (from classical Hamming code)
        self.x_stabilizers = [
            [1, 1, 1, 0, 1, 0, 0],  # X₁X₂X₃X₅
            [1, 1, 0, 1, 0, 1, 0],  # X₁X₂X₄X₆  
            [1, 0, 1, 1, 0, 0, 1]   # X₁X₃X₄X₇
        ]
        
        # Z-type stabilizers (same pattern as X)
        self.z_stabilizers = [
            [1, 1, 1, 0, 1, 0, 0],  # Z₁Z₂Z₃Z₅
            [1, 1, 0, 1, 0, 1, 0],  # Z₁Z₂Z₄Z₆
            [1, 0, 1, 1, 0, 0, 1]   # Z₁Z₃Z₄Z₇
        ]
        
        # Logical operators
        self.logical_x = [1, 1, 1, 1, 1, 1, 1]  # X̄ = X₁X₂X₃X₄X₅X₆X₇
        self.logical_z = [1, 1, 1, 1, 1, 1, 1]  # Z̄ = Z₁Z₂Z₃Z₄Z₅Z₆Z₇
        
        # Syndrome lookup table for error correction
        self.syndrome_table = self._build_syndrome_table()
    
    def encode_logical_zero(self) -> QuantumState:
        """Encode logical |0⟩ using Steane code"""
        # Steane code logical |0⟩ is superposition of all even-parity codewords
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Generate all 7-bit strings and keep only valid codewords
        valid_codewords = []
        for i in range(2**self.n_qubits):
            binary = format(i, '07b')
            if self._is_valid_codeword(binary):
                valid_codewords.append(i)
        
        # Create equal superposition of valid codewords
        if valid_codewords:
            norm_factor = 1.0 / np.sqrt(len(valid_codewords))
            for codeword in valid_codewords:
                state_vector[codeword] = norm_factor
        else:
            # Fallback: use computational basis states
            state_vector[0] = 1.0
            
        return QuantumState(state_vector, self.n_qubits)
    
    def encode_logical_one(self) -> QuantumState:
        """Encode logical |1⟩ using Steane code"""
        # Apply logical X to logical |0⟩
        logical_zero = self.encode_logical_zero()
        state_vector = np.zeros(2**self.n_qubits, dtype=complex)
        
        # Apply logical X operation (flip all qubits in this simplified version)
        for i, amp in enumerate(logical_zero.state_vector):
            if abs(amp) > 1e-10:
                # Apply logical X by flipping specific pattern
                flipped_state = i ^ 0b1111111  # Simplified: flip all bits
                state_vector[flipped_state] = amp
                
        return QuantumState(state_vector, self.n_qubits)
    
    def measure_syndrome(self, state: QuantumState) -> List[int]:
        """Measure syndrome for Steane code (6 stabilizers total)"""
        probs = np.abs(state.state_vector)**2
        dominant_idx = np.argmax(probs)
        binary_state = format(dominant_idx, '07b')
        
        syndrome = []
        
        # X-stabilizer measurements
        for stabilizer in self.x_stabilizers:
            parity = sum(int(binary_state[i]) * stabilizer[i] for i in range(7)) % 2
            syndrome.append(parity)
            
        # Z-stabilizer measurements
        for stabilizer in self.z_stabilizers:
            parity = sum(int(binary_state[i]) * stabilizer[i] for i in range(7)) % 2
            syndrome.append(parity)
            
        return syndrome
    
    def decode_and_correct(self, state: QuantumState, syndrome: List[int]) -> bool:
        """Apply correction based on syndrome measurement"""
        syndrome_str = ''.join(map(str, syndrome))
        
        if syndrome_str == '000000':
            return True  # No error
            
        # Look up error pattern in syndrome table
        if syndrome_str in self.syndrome_table:
            error_pattern = self.syndrome_table[syndrome_str]
            
            # Apply corrections
            for qubit_idx, error_type in enumerate(error_pattern):
                if error_type == 'X':
                    state.apply_x_error(qubit_idx)
                elif error_type == 'Z':
                    state.apply_z_error(qubit_idx)
                elif error_type == 'Y':
                    state.apply_y_error(qubit_idx)
                    
            return True
        else:
            # Unknown syndrome - might be uncorrectable
            return False
    
    def _is_valid_codeword(self, binary_string: str) -> bool:
        """Check if a 7-bit string is a valid Steane codeword"""
        bits = [int(b) for b in binary_string]
        
        # Check all stabilizer constraints
        for stabilizer in self.x_stabilizers:
            parity = sum(bits[i] * stabilizer[i] for i in range(7)) % 2
            if parity != 0:
                return False
                
        return True
    
    def _build_syndrome_table(self) -> dict:
        """Build lookup table mapping syndromes to error patterns"""
        table = {}
        
        # Single qubit X errors
        for i in range(7):
            syndrome = self._calculate_syndrome_for_error(i, 'X')
            table[''.join(map(str, syndrome))] = ['X' if j == i else 'I' for j in range(7)]
            
        # Single qubit Z errors  
        for i in range(7):
            syndrome = self._calculate_syndrome_for_error(i, 'Z')
            table[''.join(map(str, syndrome))] = ['Z' if j == i else 'I' for j in range(7)]
            
        # Single qubit Y errors
        for i in range(7):
            syndrome = self._calculate_syndrome_for_error(i, 'Y')
            table[''.join(map(str, syndrome))] = ['Y' if j == i else 'I' for j in range(7)]
            
        return table
    
    def _calculate_syndrome_for_error(self, qubit: int, error_type: str) -> List[int]:
        """Calculate expected syndrome for a specific error"""
        syndrome = []
        
        # X-stabilizer syndromes
        for stabilizer in self.x_stabilizers:
            if error_type in ['X', 'Y']:
                syndrome.append(stabilizer[qubit])
            else:
                syndrome.append(0)
                
        # Z-stabilizer syndromes  
        for stabilizer in self.z_stabilizers:
            if error_type in ['Z', 'Y']:
                syndrome.append(stabilizer[qubit])
            else:
                syndrome.append(0)
                
        return syndrome
    
    def get_code_parameters(self) -> dict:
        """Return code parameters for display"""
        return {
            'name': 'Steane [[7,1,3]] Code',
            'n_qubits': self.n_qubits,
            'k_qubits': self.k_qubits, 
            'distance': self.distance,
            'type': 'CSS Code',
            'corrects': 'Any single qubit error (X, Y, Z)',
            'stabilizers': len(self.x_stabilizers) + len(self.z_stabilizers),
            'transversal_gates': ['H', 'CNOT', 'S'],
            'threshold': '~10^-4 (theoretical)'
        }
    
    def supports_transversal_gates(self) -> bool:
        """Check if code supports transversal gate implementation"""
        return True
    
    def get_logical_operators(self) -> dict:
        """Return logical X and Z operators"""
        return {
            'logical_X': self.logical_x,
            'logical_Z': self.logical_z
        }