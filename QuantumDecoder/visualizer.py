import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

from quantum_states import QuantumState
from qec_codes import QECCode

class QuantumStateVisualizer:
    """Visualizer for quantum states and error correction processes."""
    
    def __init__(self, qec_code: QECCode):
        """
        Initialize visualizer with QEC code.
        
        Args:
            qec_code: The quantum error correction code being visualized
        """
        self.qec_code = qec_code
        self.colors = px.colors.qualitative.Set3
    
    def plot_quantum_state(self, state: QuantumState) -> go.Figure:
        """
        Create a visualization of the quantum state.
        
        Args:
            state: Quantum state to visualize
            
        Returns:
            Plotly figure showing state amplitudes and probabilities
        """
        # Get state data
        probs = state.get_probabilities()
        labels = state.get_computational_basis_labels()
        amplitudes = state.state_vector
        
        # Filter out very small amplitudes for clarity
        significant_indices = np.where(probs > 1e-6)[0]
        
        if len(significant_indices) == 0:
            # Edge case: no significant amplitudes
            significant_indices = [0]
        
        filtered_probs = probs[significant_indices]
        filtered_labels = [labels[i] for i in significant_indices]
        filtered_amplitudes = amplitudes[significant_indices]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Probability Distribution',
                'Amplitude Magnitudes', 
                'Amplitude Phases',
                'Qubit States (Individual)'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Probability distribution
        fig.add_trace(
            go.Bar(
                x=filtered_labels,
                y=filtered_probs,
                name="Probability",
                marker_color=self.colors[0],
                text=[f"{p:.3f}" for p in filtered_probs],
                textposition="outside"
            ),
            row=1, col=1
        )
        
        # 2. Amplitude magnitudes
        amp_magnitudes = np.abs(filtered_amplitudes)
        fig.add_trace(
            go.Bar(
                x=filtered_labels,
                y=amp_magnitudes,
                name="Amplitude Magnitude",
                marker_color=self.colors[1],
                text=[f"{a:.3f}" for a in amp_magnitudes],
                textposition="outside"
            ),
            row=1, col=2
        )
        
        # 3. Amplitude phases
        phases = np.angle(filtered_amplitudes)
        fig.add_trace(
            go.Scatter(
                x=np.real(filtered_amplitudes),
                y=np.imag(filtered_amplitudes),
                mode='markers+text',
                text=filtered_labels,
                textposition="top center",
                name="Complex Amplitudes",
                marker=dict(
                    size=10,
                    color=phases,
                    colorscale="hsv",
                    colorbar=dict(title="Phase (radians)"),
                    line=dict(width=1, color="black")
                )
            ),
            row=2, col=1
        )
        
        # Add unit circle for reference
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name="Unit Circle",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Individual qubit states
        qubit_probs = self._calculate_individual_qubit_probabilities(state)
        qubit_labels = [f"Q{i}" for i in range(state.n_qubits)]
        
        fig.add_trace(
            go.Bar(
                x=qubit_labels,
                y=qubit_probs,
                name="Qubit |1⟩ Probability",
                marker_color=self.colors[2],
                text=[f"{p:.3f}" for p in qubit_probs],
                textposition="outside"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Quantum State Visualization ({self.qec_code.n_qubits} qubits)",
            height=600,
            showlegend=True
        )
        
        # Update subplot axes
        fig.update_xaxes(title_text="Basis States", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        
        fig.update_xaxes(title_text="Basis States", row=1, col=2)
        fig.update_yaxes(title_text="Amplitude Magnitude", row=1, col=2)
        
        fig.update_xaxes(title_text="Real Part", row=2, col=1)
        fig.update_yaxes(title_text="Imaginary Part", row=2, col=1)
        
        fig.update_xaxes(title_text="Qubit", row=2, col=2)
        fig.update_yaxes(title_text="P(|1⟩)", row=2, col=2, range=[0, 1])
        
        return fig
    
    def plot_syndrome(self, syndrome: List[int]) -> go.Figure:
        """
        Visualize the error syndrome.
        
        Args:
            syndrome: List of syndrome measurement results
            
        Returns:
            Plotly figure showing syndrome pattern
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Syndrome Pattern", "Syndrome Interpretation"),
            specs=[[{"type": "bar"}, {"type": "table"}]]
        )
        
        # Syndrome pattern visualization
        syndrome_labels = [f"S{i+1}" for i in range(len(syndrome))]
        colors = [self.colors[3] if s == 1 else self.colors[4] for s in syndrome]
        
        fig.add_trace(
            go.Bar(
                x=syndrome_labels,
                y=syndrome,
                name="Syndrome",
                marker_color=colors,
                text=syndrome,
                textposition="inside"
            ),
            row=1, col=1
        )
        
        # Syndrome-to-correction mapping table
        mapping = self._get_syndrome_correction_mapping(syndrome)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Syndrome", "Error Location", "Correction", "Action"],
                    fill_color=self.colors[0],
                    align="left",
                    font=dict(size=12, color="white")
                ),
                cells=dict(
                    values=[
                        mapping["syndromes"],
                        mapping["locations"],
                        mapping["corrections"],
                        mapping["actions"]
                    ],
                    fill_color=[self.colors[1] if i % 2 == 0 else self.colors[2] for i in range(len(mapping["syndromes"]))],
                    align="left",
                    font=dict(size=11)
                )
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Error Syndrome Analysis & Correction Mapping",
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Syndrome Value", range=[-0.1, 1.1], row=1, col=1)
        
        return fig
    
    def plot_error_correction_timeline(self, timeline_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a timeline visualization of the error correction process.
        
        Args:
            timeline_data: List of dictionaries containing timeline events
            
        Returns:
            Plotly figure showing the correction timeline
        """
        fig = go.Figure()
        
        steps = [data["step"] for data in timeline_data]
        descriptions = [data["description"] for data in timeline_data]
        success = [data.get("success", True) for data in timeline_data]
        
        colors = [self.colors[0] if s else self.colors[5] for s in success]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(steps))),
                y=[1] * len(steps),
                mode='markers+text',
                text=steps,
                textposition="top center",
                marker=dict(
                    size=20,
                    color=colors,
                    line=dict(width=2, color="black")
                ),
                hovertext=descriptions,
                hoverinfo="text",
                name="Correction Steps"
            )
        )
        
        # Add connecting lines
        fig.add_trace(
            go.Scatter(
                x=list(range(len(steps))),
                y=[1] * len(steps),
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            )
        )
        
        fig.update_layout(
            title="Error Correction Process Timeline",
            xaxis_title="Step Number",
            yaxis_title="",
            height=300,
            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
            showlegend=False
        )
        
        return fig
    
    def plot_bloch_spheres(self, state: QuantumState) -> go.Figure:
        """
        Plot Bloch sphere representation for individual qubits.
        
        Args:
            state: Quantum state to visualize
            
        Returns:
            Plotly figure with Bloch sphere representations
        """
        n_qubits = state.n_qubits
        cols = min(3, n_qubits)
        rows = (n_qubits + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=[f"Qubit {i}" for i in range(n_qubits)]
        )
        
        for i in range(n_qubits):
            row = i // cols + 1
            col = i % cols + 1
            
            # Get Bloch coordinates (simplified)
            x, y, z = state.get_bloch_sphere_coordinates(i)
            
            # Create sphere surface
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            sphere_x = np.outer(np.cos(u), np.sin(v))
            sphere_y = np.outer(np.sin(u), np.sin(v))
            sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(
                go.Surface(
                    x=sphere_x, y=sphere_y, z=sphere_z,
                    opacity=0.3,
                    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                    showscale=False
                ),
                row=row, col=col
            )
            
            # Add state vector
            fig.add_trace(
                go.Scatter3d(
                    x=[0, x], y=[0, y], z=[0, z],
                    mode='lines+markers',
                    line=dict(color='red', width=5),
                    marker=dict(size=[3, 8], color=['blue', 'red']),
                    name=f"Q{i} State"
                ),
                row=row, col=col
            )
            
            # Add coordinate axes
            axes_data = [
                ([0, 1], [0, 0], [0, 0], 'X'),
                ([0, 0], [0, 1], [0, 0], 'Y'),
                ([0, 0], [0, 0], [0, 1], 'Z')
            ]
            
            for ax_x, ax_y, ax_z, label in axes_data:
                fig.add_trace(
                    go.Scatter3d(
                        x=ax_x, y=ax_y, z=ax_z,
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Bloch Sphere Representation",
            height=400 * rows,
            showlegend=False
        )
        
        return fig
    
    def _calculate_individual_qubit_probabilities(self, state: QuantumState) -> List[float]:
        """Calculate the probability of each qubit being in |1⟩ state."""
        probs = []
        
        for qubit_idx in range(state.n_qubits):
            prob_1 = 0
            for i, amplitude in enumerate(state.state_vector):
                if (i >> (state.n_qubits - 1 - qubit_idx)) & 1:
                    prob_1 += abs(amplitude)**2
            probs.append(prob_1)
        
        return probs
    
    def calculate_state_overlap(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate the overlap (fidelity) between two quantum states."""
        # Calculate the inner product between state vectors
        overlap = np.abs(np.vdot(state1.state_vector, state2.state_vector))**2
        return float(overlap)
    
    def plot_error_patterns(self, error_types: List[str], error_qubits: List[int], n_qubits: int) -> go.Figure:
        """Plot error patterns and statistics."""
        # Create error pattern analysis
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Error Type Distribution", "Error Qubit Distribution", "Error Timeline"),
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Error type distribution
        if error_types:
            type_counts = pd.Series(error_types).value_counts()
            fig.add_trace(
                go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    name="Error Types",
                    marker_colors=self.colors[:len(type_counts)]
                ),
                row=1, col=1
            )
        
        # Error qubit distribution
        if error_qubits:
            qubit_counts = pd.Series(error_qubits).value_counts().reindex(range(n_qubits), fill_value=0)
            fig.add_trace(
                go.Bar(
                    x=[f"Q{i}" for i in range(n_qubits)],
                    y=qubit_counts.values,
                    name="Qubit Errors",
                    marker_color=self.colors[0]
                ),
                row=1, col=2
            )
        
        # Error timeline
        if error_types and error_qubits:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(error_types))),
                    y=error_qubits,
                    mode='markers+lines',
                    name="Error Timeline",
                    marker=dict(
                        size=10,
                        color=[self.colors[0] if t == 'X' else self.colors[1] if t == 'Z' else self.colors[2] for t in error_types]
                    ),
                    text=[f"{t} error" for t in error_types],
                    hoverinfo="text"
                ),
                row=1, col=3
            )
        
        fig.update_layout(
            title="Error Pattern Analysis",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _get_syndrome_correction_mapping(self, current_syndrome: List[int]) -> Dict[str, List[str]]:
        """Get complete syndrome-to-correction mapping for educational display."""
        if len(current_syndrome) == 2:  # 3-qubit code
            return {
                "syndromes": ["00", "01", "10", "11"],
                "locations": ["No Error", "Qubit 2", "Qubit 0", "Qubit 1"],
                "corrections": ["None", "Apply X₂", "Apply X₀", "Apply X₁"],
                "actions": [
                    "✅ State OK" if current_syndrome == [0,0] else "State OK",
                    "🔧 Flip Q2" if current_syndrome == [0,1] else "Flip Q2", 
                    "🔧 Flip Q0" if current_syndrome == [1,0] else "Flip Q0",
                    "🔧 Flip Q1" if current_syndrome == [1,1] else "Flip Q1"
                ]
            }
        else:  # 5-qubit code
            return {
                "syndromes": ["0000", "0001", "0010", "0100", "1000", "..."],
                "locations": ["No Error", "Error Q4", "Error Q3", "Error Q2", "Error Q1", "Others"],
                "corrections": ["None", "Apply X₄", "Apply X₃", "Apply X₂", "Apply X₁", "Pattern"],
                "actions": [
                    "✅ Perfect" if all(s == 0 for s in current_syndrome) else "Perfect",
                    "🔧 Correct" if current_syndrome == [0,0,0,1] else "Correct",
                    "🔧 Correct" if current_syndrome == [0,0,1,0] else "Correct", 
                    "🔧 Correct" if current_syndrome == [0,1,0,0] else "Correct",
                    "🔧 Correct" if current_syndrome == [1,0,0,0] else "Correct",
                    "📊 Complex"
                ]
            }
    
    def _interpret_syndrome(self, syndrome: List[int]) -> Dict[str, str]:
        """Interpret the syndrome measurement results."""
        syndrome_int = sum(bit * (2**i) for i, bit in enumerate(syndrome))
        
        if hasattr(self.qec_code, 'decode_and_correct'):
            # Try to determine error type based on QEC code
            if all(s == 0 for s in syndrome):
                return {
                    "error_type": "No Error",
                    "affected_qubits": "None",
                    "correctable": "N/A"
                }
            else:
                return {
                    "error_type": "Single Qubit Error",
                    "affected_qubits": f"Syndrome: {''.join(map(str, syndrome))}",
                    "correctable": "Yes"
                }
        else:
            return {
                "error_type": "Unknown",
                "affected_qubits": "Analysis pending",
                "correctable": "Unknown"
            }
