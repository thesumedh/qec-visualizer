"""
Enhanced 2D/3D Quantum Visualizations - PanQEC Style
For CQHack25 - Classiq Track
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Enhanced3DVisualizer:
    """Create impressive 2D/3D visualizations like PanQEC"""
    
    def create_3d_surface_code_lattice(self, distance=3, highlight_errors=None):
        """Create 3D Surface Code lattice visualization"""
        
        # Create 3D lattice points
        x_coords, y_coords, z_coords = [], [], []
        colors = []
        
        for i in range(distance):
            for j in range(distance):
                x_coords.append(i)
                y_coords.append(j)
                z_coords.append(0)  # Data qubits on z=0 plane
                
                # Color based on error status
                if highlight_errors and (i * distance + j) in highlight_errors:
                    colors.append('red')  # Error qubit
                else:
                    colors.append('lightblue')  # Normal qubit
        
        # Add stabilizer qubits
        for i in range(distance-1):
            for j in range(distance-1):
                x_coords.append(i + 0.5)
                y_coords.append(j + 0.5)
                z_coords.append(0.2)  # Stabilizers slightly above
                colors.append('orange')
        
        fig = go.Figure()
        
        # Data qubits
        fig.add_trace(go.Scatter3d(
            x=x_coords[:distance*distance],
            y=y_coords[:distance*distance],
            z=z_coords[:distance*distance],
            mode='markers+text',
            marker=dict(
                size=12,
                color=colors[:distance*distance],
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=[f'D{i}' for i in range(distance*distance)],
            textposition="middle center",
            name="Data Qubits"
        ))
        
        # Stabilizer qubits
        if len(x_coords) > distance*distance:
            fig.add_trace(go.Scatter3d(
                x=x_coords[distance*distance:],
                y=y_coords[distance*distance:],
                z=z_coords[distance*distance:],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color='orange',
                    symbol='diamond',
                    opacity=0.7
                ),
                text=[f'S{i}' for i in range(len(x_coords) - distance*distance)],
                textposition="middle center",
                name="Stabilizers"
            ))
        
        # Add connections
        for i in range(distance-1):
            for j in range(distance):
                # Horizontal connections
                fig.add_trace(go.Scatter3d(
                    x=[i, i+1],
                    y=[j, j],
                    z=[0, 0],
                    mode='lines',
                    line=dict(color='gray', width=3),
                    showlegend=False
                ))
                
                # Vertical connections
                if j < distance-1:
                    fig.add_trace(go.Scatter3d(
                        x=[i, i],
                        y=[j, j+1],
                        z=[0, 0],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title="3D Surface Code Lattice (PanQEC Style)",
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_syndrome_heatmap_2d(self, syndrome_history):
        """Create 2D syndrome pattern heatmap"""
        
        if not syndrome_history:
            return None
            
        # Create syndrome matrix
        n_measurements = len(syndrome_history)
        syndrome_matrix = []
        
        for i, measurement in enumerate(syndrome_history):
            if 'syndrome' in measurement:
                syndrome_matrix.append(measurement['syndrome'])
            else:
                syndrome_matrix.append([0, 0])  # Default
        
        syndrome_array = np.array(syndrome_matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=syndrome_array.T,
            x=list(range(n_measurements)),
            y=['S1', 'S2'],
            colorscale='RdYlBu_r',
            showscale=True
        ))
        
        fig.update_layout(
            title="Syndrome Pattern Evolution",
            xaxis_title="Measurement Round",
            yaxis_title="Stabilizer",
            height=300
        )
        
        return fig
    
    def create_bloch_sphere_3d(self, quantum_state):
        """Create 3D Bloch sphere representation"""
        
        # Calculate Bloch vector components
        if len(quantum_state.state_vector) >= 2:
            alpha = quantum_state.state_vector[0]
            beta = quantum_state.state_vector[1] if len(quantum_state.state_vector) > 1 else 0
            
            # Bloch sphere coordinates
            theta = 2 * np.arccos(np.abs(alpha))
            phi = np.angle(beta) - np.angle(alpha)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
        else:
            x, y, z = 0, 0, 1
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add sphere surface
        fig.add_trace(go.Surface(
            x=sphere_x, y=sphere_y, z=sphere_z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name="Bloch Sphere"
        ))
        
        # Add state vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=[0, 12], color=['red', 'red']),
            name="State Vector"
        ))
        
        # Add coordinate axes
        fig.add_trace(go.Scatter3d(
            x=[-1, 1], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='black', width=2),
            name="X-axis", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-1, 1], z=[0, 0],
            mode='lines', line=dict(color='black', width=2),
            name="Y-axis", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1, 1],
            mode='lines', line=dict(color='black', width=2),
            name="Z-axis", showlegend=False
        ))
        
        fig.update_layout(
            title="3D Bloch Sphere Representation",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=500
        )
        
        return fig
    
    def create_error_propagation_animation(self, qec_code, error_qubit, error_type):
        """Create animated error propagation visualization"""
        
        n_qubits = qec_code.n_qubits
        
        # Create frames for animation
        frames = []
        
        # Initial state
        colors = ['lightblue'] * n_qubits
        frame_data = go.Scatter(
            x=list(range(n_qubits)),
            y=[1] * n_qubits,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=[f'Q{i}' for i in range(n_qubits)],
            textposition="middle center"
        )
        frames.append(go.Frame(data=[frame_data], name="initial"))
        
        # Error injection
        colors[error_qubit] = 'red'
        frame_data = go.Scatter(
            x=list(range(n_qubits)),
            y=[1] * n_qubits,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=[f'Q{i}' for i in range(n_qubits)],
            textposition="middle center"
        )
        frames.append(go.Frame(data=[frame_data], name="error"))
        
        # Syndrome measurement
        colors = ['yellow'] * n_qubits
        colors[error_qubit] = 'red'
        frame_data = go.Scatter(
            x=list(range(n_qubits)),
            y=[1] * n_qubits,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=[f'Q{i}' for i in range(n_qubits)],
            textposition="middle center"
        )
        frames.append(go.Frame(data=[frame_data], name="syndrome"))
        
        # Correction
        colors = ['lightgreen'] * n_qubits
        frame_data = go.Scatter(
            x=list(range(n_qubits)),
            y=[1] * n_qubits,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=[f'Q{i}' for i in range(n_qubits)],
            textposition="middle center"
        )
        frames.append(go.Frame(data=[frame_data], name="corrected"))
        
        # Create figure with animation
        fig = go.Figure(
            data=[frames[0].data[0]],
            frames=frames
        )
        
        fig.update_layout(
            title=f"Error Propagation Animation - {error_type} on Qubit {error_qubit}",
            xaxis=dict(range=[-0.5, n_qubits-0.5], title="Qubit Index"),
            yaxis=dict(range=[0.5, 1.5], showticklabels=False),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000, "redraw": True},
                                      "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            height=400
        )
        
        return fig
    
    def create_fidelity_landscape_3d(self, trial_data):
        """Create 3D fidelity landscape"""
        
        if not trial_data:
            return None
            
        # Extract data
        trials = [d['trial'] for d in trial_data]
        fidelities = [d['fidelity'] for d in trial_data]
        error_types = [d.get('error_type', 'None') for d in trial_data]
        
        # Create 3D surface
        fig = go.Figure()
        
        # Group by error type
        error_type_map = {'X': 0, 'Y': 1, 'Z': 2, 'None': 3}
        
        x_vals = []
        y_vals = []
        z_vals = []
        colors = []
        
        for i, (trial, fidelity, error_type) in enumerate(zip(trials, fidelities, error_types)):
            x_vals.append(trial)
            y_vals.append(error_type_map.get(error_type, 3))
            z_vals.append(fidelity)
            colors.append(fidelity)
        
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Fidelity")
            ),
            text=[f"Trial {t}<br>Error: {e}<br>Fidelity: {f:.3f}" 
                  for t, e, f in zip(trials, error_types, fidelities)],
            hovertemplate="%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="3D Fidelity Landscape",
            scene=dict(
                xaxis_title="Trial Number",
                yaxis_title="Error Type",
                zaxis_title="Fidelity",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['X', 'Y', 'Z', 'None']
                )
            ),
            height=600
        )
        
        return fig

# Global instance
enhanced_viz = Enhanced3DVisualizer()