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
    
    def create_3d_surface_code_lattice(self, distance=3, highlight_errors=None, step_info=None):
        """Create dynamic PanQEC-style 3D Surface Code lattice with step animations"""
        
        fig = go.Figure()
        
        # Step-based animation effects
        if step_info:
            current_step = step_info.get('current_step', 1)
            error_applied = step_info.get('error_applied', False)
            syndrome_measured = step_info.get('syndrome_measured', False)
            correction_applied = step_info.get('correction_applied', False)
        else:
            current_step = 1
            error_applied = False
            syndrome_measured = False
            correction_applied = False
        
        # Data qubits with step-based coloring
        for i in range(distance):
            for j in range(distance):
                qubit_id = i * distance + j
                is_error = highlight_errors and qubit_id in highlight_errors
                
                # Dynamic coloring based on step
                if correction_applied:
                    color = '#4caf50'  # Green for corrected
                    size = 22
                elif is_error and error_applied:
                    color = '#ff4444'  # Red for error
                    size = 25
                elif syndrome_measured:
                    color = '#ffeb3b'  # Yellow during measurement
                    size = 20
                else:
                    color = '#00d4ff'  # Cyan for normal
                    size = 20
                
                fig.add_trace(go.Scatter3d(
                    x=[i], y=[j], z=[0],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.9,
                        symbol='circle',
                        line=dict(width=2, color='#333')
                    ),
                    text=[f'Q{qubit_id}'],
                    textposition="middle center",
                    name=f"Data Qubit {qubit_id}",
                    showlegend=False,
                    hovertemplate=f"Data Qubit {qubit_id}<br>Status: {'ERROR' if is_error else 'OK'}<extra></extra>"
                ))
        
        # X-stabilizers with dynamic syndrome indication
        for i in range(distance-1):
            for j in range(distance-1):
                stab_id = i * (distance-1) + j
                syndrome_active = highlight_errors and len(highlight_errors) > 0 and syndrome_measured
                
                # Pulsing effect during syndrome measurement
                if syndrome_measured and not correction_applied:
                    color = '#ffeb3b' if syndrome_active else '#ff9800'
                    size = 18 if syndrome_active else 15
                elif correction_applied:
                    color = '#4caf50'
                    size = 15
                else:
                    color = '#ff9800'
                    size = 15
                
                fig.add_trace(go.Scatter3d(
                    x=[i+0.5], y=[j+0.5], z=[0.3],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        symbol='diamond',
                        line=dict(width=2, color='#666')
                    ),
                    text=[f'X{stab_id}'],
                    textposition="middle center",
                    name=f"X-Stabilizer {stab_id}",
                    showlegend=False,
                    hovertemplate=f"X-Stabilizer {stab_id}<br>Syndrome: {'TRIGGERED' if syndrome_active else 'OK'}<extra></extra>"
                ))
        
        # Z-stabilizers with dynamic syndrome indication
        for i in range(distance-1):
            for j in range(distance-1):
                stab_id = i * (distance-1) + j
                syndrome_active = highlight_errors and len(highlight_errors) > 0 and syndrome_measured
                
                if syndrome_measured and not correction_applied:
                    color = '#ff4444' if syndrome_active else '#4caf50'
                    size = 18 if syndrome_active else 15
                elif correction_applied:
                    color = '#4caf50'
                    size = 15
                else:
                    color = '#4caf50'
                    size = 15
                
                fig.add_trace(go.Scatter3d(
                    x=[i+0.5], y=[j+0.5], z=[-0.3],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        symbol='square',
                        line=dict(width=2, color='#666')
                    ),
                    text=[f'Z{stab_id}'],
                    textposition="middle center",
                    name=f"Z-Stabilizer {stab_id}",
                    showlegend=False,
                    hovertemplate=f"Z-Stabilizer {stab_id}<br>Syndrome: {'TRIGGERED' if syndrome_active else 'OK'}<extra></extra>"
                ))
        
        # Dynamic connection lines based on step
        line_opacity = 0.6 if syndrome_measured else 0.3
        line_width = 3 if syndrome_measured else 2
        
        for i in range(distance-1):
            for j in range(distance-1):
                connections = [(i,j), (i+1,j), (i,j+1), (i+1,j+1)]
                
                # X-stabilizer connections
                for qi, qj in connections:
                    fig.add_trace(go.Scatter3d(
                        x=[qi, i+0.5], y=[qj, j+0.5], z=[0, 0.3],
                        mode='lines',
                        line=dict(color='#ff9800', width=line_width, dash='dot'),
                        opacity=line_opacity,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Z-stabilizer connections
                for qi, qj in connections:
                    fig.add_trace(go.Scatter3d(
                        x=[qi, i+0.5], y=[qj, j+0.5], z=[0, -0.3],
                        mode='lines',
                        line=dict(color='#4caf50', width=line_width, dash='dot'),
                        opacity=line_opacity,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Step-based title
        if step_info:
            if correction_applied:
                title = "✅ Surface Code - Error Corrected"
            elif syndrome_measured:
                title = "🎯 Surface Code - Syndrome Measurement"
            elif error_applied:
                title = "🚨 Surface Code - Error Detected"
            else:
                title = "🔵 Surface Code - Initial State"
        else:
            title = "🌍 3D Surface Code Lattice (PanQEC Style)"
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position", 
                zaxis_title="Z Position",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
                aspectmode='cube',
                bgcolor='rgba(26,26,46,0.8)'
            ),
            height=700,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
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
    
    def create_bloch_sphere_3d(self, quantum_state, step_info=None):
        """Create dynamic 3D Bloch sphere with step-based animations"""
        
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
        
        # Create sphere surface with step-based coloring
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Dynamic sphere color based on step
        if step_info:
            if step_info.get('error_applied'):
                sphere_color = 'Reds'
                sphere_opacity = 0.4
            elif step_info.get('correction_applied'):
                sphere_color = 'Greens'
                sphere_opacity = 0.3
            else:
                sphere_color = 'Blues'
                sphere_opacity = 0.2
        else:
            sphere_color = 'Blues'
            sphere_opacity = 0.2
        
        # Add sphere surface
        fig.add_trace(go.Surface(
            x=sphere_x, y=sphere_y, z=sphere_z,
            opacity=sphere_opacity,
            colorscale=sphere_color,
            showscale=False,
            name="Bloch Sphere"
        ))
        
        # Dynamic state vector color and size
        if step_info:
            if step_info.get('error_applied') and not step_info.get('correction_applied'):
                vector_color = '#ff4444'  # Red for error
                vector_width = 10
                marker_size = 15
            elif step_info.get('correction_applied'):
                vector_color = '#4caf50'  # Green for corrected
                vector_width = 8
                marker_size = 12
            else:
                vector_color = '#00d4ff'  # Cyan for initial
                vector_width = 6
                marker_size = 10
        else:
            vector_color = '#00d4ff'
            vector_width = 6
            marker_size = 10
        
        # Add state vector with animation trail
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color=vector_color, width=vector_width),
            marker=dict(size=[0, marker_size], color=[vector_color, vector_color]),
            name="State Vector"
        ))
        
        # Add coordinate axes with labels
        fig.add_trace(go.Scatter3d(
            x=[-1, 1], y=[0, 0], z=[0, 0],
            mode='lines+text',
            line=dict(color='#666', width=3),
            text=['', '|+⟩'],
            textposition='middle right',
            name="X-axis", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[-1, 1], z=[0, 0],
            mode='lines+text',
            line=dict(color='#666', width=3),
            text=['', '|+i⟩'],
            textposition='middle right',
            name="Y-axis", showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[-1, 1],
            mode='lines+text',
            line=dict(color='#666', width=3),
            text=['|1⟩', '|0⟩'],
            textposition='middle center',
            name="Z-axis", showlegend=False
        ))
        
        # Add step-based title and annotations
        if step_info:
            if step_info.get('error_applied') and not step_info.get('correction_applied'):
                title = "🚨 Error Detected - State Corrupted"
                annotation_text = f"Error Type: {step_info.get('error_type', 'Unknown')}"
            elif step_info.get('correction_applied'):
                fidelity = step_info.get('fidelity', 0.95)
                title = f"✅ State Corrected - Fidelity: {fidelity:.3f}"
                annotation_text = "QEC Recovery Complete"
            else:
                title = "🔵 Initial Encoded State"
                annotation_text = "Perfect Quantum State"
        else:
            title = "3D Bloch Sphere Representation"
            annotation_text = ""
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(26,26,46,0.8)'
            ),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    text=annotation_text,
                    x=0.5, y=0.02,
                    xref='paper', yref='paper',
                    showarrow=False,
                    font=dict(size=12, color='white')
                )
            ] if annotation_text else []
        )
        
        return fig
    
    def create_error_propagation_animation(self, qec_code, error_qubit, error_type, current_step=1):
        """Create dynamic error propagation with step-by-step animation"""
        
        n_qubits = qec_code.n_qubits
        
        # Create animated visualization based on current step
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f'Step {current_step}: QEC Process Animation']
        )
        
        # Base qubit positions
        x_positions = list(range(n_qubits))
        y_positions = [1] * n_qubits
        
        # Step-based visualization
        if current_step == 1:  # Initial state
            colors = ['#00d4ff'] * n_qubits
            sizes = [30] * n_qubits
            texts = [f'Q{i}' for i in range(n_qubits)]
            title_suffix = "Perfect Encoded State"
            
        elif current_step == 2:  # Error injection
            colors = ['#00d4ff'] * n_qubits
            colors[error_qubit] = '#ff4444'
            sizes = [30] * n_qubits
            sizes[error_qubit] = 40  # Larger for error
            texts = [f'Q{i}' for i in range(n_qubits)]
            texts[error_qubit] = f'Q{error_qubit}\n{error_type}'
            title_suffix = f"{error_type} Error on Qubit {error_qubit}"
            
            # Add error propagation lines
            for i in range(n_qubits):
                if i != error_qubit:
                    fig.add_trace(go.Scatter(
                        x=[error_qubit, i], y=[1, 1],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dot'),
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
        elif current_step == 3:  # Syndrome detection
            colors = ['#ffeb3b'] * n_qubits  # Yellow for syndrome measurement
            colors[error_qubit] = '#ff4444'
            sizes = [35] * n_qubits
            texts = [f'Q{i}\nMeas' for i in range(n_qubits)]
            title_suffix = "Syndrome Measurement Active"
            
            # Add syndrome measurement indicators
            for i in range(min(2, n_qubits-1)):
                fig.add_trace(go.Scatter(
                    x=[i + 0.5], y=[1.5],
                    mode='markers+text',
                    marker=dict(size=25, color='orange', symbol='diamond'),
                    text=[f'S{i}'],
                    textposition="middle center",
                    showlegend=False,
                    name=f"Syndrome {i}"
                ))
            
        else:  # Step 4: Correction
            colors = ['#4caf50'] * n_qubits  # Green for corrected
            sizes = [30] * n_qubits
            texts = [f'Q{i}' for i in range(n_qubits)]
            title_suffix = "Error Corrected - State Recovered"
            
            # Add correction indicator
            fig.add_trace(go.Scatter(
                x=[error_qubit], y=[1.3],
                mode='markers+text',
                marker=dict(size=20, color='white', symbol='x'),
                text=['FIX'],
                textposition="middle center",
                showlegend=False,
                name="Correction"
            ))
        
        # Add main qubits
        fig.add_trace(go.Scatter(
            x=x_positions, y=y_positions,
            mode='markers+text',
            marker=dict(size=sizes, color=colors, line=dict(width=2, color='#333')),
            text=texts,
            textposition="middle center",
            showlegend=False,
            name="Qubits",
            hovertemplate="Qubit %{x}<br>Status: %{text}<extra></extra>"
        ))
        
        # Add connecting lines between qubits
        for i in range(n_qubits-1):
            fig.add_trace(go.Scatter(
                x=[i, i+1], y=[1, 1],
                mode='lines',
                line=dict(color='#666', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f"⚡ {title_suffix}",
            xaxis=dict(
                range=[-0.5, n_qubits-0.5],
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[0.5, 2],
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)'
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

    def create_dynamic_state_evolution_2d(self, state_history):
        """Create 2D graph showing quantum state evolution over time"""
        
        if not state_history:
            return None
        
        fig = go.Figure()
        
        # Extract state probabilities over time
        times = list(range(len(state_history)))
        
        # Track top 3 basis states
        all_states = set()
        for state_data in state_history:
            if 'probabilities' in state_data:
                all_states.update(state_data['probabilities'].keys())
        
        # Get top states by maximum probability
        top_states = sorted(list(all_states))[:4]  # Limit to 4 for readability
        
        colors = ['#00d4ff', '#ff9800', '#4caf50', '#ff4444']
        
        for i, state in enumerate(top_states):
            probs = []
            for state_data in state_history:
                if 'probabilities' in state_data:
                    probs.append(state_data['probabilities'].get(state, 0))
                else:
                    probs.append(0)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=probs,
                mode='lines+markers',
                name=f'|{state}⟩',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        # Add step markers
        step_names = ['Initialize', 'Error', 'Syndrome', 'Correct']
        for i, step in enumerate(step_names[:len(times)]):
            fig.add_vline(
                x=i, line_dash="dash", line_color="gray",
                annotation_text=step,
                annotation_position="top"
            )
        
        fig.update_layout(
            title="📈 Quantum State Evolution During QEC",
            xaxis_title="QEC Process Step",
            yaxis_title="Probability",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            font=dict(color='white')
        )
        
        return fig
    
    def create_fidelity_meter_2d(self, current_fidelity, target_fidelity=1.0):
        """Create dynamic fidelity meter showing recovery progress"""
        
        fig = go.Figure()
        
        # Create gauge-style meter
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=current_fidelity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Recovery Fidelity"},
            delta={'reference': target_fidelity},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "#00d4ff"},
                'steps': [
                    {'range': [0, 0.5], 'color': "#ff4444"},
                    {'range': [0.5, 0.8], 'color': "#ff9800"},
                    {'range': [0.8, 1], 'color': "#4caf50"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.95
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"}
        )
        
        return fig

# Global instance
enhanced_viz = Enhanced3DVisualizer()