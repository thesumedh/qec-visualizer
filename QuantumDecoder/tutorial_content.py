"""
Interactive tutorial content for the QEC Visualizer.
Contains guided lessons and demonstrations for CQHack25.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Any


class QECTutorial:
    """Interactive tutorial system for quantum error correction education."""
    
    def __init__(self):
        self.lessons = {
            "basics": {
                "title": "🎓 QEC Basics",
                "description": "Learn the fundamentals of quantum error correction",
                "steps": [
                    {
                        "title": "What is Quantum Error Correction?",
                        "content": """
                        Quantum computers are fragile! Unlike classical bits that are either 0 or 1,
                        quantum bits (qubits) can exist in superposition and are easily disturbed by:
                        
                        • **Decoherence**: Environment interaction destroys quantum states
                        • **Gate errors**: Imperfect quantum operations
                        • **Measurement errors**: Faulty readout of quantum states
                        
                        QEC codes protect quantum information by encoding logical qubits 
                        into multiple physical qubits with redundancy.
                        """,
                        "action": "Initialize a 3-qubit logical |0⟩ state to see encoding in action!"
                    },
                    {
                        "title": "The 3-Qubit Bit Flip Code",
                        "content": """
                        The simplest QEC code protects against bit flip (X) errors:
                        
                        • **Encoding**: |0⟩ → |000⟩, |1⟩ → |111⟩
                        • **Protection**: Can correct any single bit flip
                        • **Detection**: Use syndrome measurements to find errors
                        
                        **How it works**: If one qubit flips, majority voting recovers the original state.
                        """,
                        "action": "Apply a single X error and watch the syndrome detection!"
                    },
                    {
                        "title": "Syndrome Measurement",
                        "content": """
                        Syndromes are like "error fingerprints" that tell us what went wrong:
                        
                        • **S₁ = Z₁⊗Z₂**: Measures parity of qubits 0 and 1
                        • **S₂ = Z₂⊗Z₃**: Measures parity of qubits 1 and 2
                        
                        **Syndrome patterns**:
                        - 00: No error
                        - 10: Error on qubit 0  
                        - 11: Error on qubit 1
                        - 01: Error on qubit 2
                        """,
                        "action": "Try different error patterns and observe the syndromes!"
                    }
                ]
            },
            "advanced": {
                "title": "🚀 Advanced QEC",
                "description": "Explore the powerful 5-qubit code",
                "steps": [
                    {
                        "title": "The 5-Qubit Perfect Code",
                        "content": """
                        The 5-qubit code is the smallest code that can correct arbitrary single-qubit errors:
                        
                        • **Universal protection**: Corrects X, Y, and Z errors
                        • **Optimal**: Uses minimum number of qubits for distance-3 code
                        • **Perfect**: No information leakage during error correction
                        
                        This code uses 4 stabilizer generators to detect all single-qubit errors.
                        """,
                        "action": "Switch to 5-qubit code and try different error types!"
                    },
                    {
                        "title": "Stabilizer Codes",
                        "content": """
                        Stabilizer codes use group theory to systematically construct QEC codes:
                        
                        • **Stabilizers**: Operators that leave the code space unchanged
                        • **Syndrome extraction**: Measure stabilizer eigenvalues
                        • **Error correction**: Use syndrome lookup table
                        
                        The 5-qubit code has 4 stabilizers generating a 16-element stabilizer group.
                        """,
                        "action": "Apply mixed X/Z errors and see how the code handles them!"
                    }
                ]
            },
            "classiq": {
                "title": "⚡ Classiq Integration",
                "description": "Learn how Classiq powers real quantum circuits",
                "steps": [
                    {
                        "title": "Circuit Generation with Classiq",
                        "content": """
                        Classiq automatically generates optimized quantum circuits:
                        
                        • **Encoding circuits**: Transform logical states to physical qubits
                        • **Syndrome circuits**: Extract error information
                        • **Decoder circuits**: Apply corrections based on syndromes
                        
                        This enables real quantum hardware implementation!
                        """,
                        "action": "Check the Classiq Circuits tab to see generated QASM code!"
                    },
                    {
                        "title": "Hardware Considerations",
                        "content": """
                        Real quantum devices have constraints:
                        
                        • **Connectivity**: Not all qubits can interact directly
                        • **Gate fidelities**: Each operation has error rates
                        • **Coherence times**: Qubits lose information over time
                        
                        Classiq optimizes circuits for specific hardware backends.
                        """,
                        "action": "Explore the hardware requirements in the circuit info!"
                    }
                ]
            }
        }
    
    def render_tutorial_selector(self) -> str:
        """Render tutorial selection interface."""
        st.subheader("🎓 Interactive QEC Tutorial")
        
        tutorial_options = {
            "basics": "🎓 QEC Basics - Start here!",
            "advanced": "🚀 Advanced QEC - 5-qubit code",
            "classiq": "⚡ Classiq Integration - Real circuits"
        }
        
        selected = st.selectbox(
            "Choose your learning path:",
            list(tutorial_options.keys()),
            format_func=lambda x: tutorial_options[x],
            key="tutorial_selector"
        )
        
        return selected
    
    def render_lesson(self, lesson_key: str):
        """Render a specific tutorial lesson."""
        if lesson_key not in self.lessons:
            st.error("Tutorial not found!")
            return
        
        lesson = self.lessons[lesson_key]
        
        st.markdown(f"## {lesson['title']}")
        st.markdown(f"*{lesson['description']}*")
        
        # Progress tracking
        if f"tutorial_step_{lesson_key}" not in st.session_state:
            st.session_state[f"tutorial_step_{lesson_key}"] = 0
        
        current_step = st.session_state[f"tutorial_step_{lesson_key}"]
        total_steps = len(lesson['steps'])
        
        # Progress bar
        progress = (current_step + 1) / total_steps
        st.progress(progress, text=f"Step {current_step + 1} of {total_steps}")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_step > 0:
                if st.button("⬅️ Previous"):
                    st.session_state[f"tutorial_step_{lesson_key}"] -= 1
                    st.rerun()
        
        with col3:
            if current_step < total_steps - 1:
                if st.button("Next ➡️"):
                    st.session_state[f"tutorial_step_{lesson_key}"] += 1
                    st.rerun()
            elif current_step == total_steps - 1:
                if st.button("🏆 Complete!"):
                    st.balloons()
                    st.success("Tutorial completed! You're now a QEC expert!")
        
        # Current step content
        step = lesson['steps'][current_step]
        
        st.markdown(f"### {step['title']}")
        st.markdown(step['content'])
        
        # Action prompt
        if 'action' in step:
            st.info(f"💡 **Try this**: {step['action']}")
        
        # Quick facts for each lesson type
        if lesson_key == "basics" and current_step == 0:
            self._render_qec_quick_facts()
        elif lesson_key == "advanced" and current_step == 0:
            self._render_five_qubit_facts()
        elif lesson_key == "classiq":
            self._render_classiq_facts()
    
    def _render_qec_quick_facts(self):
        """Render quick facts about QEC basics."""
        st.markdown("#### 📊 Quick Facts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Classical Error Rate", "~10⁻¹⁷", help="Error rate in classical computers")
            st.metric("Quantum Error Rate", "~10⁻³", help="Typical error rate in quantum computers")
        
        with col2:
            st.metric("3-Qubit Code Distance", "3", help="Can correct 1 error, detect 2")
            st.metric("Physical/Logical Ratio", "3:1", help="3 physical qubits encode 1 logical")
    
    def _render_five_qubit_facts(self):
        """Render quick facts about the 5-qubit code."""
        st.markdown("#### 🚀 5-Qubit Code Facts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Code Parameters", "[[5,1,3]]", help="5 physical, 1 logical, distance 3")
            st.metric("Stabilizer Generators", "4", help="Number of independent stabilizers")
        
        with col2:
            st.metric("Error Patterns", "16", help="Total detectable error syndromes")
            st.metric("Correctable Errors", "15", help="Single-qubit error patterns")
    
    def _render_classiq_facts(self):
        """Render quick facts about Classiq integration."""
        st.markdown("#### ⚡ Classiq Integration Facts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Circuit Optimization", "Active", help="Automatic gate count reduction")
            st.metric("Hardware Backends", "3+", help="IBM, IonQ, Rigetti support")
        
        with col2:
            st.metric("QASM Generation", "Real-time", help="Live quantum assembly code")
            st.metric("Transpilation", "Automatic", help="Hardware-specific optimization")

    def get_tutorial_completion_status(self) -> Dict[str, bool]:
        """Get completion status for all tutorials."""
        status = {}
        for lesson_key, lesson in self.lessons.items():
            step_key = f"tutorial_step_{lesson_key}"
            if step_key in st.session_state:
                current_step = st.session_state[step_key]
                total_steps = len(lesson['steps'])
                status[lesson_key] = current_step >= total_steps - 1
            else:
                status[lesson_key] = False
        return status