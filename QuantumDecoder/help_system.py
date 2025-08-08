"""
Simple Help System for QEC Visualizer
Provides contextual help and explanations to make QEC concepts accessible
"""

import streamlit as st
from typing import Dict, List

class SimpleHelpSystem:
    """Easy-to-use help system with tooltips and explanations"""
    
    def __init__(self):
        self.help_content = self._load_help_content()
        self.glossary = self._load_glossary()
    
    def _load_help_content(self) -> Dict[str, Dict[str, str]]:
        """Load contextual help content"""
        return {
            "qec_codes": {
                "3-qubit": {
                    "title": "3-Qubit Repetition Code",
                    "description": "The simplest quantum error correction code that protects against bit-flip errors.",
                    "how_it_works": "Encodes one logical qubit into three physical qubits: |0⟩ → |000⟩ and |1⟩ → |111⟩. Uses majority voting to detect and correct single bit-flip errors.",
                    "best_for": "Learning basic QEC concepts and understanding redundancy in quantum information."
                },
                "5-qubit": {
                    "title": "5-Qubit Perfect Code",
                    "description": "The smallest quantum code that can correct any single-qubit error (X, Y, or Z).",
                    "how_it_works": "Uses 4 stabilizer measurements to detect and locate any single-qubit error. Optimal in the sense that it uses the minimum number of qubits for universal single-error correction.",
                    "best_for": "Understanding stabilizer codes and universal error correction."
                },
                "steane": {
                    "title": "Steane 7-Qubit CSS Code",
                    "description": "The first quantum code capable of correcting both bit-flip and phase-flip errors.",
                    "how_it_works": "Uses classical Hamming code structure for both X and Z errors. Has 6 stabilizers (3 X-type, 3 Z-type) and supports transversal gates.",
                    "best_for": "Learning CSS codes and fault-tolerant quantum computing concepts."
                },
                "surface": {
                    "title": "Surface Code (Industry Standard)",
                    "description": "The leading candidate for fault-tolerant quantum computing, used by Google and IBM.",
                    "how_it_works": "Arranges qubits on a 2D lattice with local parity checks. Uses minimum weight perfect matching for decoding. Scales well to larger distances.",
                    "best_for": "Understanding real-world quantum error correction and topological codes."
                }
            },
            "concepts": {
                "syndrome": {
                    "title": "Error Syndrome",
                    "description": "Information about errors without revealing the quantum state itself.",
                    "why_important": "Syndrome measurement is the key insight of QEC - we can detect errors without destroying the quantum information we're trying to protect.",
                    "analogy": "Like checking if a book has typos by looking at checksums, without reading the actual content."
                },
                "stabilizers": {
                    "title": "Stabilizer Measurements",
                    "description": "Special measurements that detect errors without affecting the logical information.",
                    "why_important": "Stabilizers are the 'guardians' of your quantum information - they watch for errors while leaving your data untouched.",
                    "analogy": "Like security cameras that can detect intruders without disturbing the people inside."
                },
                "logical_qubits": {
                    "title": "Logical vs Physical Qubits",
                    "description": "Logical qubits store your actual information, while physical qubits provide protection.",
                    "why_important": "This redundancy is what makes quantum error correction possible - many physical qubits work together to protect one logical qubit.",
                    "analogy": "Like having multiple backup copies of an important document stored in different locations."
                }
            },
            "process": {
                "encoding": {
                    "title": "Quantum Error Correction Process",
                    "description": "The 4-step process that protects quantum information from errors.",
                    "steps": [
                        "🔵 **Encode**: Transform logical qubit into protected multi-qubit state",
                        "⚡ **Error**: Quantum noise corrupts some physical qubits", 
                        "🎯 **Detect**: Measure syndrome to locate errors without destroying data",
                        "🛠️ **Correct**: Apply corrective operations to restore original state"
                    ],
                    "key_insight": "The magic is in step 3 - we can detect errors without measuring the logical qubit directly!"
                }
            }
        }
    
    def _load_glossary(self) -> Dict[str, str]:
        """Load technical term definitions"""
        return {
            "Qubit": "The basic unit of quantum information, like a classical bit but can be in superposition of 0 and 1.",
            "Superposition": "A quantum state that is a combination of multiple classical states simultaneously.",
            "Entanglement": "A quantum phenomenon where qubits become correlated in ways that have no classical analog.",
            "Decoherence": "The process by which quantum systems lose their quantum properties due to interaction with the environment.",
            "Pauli Gates": "Basic quantum gates: X (bit-flip), Y (bit+phase flip), Z (phase-flip).",
            "Syndrome": "Classical information that reveals the presence and location of errors without disturbing the quantum state.",
            "Stabilizer": "A quantum measurement that can detect errors while preserving the logical information.",
            "CSS Code": "Calderbank-Shor-Steane codes that treat X and Z errors separately using classical codes.",
            "Distance": "The minimum number of errors needed to cause a logical error - higher distance means better protection.",
            "Threshold": "The error rate below which quantum error correction provides a net benefit.",
            "Fidelity": "A measure of how close two quantum states are - 1.0 means identical, 0.0 means completely different.",
            "QASM": "Quantum Assembly Language - a standard format for describing quantum circuits."
        }
    
    def show_code_help(self, code_type: str):
        """Show help for a specific QEC code"""
        if code_type in self.help_content["qec_codes"]:
            info = self.help_content["qec_codes"][code_type]
            
            with st.expander(f"❓ Help: {info['title']}", expanded=False):
                st.markdown(f"**What it is:** {info['description']}")
                st.markdown(f"**How it works:** {info['how_it_works']}")
                st.markdown(f"**Best for:** {info['best_for']}")
    
    def show_concept_help(self, concept: str):
        """Show help for a QEC concept"""
        if concept in self.help_content["concepts"]:
            info = self.help_content["concepts"][concept]
            
            with st.expander(f"💡 Understanding: {info['title']}", expanded=False):
                st.markdown(f"**What it is:** {info['description']}")
                if "why_important" in info:
                    st.markdown(f"**Why it matters:** {info['why_important']}")
                if "analogy" in info:
                    st.markdown(f"**Think of it like:** {info['analogy']}")
    
    def show_process_help(self):
        """Show help for the QEC process"""
        info = self.help_content["process"]["encoding"]
        
        with st.expander("🔄 How Quantum Error Correction Works", expanded=False):
            st.markdown(f"**Overview:** {info['description']}")
            st.markdown("**The 4 Steps:**")
            for step in info['steps']:
                st.markdown(f"- {step}")
            st.info(f"🔑 **Key Insight:** {info['key_insight']}")
    
    def show_glossary_term(self, term: str):
        """Show definition of a technical term"""
        if term in self.glossary:
            st.info(f"**{term}:** {self.glossary[term]}")
    
    def show_quick_help_panel(self):
        """Show a quick help panel with common questions"""
        with st.sidebar:
            with st.expander("❓ Quick Help", expanded=False):
                st.markdown("### 🚀 Getting Started")
                st.markdown("1. Choose a QEC code (start with 3-Qubit)")
                st.markdown("2. Click 'Initialize' to encode your qubit")
                st.markdown("3. Inject an error to see what happens")
                st.markdown("4. Measure syndrome to detect the error")
                st.markdown("5. Apply correction to fix it!")
                
                st.markdown("### 🎓 New to QEC?")
                st.markdown("- Start with the **Guided Tutorial** tab")
                st.markdown("- Use the **3-Qubit code** first")
                st.markdown("- Read the help tooltips (❓ icons)")
                
                st.markdown("### 🔧 Troubleshooting")
                st.markdown("- **Nothing happening?** Make sure to initialize first")
                st.markdown("- **Confused by syndrome?** It's just error location info")
                st.markdown("- **Want to start over?** Use the Reset button")
    
    def create_smart_tooltip(self, text: str, help_key: str) -> str:
        """Create a tooltip with contextual help"""
        if help_key in self.glossary:
            return f"{text} ❓"
        return text

# Global help system instance
help_system = SimpleHelpSystem()