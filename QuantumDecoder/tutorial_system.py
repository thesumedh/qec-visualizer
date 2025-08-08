"""
Simple Tutorial System for QEC Visualizer
Makes quantum error correction easy to understand step-by-step
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TutorialLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"

@dataclass
class TutorialStep:
    """Single step in a tutorial sequence"""
    title: str
    description: str
    instructions: str
    expected_action: str
    hints: List[str]
    success_message: str
    
class SimpleTutorialEngine:
    """Easy-to-use tutorial system for QEC learning"""
    
    def __init__(self):
        self.tutorials = self._load_tutorials()
        self.current_tutorial = None
        self.current_step = 0
        
    def _load_tutorials(self) -> Dict[str, List[TutorialStep]]:
        """Load all available tutorials"""
        return {
            "3-qubit-basics": [
                TutorialStep(
                    title="🎯 Step 1: Initialize Your First QEC Code",
                    description="Let's start with the simplest quantum error correction code - the 3-qubit repetition code.",
                    instructions="1. Make sure '🎓 3-Qubit (Start Here)' is selected\n2. Choose '|0⟩' as your initial state\n3. Click the '▶️ 1. Initialize' button",
                    expected_action="initialize",
                    hints=[
                        "The 3-qubit code protects one logical qubit using three physical qubits",
                        "It encodes |0⟩ as |000⟩ and |1⟩ as |111⟩",
                        "This redundancy lets us detect and fix single bit-flip errors"
                    ],
                    success_message="Great! You've encoded your logical |0⟩ into three physical qubits. Notice how the state is now |000⟩."
                ),
                TutorialStep(
                    title="⚡ Step 2: Inject a Quantum Error",
                    description="Now let's simulate what happens when quantum noise corrupts your information.",
                    instructions="1. Select 'X' as the error type (bit-flip error)\n2. Click the '▶️ 2. Inject Error' button\n3. Watch how one qubit gets flipped!",
                    expected_action="inject_error",
                    hints=[
                        "X errors flip qubits: |0⟩ → |1⟩ and |1⟩ → |0⟩",
                        "In real quantum computers, these happen due to electromagnetic noise",
                        "The error will randomly affect one of your three qubits"
                    ],
                    success_message="Perfect! An error has corrupted one of your qubits. But don't worry - QEC can fix this!"
                ),
                TutorialStep(
                    title="🎯 Step 3: Detect the Error (Syndrome)",
                    description="This is the magic of QEC - we can detect errors without destroying the quantum information!",
                    instructions="1. Click the '▶️ 3. Measure Syndrome' button\n2. Look at the syndrome result - it tells us which qubit has the error\n3. Notice the quantum state is still protected!",
                    expected_action="measure_syndrome",
                    hints=[
                        "Syndrome measurement uses ancilla qubits to check parity",
                        "Different syndrome patterns point to different error locations",
                        "The logical information remains intact during syndrome measurement"
                    ],
                    success_message="Excellent! The syndrome measurement detected the error location without destroying your data."
                ),
                TutorialStep(
                    title="🛠️ Step 4: Fix the Error",
                    description="Now we use the syndrome information to correct the error and recover the original state.",
                    instructions="1. Click the '▶️ 4. Apply Correction' button\n2. Watch as the decoder fixes the corrupted qubit\n3. Your original |000⟩ state is restored!",
                    expected_action="apply_correction",
                    hints=[
                        "The decoder uses a lookup table: syndrome → correction",
                        "For 3-qubit code: syndrome '10' means fix qubit 0, '11' means fix qubit 1, etc.",
                        "The correction applies an X gate to flip the corrupted qubit back"
                    ],
                    success_message="🎉 Congratulations! You've successfully completed your first quantum error correction cycle!"
                )
            ],
            
            "5-qubit-advanced": [
                TutorialStep(
                    title="🔬 Advanced: The Perfect 5-Qubit Code",
                    description="The 5-qubit code is the smallest code that can correct ANY single-qubit error (X, Y, or Z).",
                    instructions="1. Select '🔬 5-Qubit (Perfect Code)'\n2. Try different error types: X, Y, Z\n3. See how it handles all error types!",
                    expected_action="initialize",
                    hints=[
                        "Unlike 3-qubit code, this corrects both bit-flip AND phase-flip errors",
                        "It uses 4 stabilizer measurements instead of 2",
                        "This is the theoretical minimum for universal error correction"
                    ],
                    success_message="You're now working with a universal quantum error correction code!"
                )
            ],
            
            "steane-css": [
                TutorialStep(
                    title="⭐ CSS Codes: Steane's Breakthrough",
                    description="The Steane code combines classical error correction with quantum mechanics.",
                    instructions="1. Select '⭐ Steane 7-Qubit (CSS Code)'\n2. Notice it has 6 stabilizers (3 X-type, 3 Z-type)\n3. Try Y errors - see how CSS structure handles them!",
                    expected_action="initialize", 
                    hints=[
                        "CSS = Calderbank-Shor-Steane codes",
                        "They separate X and Z error correction",
                        "Support transversal gates for fault-tolerant computing"
                    ],
                    success_message="You're exploring the foundation of fault-tolerant quantum computing!"
                )
            ]
        }
    
    def start_tutorial(self, tutorial_name: str) -> bool:
        """Start a specific tutorial"""
        if tutorial_name in self.tutorials:
            self.current_tutorial = tutorial_name
            self.current_step = 0
            return True
        return False
    
    def get_current_step(self) -> Optional[TutorialStep]:
        """Get the current tutorial step"""
        if self.current_tutorial and self.current_step < len(self.tutorials[self.current_tutorial]):
            return self.tutorials[self.current_tutorial][self.current_step]
        return None
    
    def advance_step(self) -> bool:
        """Move to next tutorial step"""
        if self.current_tutorial:
            self.current_step += 1
            return self.current_step < len(self.tutorials[self.current_tutorial])
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current tutorial progress"""
        if not self.current_tutorial:
            return {"tutorial": None, "step": 0, "total": 0, "progress": 0}
            
        total_steps = len(self.tutorials[self.current_tutorial])
        return {
            "tutorial": self.current_tutorial,
            "step": self.current_step + 1,
            "total": total_steps,
            "progress": (self.current_step + 1) / total_steps
        }
    
    def show_tutorial_panel(self):
        """Display tutorial guidance panel in Streamlit"""
        if not self.current_tutorial:
            self._show_tutorial_selection()
        else:
            self._show_current_tutorial()
    
    def _show_tutorial_selection(self):
        """Show tutorial selection interface"""
        st.markdown("### 🎓 Choose Your Learning Path")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Beginner\n3-Qubit Basics", help="Perfect for first-time learners"):
                self.start_tutorial("3-qubit-basics")
                st.rerun()
                
        with col2:
            if st.button("🔬 Intermediate\n5-Qubit Perfect", help="Learn universal error correction"):
                self.start_tutorial("5-qubit-advanced") 
                st.rerun()
                
        with col3:
            if st.button("⭐ Advanced\nSteane CSS Code", help="Explore fault-tolerant codes"):
                self.start_tutorial("steane-css")
                st.rerun()
    
    def _show_current_tutorial(self):
        """Show current tutorial step"""
        step = self.get_current_step()
        progress = self.get_progress()
        
        if not step:
            st.success("🎉 Tutorial Complete! Try another one or explore freely.")
            if st.button("🔄 Choose New Tutorial"):
                self.current_tutorial = None
                self.current_step = 0
                st.rerun()
            return
        
        # Progress bar
        st.progress(progress["progress"])
        st.caption(f"Step {progress['step']} of {progress['total']}")
        
        # Current step content
        st.markdown(f"## {step.title}")
        st.info(step.description)
        
        # Instructions
        st.markdown("### 📋 What to do:")
        st.markdown(step.instructions)
        
        # Hints in expandable section
        with st.expander("💡 Need help? Click for hints"):
            for i, hint in enumerate(step.hints, 1):
                st.write(f"{i}. {hint}")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⏭️ Skip Step"):
                self.advance_step()
                st.rerun()
                
        with col2:
            if st.button("🔄 Restart Tutorial"):
                self.current_step = 0
                st.rerun()
                
        with col3:
            if st.button("❌ Exit Tutorial"):
                self.current_tutorial = None
                self.current_step = 0
                st.rerun()
    
    def check_action_completed(self, action: str) -> bool:
        """Check if expected action was completed"""
        step = self.get_current_step()
        if step and step.expected_action == action:
            # Show success message
            st.success(step.success_message)
            # Auto-advance after short delay
            if self.advance_step():
                st.rerun()
            return True
        return False

# Global tutorial engine instance
tutorial_engine = SimpleTutorialEngine()