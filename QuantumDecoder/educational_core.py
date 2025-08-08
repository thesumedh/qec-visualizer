"""
Educational Core - Making Quantum Error Correction Accessible
The heart of beginner-friendly quantum education
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

class QuantumEducator:
    """Makes quantum concepts accessible to everyone"""
    
    def create_quantum_analogy_section(self):
        """Explain quantum concepts using everyday analogies"""
        
        st.markdown("""
        ## 🌟 Why Quantum Computers Need Error Correction
        
        ### 🎭 The Theater Analogy
        
        Imagine you're directing a play with **very sensitive actors**:
        
        **Classical Computer (Regular Theater):**
        - Actors are professionals - they remember their lines perfectly
        - If someone forgets a line, they just look at the script
        - Very reliable, but limited in what they can perform
        
        **Quantum Computer (Quantum Theater):**
        - Actors are **incredibly talented** but **extremely sensitive**
        - They can perform **impossible feats** (like being in two places at once!)
        - But a cough from the audience can make them forget everything
        - **That's where Quantum Error Correction comes in!**
        """)
        
        # Interactive analogy visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Classical Theater")
            classical_fig = go.Figure()
            classical_fig.add_trace(go.Bar(
                x=['Actor 1', 'Actor 2', 'Actor 3'],
                y=[1, 1, 1],
                marker_color='lightblue',
                name='Reliable Performance'
            ))
            classical_fig.update_layout(
                title="Stable but Limited",
                yaxis_title="Performance Quality",
                height=300
            )
            st.plotly_chart(classical_fig, use_container_width=True)
            st.caption("✅ Reliable but can only do basic performances")
        
        with col2:
            st.markdown("### ⚛️ Quantum Theater")
            quantum_fig = go.Figure()
            quantum_fig.add_trace(go.Bar(
                x=['Qubit 1', 'Qubit 2', 'Qubit 3'],
                y=[0.8, 0.3, 0.9],  # Showing fragility
                marker_color=['lightgreen', 'lightcoral', 'lightgreen'],
                name='Fragile but Powerful'
            ))
            quantum_fig.update_layout(
                title="Powerful but Fragile",
                yaxis_title="Performance Quality", 
                height=300
            )
            st.plotly_chart(quantum_fig, use_container_width=True)
            st.caption("⚡ Amazing capabilities but needs protection!")
    
    def create_qec_story_progression(self):
        """Tell the QEC story in a compelling way"""
        
        st.markdown("""
        ## 📚 The Quantum Error Correction Story
        
        ### Chapter 1: The Problem 😰
        """)
        
        problem_col1, problem_col2 = st.columns(2)
        
        with problem_col1:
            st.markdown("""
            **The Quantum Dilemma:**
            - Quantum computers can solve impossible problems
            - But they're incredibly fragile
            - Any tiny disturbance destroys the computation
            - Like trying to balance a pencil on its tip in a hurricane!
            """)
        
        with problem_col2:
            # Visualization of quantum fragility
            fragility_fig = go.Figure()
            time_steps = list(range(10))
            perfect_info = [1.0] * 10
            degraded_info = [1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
            
            fragility_fig.add_trace(go.Scatter(
                x=time_steps, y=perfect_info,
                name='What We Want', line=dict(color='green', width=3)
            ))
            fragility_fig.add_trace(go.Scatter(
                x=time_steps, y=degraded_info,
                name='What Actually Happens', line=dict(color='red', width=3)
            ))
            fragility_fig.update_layout(
                title="Quantum Information Decay",
                xaxis_title="Time",
                yaxis_title="Information Quality",
                height=300
            )
            st.plotly_chart(fragility_fig, use_container_width=True)
        
        st.markdown("""
        ### Chapter 2: The Breakthrough 💡
        
        **Scientists discovered something amazing:**
        - We can't prevent quantum errors...
        - **But we can detect and fix them!**
        - The secret: Use multiple qubits to protect one piece of information
        - Like having backup singers who can cover if the lead singer forgets the words!
        """)
        
        # Show the breakthrough concept
        breakthrough_col1, breakthrough_col2, breakthrough_col3 = st.columns(3)
        
        with breakthrough_col1:
            st.markdown("#### 🎤 Solo Performance")
            solo_fig = go.Figure(data=go.Bar(x=['Singer'], y=[1], marker_color='lightcoral'))
            solo_fig.update_layout(title="Risky!", height=200)
            st.plotly_chart(solo_fig, use_container_width=True)
            st.caption("❌ If they mess up, show is ruined")
        
        with breakthrough_col2:
            st.markdown("#### 🎵 With Backup")
            backup_fig = go.Figure(data=go.Bar(
                x=['Lead', 'Backup 1', 'Backup 2'], 
                y=[1, 1, 1], 
                marker_color='lightgreen'
            ))
            backup_fig.update_layout(title="Protected!", height=200)
            st.plotly_chart(backup_fig, use_container_width=True)
            st.caption("✅ If one messes up, others can cover")
        
        with breakthrough_col3:
            st.markdown("#### ⚛️ Quantum Version")
            quantum_backup_fig = go.Figure(data=go.Bar(
                x=['Q0', 'Q1', 'Q2'], 
                y=[1, 1, 1], 
                marker_color='lightblue'
            ))
            quantum_backup_fig.update_layout(title="QEC Magic!", height=200)
            st.plotly_chart(quantum_backup_fig, use_container_width=True)
            st.caption("🎯 3 qubits protect 1 logical qubit")
        
        st.markdown("""
        ### Chapter 3: How It Works 🔧
        
        **The 4-Step QEC Dance:**
        """)
        
        steps_col1, steps_col2 = st.columns(2)
        
        with steps_col1:
            st.markdown("""
            **1. 🛡️ Encoding (Protection)**
            - Take your precious quantum information
            - Spread it across multiple qubits
            - Like making multiple copies of a treasure map
            
            **2. 💥 Error Happens**
            - Environment attacks one qubit
            - But the information is still safe in the others!
            - Like one backup singer getting a cold
            """)
        
        with steps_col2:
            st.markdown("""
            **3. 🔍 Detection (Detective Work)**
            - Check which qubit got corrupted
            - Use clever measurements called "syndromes"
            - Like listening to find which singer is off-key
            
            **4. ✨ Correction (The Fix)**
            - Apply the right fix to the corrupted qubit
            - Restore the original information
            - Like the conductor getting everyone back in harmony
            """)
    
    def create_interactive_learning_path(self):
        """Create an interactive learning progression"""
        
        st.markdown("## 🎯 Your Learning Journey")
        
        # Learning path selector
        learning_level = st.selectbox(
            "Choose your starting point:",
            [
                "🌱 Complete Beginner - What is quantum?",
                "🎓 Student - I know some physics", 
                "💻 Developer - I code but new to quantum",
                "🔬 Researcher - I want the technical details"
            ]
        )
        
        if "Complete Beginner" in learning_level:
            st.markdown("""
            ### 🌱 Perfect! Let's start from the very beginning.
            
            **What makes quantum special?**
            - Regular bits are like coins: heads (0) or tails (1)
            - Quantum bits (qubits) are like **spinning coins**
            - They can be heads, tails, or **both at the same time!**
            - This "both at once" is called **superposition**
            
            **Why is this useful?**
            - A regular computer with 3 bits can be in 1 state: like 101
            - A quantum computer with 3 qubits can be in **all 8 states simultaneously!**
            - This is why quantum computers can solve certain problems exponentially faster
            
            **The catch?**
            - This superposition is incredibly fragile
            - Any disturbance collapses it back to just one state
            - That's why we need error correction!
            """)
            
        elif "Student" in learning_level:
            st.markdown("""
            ### 🎓 Great! You have the physics background.
            
            **Quantum Error Correction Fundamentals:**
            - Quantum states are described by complex amplitudes
            - Decoherence destroys quantum coherence
            - QEC uses redundancy in Hilbert space
            - Stabilizer codes detect errors without measuring the logical state
            
            **Key Insight:**
            - We can't clone quantum states (no-cloning theorem)
            - But we can encode them in entangled subspaces
            - Error syndromes give us classical information about quantum errors
            """)
            
        elif "Developer" in learning_level:
            st.markdown("""
            ### 💻 Excellent! Let's connect this to programming concepts.
            
            **Think of QEC like:**
            - **RAID for quantum data** - redundancy protects against failures
            - **Error handling in distributed systems** - detect and recover from failures
            - **Checksums for quantum information** - detect corruption without reading the data
            
            **The Classiq Connection:**
            - Classiq SDK abstracts the low-level quantum circuit details
            - You write high-level quantum functions (@qfunc)
            - Classiq optimizes and compiles to real hardware
            - Perfect for implementing QEC algorithms at scale
            """)
            
        else:  # Researcher
            st.markdown("""
            ### 🔬 Perfect! Let's dive into the technical details.
            
            **Advanced QEC Theory:**
            - Stabilizer formalism and Pauli group structure
            - Distance-3 codes can correct single-qubit errors
            - Surface codes approach the threshold theorem
            - ML decoders can outperform classical MWPM
            
            **This Implementation:**
            - Real stabilizer measurements and syndrome extraction
            - Realistic noise models (T1, T2, gate errors)
            - Neural network decoders with explainable AI
            - Classiq SDK integration for hardware deployment
            """)
    
    def create_motivation_section(self):
        """Create compelling motivation for learning QEC"""
        
        st.markdown("""
        ## 🚀 Why This Matters for YOUR Future
        
        ### The Quantum Revolution is Happening NOW
        """)
        
        motivation_tabs = st.tabs(["🌍 Global Impact", "💼 Career Opportunities", "🧠 Intellectual Challenge"])
        
        with motivation_tabs[0]:
            st.markdown("""
            **Quantum computing is solving real problems:**
            
            🧬 **Drug Discovery**: Finding new medicines 1000x faster
            - COVID-19 vaccine development accelerated by quantum simulations
            - Personalized medicine based on quantum molecular modeling
            
            🌱 **Climate Change**: Optimizing renewable energy
            - Better solar panels through quantum material design
            - Efficient carbon capture using quantum chemistry
            
            🔐 **Cybersecurity**: Protecting our digital world
            - Quantum-safe encryption for the post-quantum era
            - Secure quantum communication networks
            
            💰 **Finance**: Revolutionizing risk analysis
            - Portfolio optimization with quantum algorithms
            - Real-time fraud detection using quantum ML
            """)
            
        with motivation_tabs[1]:
            st.markdown("""
            **Quantum careers are exploding:**
            
            📈 **Job Growth**: 500% increase in quantum job postings (2020-2024)
            
            💰 **Salaries**: $120k-$300k+ for quantum engineers
            
            🏢 **Companies Hiring**:
            - **Tech Giants**: Google, IBM, Microsoft, Amazon
            - **Startups**: IonQ, Rigetti, Xanadu, PsiQuantum
            - **Finance**: Goldman Sachs, JPMorgan, Wells Fargo
            - **Pharma**: Roche, Merck, Johnson & Johnson
            
            🎯 **Skills in Demand**:
            - Quantum error correction (that's what you're learning!)
            - Quantum algorithms and programming
            - Quantum hardware and control systems
            """)
            
        with motivation_tabs[2]:
            st.markdown("""
            **The intellectual thrill:**
            
            🤯 **Mind-Bending Physics**: 
            - Particles that exist in multiple states simultaneously
            - Information that can be teleported instantly
            - Computers that harness the fabric of reality itself
            
            🧩 **Beautiful Mathematics**:
            - Linear algebra in complex vector spaces
            - Group theory and symmetries
            - Information theory meets quantum mechanics
            
            🔬 **Cutting-Edge Research**:
            - You're learning what Nobel Prize winners discovered
            - Contributing to humanity's greatest technological leap
            - Joining the ranks of quantum pioneers
            """)
    
    def create_reference_sites_section(self):
        """Create comprehensive reference sites for learning"""
        
        st.markdown("""
        ## 📚 Essential Learning Resources
        
        **Your complete guide to mastering quantum error correction!**
        """)
        
        # Learning level selector
        selected_level = st.selectbox(
            "Choose your learning level for personalized recommendations:",
            ["beginner", "intermediate", "advanced"]
        )
        
        # Get personalized resources
        path_resources = get_learning_path_resources(selected_level)
        
        st.success(f"🎯 **{selected_level.title()} Path**: {path_resources['description']}")
        st.info(f"📈 **Next Steps**: {path_resources['next_steps']}")
        
        # Display all resource categories
        for category_key, category_data in REFERENCE_SITES.items():
            with st.expander(f"{category_data['title']} ({len(category_data['sites'])} resources)"):
                for site in category_data['sites']:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**[{site['name']}]({site['url']})**")
                        st.write(site['description'])
                        
                        # Highlight key features
                        highlights_text = " • ".join(site['highlights'])
                        st.caption(f"✨ {highlights_text}")
                    
                    with col2:
                        # Level indicator
                        level_color = {
                            "Beginner": "🟢",
                            "Intermediate": "🟡", 
                            "Advanced": "🔴",
                            "All levels": "🔵",
                            "Professional": "🟣"
                        }
                        
                        level_key = site['level']
                        for key in level_color.keys():
                            if key in level_key:
                                st.markdown(f"{level_color[key]} {level_key}")
                                break
                    
                    st.markdown("---")
        
        # Career-focused resources
        st.markdown("## 💼 Career Development Resources")
        
        career_path = st.selectbox(
            "Choose your career focus:",
            ["academic_path", "industry_path", "startup_path"]
        )
        
        career_resources = get_career_focused_resources()[career_path]
        
        st.info(f"🎯 **{career_path.replace('_', ' ').title()}**: {career_resources['description']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 Recommended Resources:**")
            for resource in career_resources['resources']:
                st.write(f"• {resource}")
        
        with col2:
            st.markdown("**🛠️ Key Skills to Develop:**")
            for skill in career_resources['skills']:
                st.write(f"• {skill}")
        
        # Quick access links
        st.markdown("## ⚡ Quick Access Links")
        
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            st.markdown("""
            **🚀 Start Coding Now:**
            - [IBM Qiskit](https://qiskit.org/)
            - [Google Cirq](https://quantumai.google/cirq)
            - [Classiq Platform](https://www.classiq.io/)
            """)
        
        with quick_col2:
            st.markdown("""
            **📖 Learn Theory:**
            - [Qiskit Textbook](https://qiskit.org/textbook/)
            - [Microsoft Quantum Katas](https://github.com/Microsoft/QuantumKatas)
            - [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/)
            """)
        
        with quick_col3:
            st.markdown("""
            **💼 Find Jobs:**
            - [Quantum Jobs](https://quantumjobs.net/)
            - [IBM Quantum Network](https://quantum-network.org/)
            - [IEEE Quantum](https://quantum.ieee.org/)
            """)
        
        # Study plan generator
        st.markdown("## 📅 Personalized Study Plan")
        
        study_duration = st.selectbox(
            "How much time can you dedicate per week?",
            ["2-3 hours (casual)", "5-7 hours (focused)", "10+ hours (intensive)"]
        )
        
        study_goal = st.selectbox(
            "What's your main goal?",
            ["Understand the basics", "Get a quantum job", "Contribute to research", "Start a quantum company"]
        )
        
        if st.button("🎯 Generate My Study Plan"):
            st.success("📋 **Your Personalized Quantum Error Correction Study Plan:**")
            
            if "casual" in study_duration:
                st.markdown("""
                **Week 1-2**: Start with IBM Qiskit Textbook basics
                **Week 3-4**: Try Microsoft Quantum Katas exercises  
                **Week 5-6**: Experiment with this QEC Visualizer
                **Week 7-8**: Join Quantum Computing Stack Exchange community
                """)
            elif "focused" in study_duration:
                st.markdown("""
                **Week 1**: Master quantum basics with Qiskit Textbook
                **Week 2**: Complete Quantum Katas error correction module
                **Week 3**: Deep dive into Surface Code paper
                **Week 4**: Implement QEC codes in Qiskit
                **Week 5**: Explore ML decoders and neural networks
                **Week 6**: Build your own QEC project
                """)
            else:  # intensive
                st.markdown("""
                **Week 1**: Complete quantum fundamentals
                **Week 2**: Master stabilizer formalism
                **Week 3**: Implement multiple QEC codes
                **Week 4**: Study advanced papers and ML techniques
                **Week 5**: Contribute to open-source quantum projects
                **Week 6**: Start networking with quantum professionals
                """)
            
            st.balloons()

    def create_personalized_study_plan(self):
        """Create personalized study plan with PDF download"""
        
        st.markdown("""
        ## 📅 Get Your Personalized Study Plan
        
        **Tell us about yourself and get a custom learning roadmap!**
        """)
        
        # Simple level assessment
        col1, col2 = st.columns(2)
        
        with col1:
            experience_level = st.selectbox(
                "What's your quantum experience?",
                [
                    "🌱 Complete beginner - never heard of qubits",
                    "🎓 Student - know some physics/math", 
                    "💻 Developer - code but new to quantum",
                    "🔬 Researcher - want advanced techniques"
                ]
            )
        
        with col2:
            time_commitment = st.selectbox(
                "How much time per week?",
                ["2-3 hours", "5-7 hours", "10+ hours"]
            )
        
        goal = st.selectbox(
            "What's your main goal?",
            [
                "🎯 Understand how quantum error correction works",
                "💼 Get a job in quantum computing", 
                "🔬 Do quantum research",
                "🚀 Start a quantum company"
            ]
        )
        
        if st.button("📅 **Generate My Study Plan**", type="primary"):
            # Generate study plan based on inputs
            plan_data = self._generate_study_plan(experience_level, time_commitment, goal)
            
            st.success("🎉 **Your Personalized Quantum Error Correction Study Plan:**")
            
            # Display the plan
            for week, activities in plan_data['weekly_plan'].items():
                with st.expander(f"📅 {week}"):
                    for activity in activities:
                        st.write(f"• {activity}")
            
            # Key resources
            st.markdown("### 🔗 Essential Links for Your Level:")
            for resource in plan_data['key_resources']:
                st.markdown(f"- **[{resource['name']}]({resource['url']})** - {resource['description']}")
            
            # Generate TXT content
            txt_content = self._create_txt_content(plan_data, experience_level, time_commitment, goal)
            
            # Download button
            st.download_button(
                label="📥 **Download Study Plan**",
                data=txt_content,
                file_name=f"quantum_study_plan_{experience_level.split()[0].lower()}.txt",
                mime="text/plain",
                type="primary"
            )
            
            st.balloons()
    
    def _generate_study_plan(self, level, time, goal):
        """Generate study plan based on user inputs"""
        
        if "beginner" in level.lower():
            return {
                'weekly_plan': {
                    'Week 1': ['Watch "Quantum Computing Explained" videos', 'Try this QEC Visualizer with 3-qubit code', 'Read: What is a qubit?'],
                    'Week 2': ['IBM Qiskit Textbook - Chapters 1-2', 'Practice with quantum gates', 'Join r/QuantumComputing'],
                    'Week 3': ['Learn about quantum errors', 'Try Surface Code in this tool', 'Read about Google\'s quantum computer'],
                    'Week 4': ['Microsoft Quantum Katas - basics', 'Export QASM from this tool', 'Connect with quantum community']
                },
                'key_resources': [
                    {'name': 'IBM Qiskit Textbook', 'url': 'https://qiskit.org/textbook/', 'description': 'Free interactive quantum textbook'},
                    {'name': 'This QEC Visualizer', 'url': '#', 'description': 'Start with 3-qubit code, then try Surface Code'},
                    {'name': 'Quantum Computing Subreddit', 'url': 'https://reddit.com/r/QuantumComputing', 'description': 'Beginner-friendly community'}
                ]
            }
        elif "student" in level.lower():
            return {
                'weekly_plan': {
                    'Week 1': ['Review linear algebra basics', 'Master this QEC tool completely', 'Read Nielsen & Chuang Ch 10'],
                    'Week 2': ['Implement 3-qubit code in Qiskit', 'Study stabilizer formalism', 'Try ML decoder in this tool'],
                    'Week 3': ['Surface code deep dive', 'Read Google\'s quantum supremacy paper', 'Practice QASM programming'],
                    'Week 4': ['Build your own QEC project', 'Join IBM Quantum Network', 'Start following quantum researchers']
                },
                'key_resources': [
                    {'name': 'Nielsen & Chuang Textbook', 'url': 'https://www.cambridge.org/core/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE', 'description': 'The quantum computing bible'},
                    {'name': 'IBM Quantum Network', 'url': 'https://quantum-network.org/', 'description': 'Access real quantum computers'},
                    {'name': 'arXiv Quantum Physics', 'url': 'https://arxiv.org/list/quant-ph/recent', 'description': 'Latest research papers'}
                ]
            }
        else:  # Advanced
            return {
                'weekly_plan': {
                    'Week 1': ['Implement MWPM decoder', 'Study this tool\'s ML implementation', 'Read latest QEC papers'],
                    'Week 2': ['Contribute to open source quantum projects', 'Network with quantum professionals', 'Apply to quantum companies'],
                    'Week 3': ['Start your own quantum research', 'Publish on quantum topics', 'Mentor others in quantum'],
                    'Week 4': ['Advanced: Topological codes', 'Quantum LDPC codes', 'Industry collaboration']
                },
                'key_resources': [
                    {'name': 'Quantum Jobs', 'url': 'https://quantumjobs.net/', 'description': 'Find quantum computing positions'},
                    {'name': 'PennyLane', 'url': 'https://pennylane.ai/', 'description': 'Quantum machine learning'},
                    {'name': 'Classiq Platform', 'url': 'https://www.classiq.io/', 'description': 'Professional quantum development'}
                ]
            }
    
    def _create_txt_content(self, plan_data, level, time, goal):
        """Create TXT content for study plan"""
        
        import datetime
        txt_lines = [
            "=" * 60,
            "YOUR PERSONALIZED QUANTUM ERROR CORRECTION STUDY PLAN",
            "=" * 60,
            "",
            f"Experience Level: {level}",
            f"Time Commitment: {time} per week", 
            f"Goal: {goal}",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}",
            "",
            "YOUR LEARNING PATH:",
            "-" * 30
        ]
        
        for week, activities in plan_data['weekly_plan'].items():
            txt_lines.append(f"\n{week}:")
            for activity in activities:
                txt_lines.append(f"  - {activity}")
        
        txt_lines.extend([
            "",
            "ESSENTIAL RESOURCES:",
            "-" * 30
        ])
        
        for resource in plan_data['key_resources']:
            txt_lines.append(f"- {resource['name']}: {resource['description']}")
            txt_lines.append(f"  Link: {resource['url']}")
            txt_lines.append("")
        
        txt_lines.extend([
            "QUICK START CHECKLIST:",
            "-" * 30,
            "[ ] Bookmark this QEC Visualizer tool",
            "[ ] Join quantum computing community", 
            "[ ] Set up development environment",
            "[ ] Follow quantum researchers",
            "[ ] Practice daily with quantum concepts",
            "",
            "SUCCESS TIPS:",
            "- Start small, build momentum",
            "- Join communities for support", 
            "- Practice with real tools",
            "- Don't get overwhelmed",
            "- Celebrate small wins",
            "",
            "Generated by QEC Visualizer - Your Quantum Learning Companion"
        ])
        
        return "\n".join(txt_lines)

# Global educator instance
quantum_educator = QuantumEducator()
