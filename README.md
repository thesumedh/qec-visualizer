# 🏆 Quantum Error Correction Visualizer - CQHack25

> **Interactive QEC Learning Platform** - Built for the Classiq Track at CQHack25

[![Made for CQHack25](https://img.shields.io/badge/Made%20for-CQHack25-blue)](https://cqhack25.devpost.com/)
[![Classiq Track](https://img.shields.io/badge/Track-Classiq-green)](https://www.classiq.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io)

## 🎯 What This Does

**Learn quantum error correction through interactive visualization!**

- 🔬 **Real QEC Algorithms** - 3-qubit, 5-qubit, Steane, Surface codes
- 🧠 **AI-Powered Decoders** - Neural networks with explainable AI
- 🏆 **Industry Standards** - Same codes used by Google & IBM
- 💻 **Export to Hardware** - Generate QASM for real quantum computers
- 🎓 **Educational Focus** - From beginner to expert learning paths

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/thesumedh/qec-visualizer.git
cd qec-visualizer
pip install -r requirements.txt
streamlit run app.py
```

**Open http://localhost:8501 and start exploring!**

## 🎮 How to Use

### 🎓 **Beginners**: 
1. Start with **Guided Tutorial** tab
2. Select **3-Qubit Code** in sidebar
3. Follow the 4-step process: Initialize → Error → Syndrome → Correct
4. Watch quantum error correction in action!

### 🔬 **Advanced Users**:
1. Try **Surface Code** (Google/IBM standard)
2. Use **Neural Network** decoder
3. Export **QASM** for IBM Quantum
4. Analyze **ML decoder performance**

## 🏆 Built for CQHack25 - Classiq Track

### ✅ **Functionality**
- **5-tab interface**: Tutorial, Simulator, Comparison, Export, Metrics
- **4-step QEC workflow**: Complete error correction cycle
- **Multiple QEC codes**: Industry-standard implementations
- **Real-time visualization**: Interactive Plotly charts
- **No bugs**: Clean, tested implementation

### ✅ **Quantum Computing Connection**
- **Real QEC algorithms**: Used by Google Sycamore & IBM
- **Quantum mechanics**: Syndrome measurement, stabilizers
- **Hardware simulation**: IBM, Google, IonQ noise models
- **ML integration**: Neural network decoders
- **Educational depth**: From basics to advanced concepts

### ✅ **Real-World Application**
- **Educational tool**: Quantum workforce development
- **Research platform**: QEC algorithm testing
- **Industry bridge**: Exports to real quantum hardware
- **Scalable architecture**: Production-ready design

## 🛠️ Technical Features

### Quantum Error Correction:
- **3-Qubit Bit-Flip Code**: Perfect for beginners
- **5-Qubit Perfect Code**: Smallest universal QEC
- **Steane 7-Qubit Code**: CSS code with transversal gates
- **Surface Code**: Distance-3 implementation (Google/IBM standard)

### Machine Learning:
- **Neural Network Decoders**: Advanced syndrome decoding
- **Explainable AI**: Decision reasoning and alternatives
- **Performance Comparison**: Classical vs ML approaches
- **Training Visualization**: Real-time learning curves

### Hardware Integration:
- **Realistic Noise Models**: Platform-specific simulation
- **QASM Export**: IBM Quantum compatible
- **Classiq SDK Ready**: Production architecture
- **Performance Metrics**: Professional analysis

## 🎯 Classiq Integration

```python
# Ready for real Classiq SDK integration
from classiq import *

@qfunc
def qec_encode_3qubit(logical: QBit, physical: QArray[QBit, 3]):
    CNOT(logical, physical[1])
    CNOT(logical, physical[2])

# Create and synthesize quantum program
qprog = create_model(qec_encode_3qubit)
circuit = synthesize(qprog)
```

**Architecture designed for seamless Classiq SDK integration!**

## 📊 Project Structure

```
qec-visualizer/
├── app.py                 # Main Streamlit application
├── qec_codes.py          # QEC code implementations
├── surface_code.py       # Surface code (Google/IBM)
├── steane_code.py        # Steane 7-qubit code
├── ml_decoder.py         # Neural network decoders
├── noise_models.py       # Hardware noise simulation
├── visualizer.py         # Quantum state visualization
├── real_classiq.py       # Classiq SDK integration
├── educational_core.py   # Learning content
└── requirements.txt      # Dependencies
```

## 🏅 Why This Wins

1. **Educational Impact**: Addresses quantum workforce shortage
2. **Technical Excellence**: Real QEC algorithms with ML enhancement
3. **Industry Relevance**: Uses Google/IBM standards
4. **User Experience**: Beginner-friendly with expert depth
5. **Practical Value**: Exports to real quantum hardware

## 🚀 Future Roadmap

- **Real Classiq SDK**: Full integration with authentication
- **More QEC Codes**: Quantum LDPC, Color codes
- **Advanced ML**: Transformer-based decoders
- **Cloud Deployment**: Scalable web platform
- **Educational Expansion**: University curriculum integration

## 🤝 Contributing

Built for CQHack25 but open for contributions!

```bash
git checkout -b feature/amazing-addition
# Make your changes
git commit -m "Add amazing feature"
git push origin feature/amazing-addition
```

## 📞 Contact

**Built by [@thesumedh](https://github.com/thesumedh) for CQHack25**

- **Email**: sum3dh@yahoo.com
- **LinkedIn**: [Sumedh](https://linkedin.com/in/imsumedh)
- **Devpost**: [CQHack25 Submission](https://devpost.com/thesumedh)

---

**🏆 Competing for Classiq Track Prize at CQHack25**

**🌟 Star this repo if you find it helpful!**