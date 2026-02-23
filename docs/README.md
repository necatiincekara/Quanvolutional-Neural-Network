# Documentation Index

This directory contains comprehensive documentation for the Hybrid Quantum-Classical Convolutional Neural Network project.

## 📚 Documentation Map

### Core Documents

1. **[AUDIT_REPORT.md](AUDIT_REPORT.md)** - Start here!
   - Comprehensive codebase audit (November 2025)
   - Architectural analysis and bottlenecks
   - V7-V10 development roadmap
   - Engineering best practices
   - Quantum hardware deployment plan

2. **[EXPERIMENTS.md](EXPERIMENTS.md)**
   - Detailed experimental log (V1-V6)
   - Performance benchmarks and metrics
   - Architectural evolution history
   - Lessons learned from failures

3. **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**
   - Step-by-step development guide
   - V7-V10 implementation instructions
   - Hyperparameter configurations
   - Debugging common issues

### Research & Strategy

4. **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)**
   - Publication strategy
   - Research timeline (82% → 90% accuracy)
   - Target venues (NeurIPS, ICML)
   - Success metrics and KPIs

5. **[QUANTUM_ML_RECOMMENDATIONS.md](QUANTUM_ML_RECOMMENDATIONS.md)**
   - Quantum ML best practices
   - Circuit design guidelines
   - Gradient flow optimization
   - Barren plateau mitigation

### Platform Guides (November 2025 Updates)

6. **[COMPUTING_RESOURCES_2025.md](COMPUTING_RESOURCES_2025.md)** ⭐ **NEW - START HERE!**
   - Python 3.12/3.13 recommendations
   - M4 Mac Mini analysis (CRITICAL: CUDA limitation!)
   - Google Colab Pro + VS Code Extension complete guide
   - Optimal workflow setup instructions
   - Performance benchmarks and cost analysis

7. **[TRAINING_PLATFORM_GUIDE.md](TRAINING_PLATFORM_GUIDE.md)** (Historical)
   - Earlier platform comparison
   - See #6 for current recommendations

8. **[COLAB_SETUP.md](COLAB_SETUP.md)**
   - Google Colab configuration
   - GPU setup and optimization
   - Session management tips

---

## 🗺️ Quick Navigation

### I want to...

- **🆕 Set up my environment (Python, Colab, VS Code)** → [COMPUTING_RESOURCES_2025.md](COMPUTING_RESOURCES_2025.md) ⭐ **START HERE!**
- **Understand the current state** → [AUDIT_REPORT.md](AUDIT_REPORT.md)
- **See experimental results** → [EXPERIMENTS.md](EXPERIMENTS.md)
- **Start implementing V7** → [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Plan research timeline** → [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)
- **Optimize quantum circuits** → [QUANTUM_ML_RECOMMENDATIONS.md](QUANTUM_ML_RECOMMENDATIONS.md)
- **Understand M4 Mac limitations** → [COMPUTING_RESOURCES_2025.md § 2](COMPUTING_RESOURCES_2025.md#2-m4-mac-mini-2024---detailed-analysis)

---

## 📊 Project Status (November 2025)

**Current Baseline**: V4 (8×8 feature maps, 8.75% accuracy, 1.5h/epoch)

**Critical Findings**:
- ✅ 93.75% reduction in quantum evaluations (V1→V4)
- ❌ V6 gradient vanishing prevents further optimization
- 🎯 Target: 90% accuracy via V7-V10 roadmap

**Next Steps**: See [AUDIT_REPORT.md § 5. Development Roadmap](AUDIT_REPORT.md#5-development-roadmap)

---

## 📝 Document Update History

| Date | Document | Changes |
|------|----------|---------|
| 2025-11-16 | **COMPUTING_RESOURCES_2025.md** | 🆕 Complete platform analysis (Python 3.12, M4 Mac, Colab Pro + VS Code) |
| 2025-11-16 | requirements.txt | Updated to Python 3.12/3.13, PyTorch 2.6+, PennyLane 0.43+ |
| 2025-11-16 | AUDIT_REPORT.md | Initial comprehensive audit + platform recommendations |
| 2025-11-16 | All docs | Reorganized into docs/ directory |
| Earlier | EXPERIMENTS.md | V1-V6 experimental log |

---

## 🔗 Related Files

- **Main README**: [../README.md](../README.md)
- **Claude Instructions**: [../CLAUDE.md](../CLAUDE.md)
- **Source Code**: [../src/](../src/)
- **Experimental Scripts**: [../experiments/](../experiments/)
