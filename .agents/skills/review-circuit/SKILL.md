---
name: review-circuit
description: Review a quantum circuit or hybrid QML block with repository-aware best practices for expressivity, gradient flow, and training stability.
---

# Review Circuit

Use this skill for quantum circuit review, not for generic style feedback.

## Workflow

1. Identify the target:
   - `src/model.py`
   - `src/trainable_quantum_model.py`
   - `improved_quantum_circuit.py`
   - another user-specified file
2. Read the local implementation and surrounding training code.
3. Spawn the `quantum_ml_reviewer` custom agent when the review is substantial.
4. Focus on:
   - ansatz suitability
   - entanglement pattern
   - differentiability method
   - barren plateau risk
   - runtime vs expressivity tradeoff
   - correctness of the classical-quantum interface
5. Return findings first, then concrete improvement options.
