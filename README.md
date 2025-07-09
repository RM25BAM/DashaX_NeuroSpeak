# DashaX NeuroSpeak: 

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

---

This research investigation we are proposing a system that implements a semantic-aware deep reinforcement learning system to classify imagined words from EEG signals.
##  Project Architecture
Testing phase
### Architecture Flow:

1. **EEG Data**: We plan to use OpenNeuro imagined speech dataset + kumar imagined speech + etc... .
2. **Preprocessing**: Filtering, artifact removal (ICA), and epoching. (but still in investigation)
3. **Feature Encoder**: GRU with attention or Transformer encoder converts EEG into latent(compress and structued) representation to be fed to the DQN agent in compacted state vectors.
4. **RL Classifier**: A DQN agent learns to classify EEG signals into word categories.
5. **Semantic Reward**: BERT-based embeddings provide fine-grained rewards based on semantic similarity (not just exact match).
6. **Optional Confidence Module**: Adjusts exploration/exploitation based on prediction certainty.
7. **Output Module**: Displays predicted word and confidence score.

> The novelty lies in combining RL, imagined *word-level* decoding, and semantic shaping for improved generalization.

---

## Table of Contents
- [Installation]()
- [Documentation](./docs/README.md)

## Contributing
We love contributions! Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) for more information on how to get involved.

