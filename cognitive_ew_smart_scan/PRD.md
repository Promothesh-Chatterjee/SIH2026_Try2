# Product Requirements Document (PRD)

## 1. Project Overview
**Name:** Cognitive Electronic Warfare Smart Scan Scheduler
**Context:** Smart India Hackathon (SIH) 2026 Problem Statement SIH26056 (DRDO)
**Objective:** Build a machine learning-based Electronic Support (ES) receiver scheduler that deinterleaves radar Pulse Descriptor Words (PDWs) and autonomously schedules frequency band scanning in real-time without prior threat libraries.

## 2. Target Users & Environment
- **Target Users:** Defence operational units, Electronic Warfare (EW) systems engineers, and SIH Judges.
- **Deployment Target:** Edge hardware (NVIDIA Jetson AGX Orin).
- **Latency Constraints:** Sub-millisecond scheduling decisions.

## 3. Core Features
- **Deinterleaving:** Transformer-based metric learning (Triplet Loss) clustering with HDBSCAN on 5D/6D PDWs (ToA, CF, PW, AoA, Amplitude).
- **Smart Scheduling:** Deep Recurrent Q-Network (DRQN) fused with a Revisit Agent via Mixture of Experts (MoE) to determine optimal RF band scanning.
- **Cognitive Memory:** SQLite-backed Semantic Memory (for learned threat characteristics) and Episodic Memory (LSTM hidden states).
- **Periodic Interception:** Heuristic-based PRI estimation to preemptively tune to frequency-agile periodic emitters.
- **API & Deployment:** FastAPI REST interface and ONNX exports for low-latency inference.

## 4. Evaluation Criteria (SIH)
The final Evaluation metrics should be based upon the following parameters:
- V-measure (scan)
- AMI (scan)
- Pd (Probability of Detection)
- Pfa (Probability of False Alarm)
- Avg Intercept Rate
- Scheduler Latency
- Avg Intercept Time Error
