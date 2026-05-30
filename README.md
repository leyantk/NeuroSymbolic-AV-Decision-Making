# Neuro-Symbolic Framework for LLM-Based Autonomous Vehicle Decision-Making

## Overview

This repository contains the implementation of a hybrid decision-making framework developed as part of a bachelor's thesis investigating the reliability of Large Language Models (LLMs) in autonomous vehicle decision-making.

The system combines a generative reasoning layer powered by a Large Language Model with a deterministic Python governance layer that validates, scores, and, when necessary, overrides unsafe or unethical decisions. The objective is to evaluate whether formalized ethical constraints and deterministic validation can improve the safety, consistency, and controllability of LLM-driven autonomous driving systems.

---

## Research Objectives

The central research question is:

How can explicit norm and safety representations improve decision reliability in autonomous driving scenarios?

To address this question, the study explores the following sub-questions:

How should traffic rules be represented symbolically?
How should conflicts between safety and legality be resolved?
Can structured reasoning reduce unsafe or illegal proposals?
Does explicit constraint checking improve consistency?

---

## System Architecture

The framework follows a two-layer architecture:

### 1. Generative Decision Layer

A Large Language Model receives a natural-language driving scenario and:

* Performs spatial reasoning.
* Extracts structured environmental state variables.
* Applies an assigned ethical framework.
* Proposes a driving action.
* Produces a structured chain of thought detailing the reasoning behind its decision.

Possible actions include:

* Go forward
* Go left
* Go right
* Stop

### 2. Deterministic Governance Layer

The governance layer independently evaluates every physically possible action using rule-based validation functions.

The layer checks:

#### Safety Constraints

* Pedestrian avoidance
* Vehicle time-to-collision (TTC)
* Static obstacle avoidance

#### Traffic Law Compliance

* Traffic signals
* Speed limits
* Lane boundaries
* Right-of-way rules
* Emergency vehicle protocols

#### Ethical Penalty Framework

Each violation is assigned a weighted penalty according to the selected moral profile.

The governance layer:

1. Scores all available actions.
2. Identifies the optimal action.
3. Compares it against the LLM proposal.
4. Overrides unsafe or suboptimal decisions when necessary.

---

## Ethical Profiles

The framework supports four ethical decision-making models:

### Utilitarian

Prioritizes minimizing overall harm and may violate traffic laws if doing so prevents greater harm.

### Deontological

Treats safety and legal duties as obligations that must be followed regardless of consequences.

### Absolutist

Enforces zero tolerance for both legal and safety violations and defaults to stopping in unresolved dilemmas.

### Moral Relativist

Allows customizable moral weights that reflect context-specific values and preferences.

---

## Constraint Formalizations

Two knowledge representations are supported:

### Natural Language (NL)

Human-readable ethical and legal rules.

### First-Order Logic (FOL)

Formal symbolic representations of the same constraints.

Example:

Pedestrian Avoidance

```
∀x (Pedestrian(x) ∧ Distance(Ego,x) < 3
→ Action(Ego, Stop))
```

---

## Features

* LLM-based scenario reasoning
* Structured state extraction
* Multi-framework ethical evaluation
* Deterministic safety governance
* Automatic decision override mechanism
* Formal logic support
* Simulation logging
* CSV experiment recording
* Quantitative safety analysis

---

## Simulation Modes

### Phase 1 — Baseline

Pure LLM decision-making.

The proposed action is executed without intervention.

### Phase 2 — Governance Framework

LLM decisions are validated by the deterministic governance layer.

Unsafe or suboptimal actions may be overridden.

---

## Logged Metrics

Each simulation run records the following information for later analysis and evaluation:

* Timestamp
* Simulation phase
* Scenario ID
* Constraint formalization type
* Moral profile
* System prompt length
* Format error retry count
* LLM proposed action
* LLM action score
* LLM safety status
* LLM rule violations
* Final executed action
* Final safety status
* Final rule violations
* Maximum governance score
* Override trigger status
* Override type
* Extracted state variables
* Chain-of-thought reasoning

Results are stored in:

```
thesis_simulation_logs.csv
```

---

## Installation

### Prerequisites

* Python 3.10+
* Groq API Key

### Install Dependencies

```bash
pip install pyautogen
pip install python-dotenv
```

### Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Running the Simulation

```bash
python main.py
```

You will be prompted to select:

1. Simulation phase
2. Constraint formalization
3. Ethical profile

The system will then execute the selected driving scenario and display both the LLM decision and governance evaluation.

---

## Example Workflow

```
Driving Scenario
        │
        ▼
 Large Language Model
        │
        ▼
 Structured JSON State
        │
        ▼
 Governance Matrix
        │
        ▼
 Action Scoring
        │
        ▼
 Override Decision
        │
        ▼
 Final Executed Action
```

---

## Thesis Contribution

This work demonstrates how deterministic governance mechanisms can be integrated with generative AI systems to improve safety, reduce ethical drift, and increase decision reliability in autonomous vehicle environments.

The findings suggest that combining symbolic validation with LLM reasoning offers a practical pathway toward more trustworthy AI-driven autonomous systems while highlighting areas that require further research and refinement.

---

## License

This repository is released for academic and research purposes.
Please cite the associated thesis if using this work in future research.
