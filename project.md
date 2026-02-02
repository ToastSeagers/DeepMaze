# Deep Maze Solver

## Overview

This project implements an AI agent capable of solving mazes using **Deep Reinforcement Learning (DQN)**. Unlike traditional maze solvers that memorize a full map, this agent uses **Local Vision** (a 5x5 grid around itself) to make decisions, allowing it to generalize to new, unseen mazes.

## How It Works

### 1. The Brain (Dueling DQN)

The agent uses a **Dueling Deep Q-Network** to estimate the "value" of taking an action given its current view.

- **Input**: A flattened grid (local vision) + Goal Direction (normalized dy, dx).
- **Architecture**: Dueling DQN (Value Stream + Advantage Stream).
- **Hidden Layers**: Shared layer of 256 neurons, splitting into two streams of 128 neurons.
- **Output**: 4 values (Q-Value for Up, Down, Left, Right).

### 2. Local Vision

The agent **cannot** see the entire maze. It only sees:

- **0**: Open Path (Unvisited).
- **0.3 - 0.95**: Footprints (Visited Path). 3 visits = Wall-like darkness.
- **1**: Wall.
- **2**: Goal (if within range).
- **-1**: Out of Bounds.

### 3. Training (Q-Learning)

The agent learns through trial and error:

- **Exploration**: Initially, it moves randomly to discover the map.
- **Reward System**:
  - **+100**: Reaching the Goal.
  - **+10.0**: Mega Exploration Bonus (Finding a new tile is the #1 priority).
  - **-1**: Taking a step.
  - **-10.0 / Age**: Recency Penalty (Immediate return = -5, Return after long time = -0.1).
  - **-0.25 \* (Visits^2)**: Quadratic Boredom Penalty (repeatedly visiting the same spot).
- **Action Masking**: The agent is forcibly prevented from choosing invalid moves (hitting walls), ensuring 100% efficient movement decisions.
- **Smart Masking (New)**: The agent is physically prevented from turning 180 degrees (U-Turn) unless it is in a dead end. This guarantees forward progress.
- **Optimization**: The network updates its weights to maximize the total expected reward.

## Installation

1.  **Prerequisites**: Python 3.8+
2.  **Setup Virtual Environment** (Optional but Recommended):
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Graphical User Interface (GUI) - Recommended

The GUI provides a real-time visualization of the training process and allows you to test the agent on different maps.

**Run command:**

```sh
python gui.py
```

**Features:**

- **Configuration**:
  - **Episodes**: Structure the training duration (default: 500).
  - **Epsilon Decay**: Controls how fast the agent stops exploring and starts acting intelligently (default: 0.995).
  - **Model Management**:
    - **Save/Load**: Preserves not just weights, but also the agent's "wisdom" (epsilon) and optimizer state for endless training.
  - **Controls**:
    - **Start Training**: Begins the learning process.
    - **Stop**: Pauses/Stops the current run.
    - **Generate New Maze**: Creates a random new 10x10 maze layout.
    - **Run Test**: Runs a test episode with a **Step Counter**.
  - **Map Editor**:
    - **Left Click** on any cell in the maze to toggle a Wall (Black) or Empty Space (White).
    - **Note**: You cannot edit the Start (S) or Goal (G) positions.

### 2. Command Line Interface (CLI)

For a headless training session that saves a plot at the end.

**Run command:**

```sh
python main.py
```

## Project Structure

- **`gui.py`**: The entry point for the Tkinter-based GUI.
- **`main.py`**: The entry point for the CLI training explanation.
- **`agent.py`**: Contains the `DQNAgent` class (Brain, Memory, Learning Logic).
- **`model.py`**: Defines the PyTorch Neural Network architecture.
- **`environment.py`**: Manages the Maze logic, movement rules, and "Local Vision" extraction.
- **`requirements.txt`**: List of Python libraries required.
