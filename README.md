# AI Snake Agent – Q-Learning

A reinforcement learning project that trains an AI agent to play Snake using tabular Q-learning. Starting with random actions, the agent gradually learns effective strategies through trial and error, improving its performance over hundreds of training episodes.

## Features

* Tabular Q-Learning implementation
* Autonomous Snake-playing AI
* Configurable training parameters
* Epsilon-greedy exploration strategy
* Q-table model saving and loading
* Headless training mode for faster learning
* Terminal-based visualization
* Performance tracking during training

## How It Works

The agent learns by interacting with the Snake environment:

1. Observes the current game state
2. Selects an action
3. Receives a reward or penalty
4. Updates its Q-table
5. Repeats over thousands of episodes

Over time, the agent learns which actions maximize long-term rewards and survival.

## State Representation

The AI evaluates:

* Danger straight ahead
* Danger to the left
* Danger to the right
* Current movement direction
* Food location relative to the snake

This compact state representation allows efficient learning while keeping the Q-table manageable.

## Rewards

| Event               | Reward |
| ------------------- | ------ |
| Eat Food            | +10    |
| Move Closer to Food | +0.1   |
| Move Away from Food | -0.1   |
| Collision           | -10    |

## Installation

```bash
git clone https://github.com/yourusername/ai-snake-agent.git
cd ai-snake-agent
```

## Usage

Train the agent:

```bash
python snake_agent.py --train
```

Train and watch:

```bash
python snake_agent.py
```

Load a saved model and watch it play:

```bash
python snake_agent.py --play
```

## Technologies Used

* Python 3
* Reinforcement Learning
* Q-Learning
* JSON Model Persistence
* Object-Oriented Programming

## Learning Objectives

This project demonstrates:

* Reinforcement learning fundamentals
* Markov decision processes
* Exploration vs. exploitation
* Reward-based learning
* State representation design
* Autonomous agent development

## Future Improvements

* Deep Q-Networks (DQN)
* Larger game environments
* Neural network function approximation
* Training visualizations
* Performance analytics dashboard

## License

MIT License
