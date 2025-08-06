# Deep Q-Learning Snake AI

An implementation of Deep Q-Network (DQN) reinforcement learning to train an AI agent to play the classic Snake game. This project demonstrates the application of deep reinforcement learning techniques to game environments, showcasing how an AI agent can learn optimal strategies through trial and error.

## Table of Contents

1. [Methodology](#methodology)
2. [Hypothesis and Outcomes](#hypothesis-and-outcomes)
3. [Challenges and Potential Improvements](#challenges-and-potential-improvements)
4. [Key Takeaways](#key-takeaways)
5. [Project Structure](#project-structure)
6. [Installation and Usage](#installation-and-usage)

## Methodology

### Deep Q-Network (DQN) Architecture

This project implements a Deep Q-Network using PyTorch, which combines Q-learning with deep neural networks to handle the continuous state space of the Snake game.

#### State Representation

The agent perceives the game environment through an 11-dimensional state vector:

**Danger Detection (3 features):**
- Danger straight ahead
- Danger to the right
- Danger to the left

**Current Direction (4 features):**
- Moving left (boolean)
- Moving right (boolean)
- Moving up (boolean)
- Moving down (boolean)

**Food Location (4 features):**
- Food is left of snake head
- Food is right of snake head  
- Food is above snake head
- Food is below snake head

This state representation provides the agent with essential information about immediate dangers and the food's relative position, enabling strategic decision-making.

#### Neural Network Architecture

```
Input Layer:  11 neurons (state features)
Hidden Layer: 256 neurons (ReLU activation)
Output Layer: 3 neurons (action probabilities)
```

The network outputs Q-values for three possible actions:
- **[1,0,0]**: Continue straight
- **[0,1,0]**: Turn right
- **[0,0,1]**: Turn left

#### Training Process

**Experience Replay:**
- Memory buffer size: 100,000 transitions
- Batch size: 1,000 for long-term memory training
- Each transition stores: (state, action, reward, next_state, done)

**Reward System:**
- +10 points for eating food
- -10 points for collision (game over)
- 0 points for regular moves

**Exploration Strategy:**
- Epsilon-greedy policy with decaying exploration
- Initial epsilon: 80 (high exploration)
- Decay rate: epsilon = 80 - number_of_games
- Random action probability decreases as training progresses

**Learning Parameters:**
- Learning rate: 0.001
- Discount factor (gamma): 0.9
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

#### Training Loop

1. **State Observation**: Agent observes current game state
2. **Action Selection**: Choose action using epsilon-greedy policy
3. **Environment Interaction**: Execute action and receive reward
4. **Memory Storage**: Store experience in replay buffer
5. **Short-term Learning**: Train on current experience
6. **Long-term Learning**: Train on random batch from memory buffer
7. **Model Persistence**: Save model when new high score is achieved

## Hypothesis and Outcomes

### Initial Hypothesis

**Primary Hypothesis:** A Deep Q-Network can successfully learn to play Snake by:
1. Learning to avoid collisions with walls and its own body
2. Developing strategies to efficiently navigate toward food
3. Achieving scores comparable to or exceeding average human performance

**Secondary Hypotheses:**
- The 11-dimensional state representation provides sufficient information for optimal play
- Experience replay will stabilize learning and prevent catastrophic forgetting
- Epsilon-greedy exploration will balance exploration and exploitation effectively

### Observed Outcomes

**Learning Progression:**
- Early games (0-50): Random exploration, frequent collisions, scores typically 0-3
- Mid-training (50-200): Agent learns basic collision avoidance, scores improve to 5-15
- Advanced training (200+): Strategic food collection, scores reaching 20-40+

**Performance Metrics:**
- The agent demonstrates clear learning progression through score improvements
- Collision avoidance becomes highly effective after sufficient training
- Food collection strategies emerge, showing path optimization behavior

**Behavioral Analysis:**
- Agent successfully learns to prioritize immediate survival over food collection when necessary
- Develops efficient turning patterns to navigate tight spaces
- Shows evidence of planning ahead to avoid trapping itself

### Success Indicators

✅ **Collision Avoidance**: Agent reliably avoids walls and self-collision  
✅ **Food Navigation**: Demonstrates directed movement toward food  
✅ **Score Improvement**: Clear upward trend in performance over time  
✅ **Strategy Development**: Evidence of learned behavioral patterns  

## Challenges and Potential Improvements

### Current Challenges

#### 1. State Representation Limitations
**Challenge:** The current 11-dimensional state only provides local information about immediate dangers and food direction.

**Impact:** 
- Agent may get trapped in situations requiring longer-term planning
- Cannot anticipate complex scenarios multiple moves ahead
- Limited awareness of tail position and body configuration

#### 2. Reward Sparsity
**Challenge:** Rewards are only given for eating food (+10) or dying (-10), with no intermediate rewards.

**Impact:**
- Slow learning in early stages due to infrequent positive feedback
- Difficulty learning optimal paths when multiple routes to food exist
- No incentive for efficient movement patterns

#### 3. Exploration-Exploitation Balance
**Challenge:** Simple linear epsilon decay may not be optimal for all learning phases.

**Impact:**
- May reduce exploration too quickly in complex scenarios
- Could benefit from adaptive exploration based on performance

#### 4. Training Stability
**Challenge:** No target network implementation, which can lead to unstable learning.

**Impact:**
- Potential for oscillating Q-values during training
- Less stable convergence compared to Double DQN approaches

### Potential Improvements

#### 1. Enhanced State Representation
```python
# Proposed improvements:
- Distance to walls in all directions
- Tail position relative to head
- Safe space analysis (number of reachable empty cells)
- Food distance (Manhattan distance)
- Body segment positions for better self-collision awareness
```

#### 2. Reward Engineering
```python
# Proposed reward modifications:
- Small positive reward for moving toward food
- Small negative reward for moving away from food
- Penalty for inefficient movement patterns
- Bonus for maintaining safe distances from walls/body
```

#### 3. Advanced DQN Techniques
- **Double DQN**: Separate target network for more stable learning
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Focus on important transitions
- **Rainbow DQN**: Combination of multiple DQN improvements

#### 4. Architecture Enhancements
```python
# Neural network improvements:
- Deeper networks with multiple hidden layers
- Convolutional layers for spatial awareness
- LSTM layers for temporal dependencies
- Attention mechanisms for focus on relevant state features
```

#### 5. Training Optimizations
- **Curriculum Learning**: Start with smaller grid sizes, gradually increase
- **Population-Based Training**: Train multiple agents with different hyperparameters
- **Self-Play**: Train against previous versions of the agent

#### 6. Evaluation Metrics
- Average survival time
- Food collection efficiency (food/moves ratio)
- Spatial coverage analysis
- Comparison with human baseline performance

## Key Takeaways

### Technical Insights

#### 1. State Design is Critical
The choice of state representation fundamentally determines what the agent can learn. The 11-dimensional state works for basic gameplay but limits advanced strategic thinking.

**Lesson:** Invest significant time in designing state representations that capture all relevant information while remaining computationally tractable.

#### 2. Reward Shaping Matters
The sparse reward structure (only +10/-10) makes learning slower but more robust. The agent learns genuine survival strategies rather than exploiting reward engineering artifacts.

**Lesson:** Simple reward structures often lead to more generalizable learned behaviors, though they may require longer training times.

#### 3. Experience Replay is Powerful
The replay buffer significantly stabilizes learning by breaking temporal correlations and allowing the agent to learn from diverse experiences.

**Lesson:** Experience replay is essential for DQN implementations, especially in environments with sequential dependencies.

#### 4. Exploration Strategy Impact
The linear epsilon decay works reasonably well but could be optimized. The agent benefits from extended exploration periods in complex scenarios.

**Lesson:** Consider adaptive exploration strategies that respond to learning progress and environmental complexity.

### Reinforcement Learning Principles

#### 1. Learning is Gradual and Hierarchical
The agent first learns basic survival (collision avoidance), then progresses to goal-seeking (food collection), and finally develops strategic thinking.

**Insight:** Complex behaviors emerge from simpler learned components, following a natural curriculum.

#### 2. Generalization vs. Memorization
The agent learns general strategies rather than memorizing specific game states, as evidenced by performance on novel food configurations.

**Insight:** Proper regularization through experience replay and exploration prevents overfitting to specific scenarios.

#### 3. Sample Efficiency Challenges
DQN requires substantial training time to achieve competent performance, highlighting the sample efficiency challenges in deep RL.

**Insight:** Real-world applications need to consider training time and computational costs when choosing RL approaches.

### Practical Development Insights

#### 1. Debugging RL is Complex
Unlike supervised learning, RL debugging requires analyzing agent behavior, reward signals, and learning dynamics simultaneously.

**Strategy:** Implement comprehensive logging and visualization tools early in development.

#### 2. Hyperparameter Sensitivity
Small changes in learning rate, epsilon decay, or network architecture can significantly impact learning outcomes.

**Strategy:** Systematic hyperparameter tuning is essential for optimal performance.

#### 3. Baseline Comparisons are Valuable
Having a human-playable version provides crucial context for evaluating AI performance and identifying behavioral differences.

**Strategy:** Always implement baseline comparisons to validate learning progress.

### Broader Applications

This Snake AI project demonstrates principles applicable to many domains:

- **Robotics**: Path planning and obstacle avoidance
- **Game AI**: Strategic decision-making in constrained environments
- **Autonomous Systems**: Navigation and goal-oriented behavior
- **Resource Management**: Optimization under constraints

The techniques learned here scale to more complex environments with appropriate modifications to state representation and network architecture.

## Project Structure

```
RL_Snake/
├── agent.py              # DQN agent implementation
├── model.py               # Neural network and trainer classes
├── game.py                # Snake game environment for AI
├── snake_game_human.py    # Human-playable version for baseline
├── helper.py              # Plotting utilities for training visualization
├── arial.ttf              # Font file for game display
├── model/
│   └── model.pth          # Saved trained model weights
├── test.ipynb             # Jupyter notebook for experimentation
└── README.md              # This comprehensive documentation
```

## Installation and Usage

### Prerequisites
```bash
pip install torch pygame matplotlib numpy ipython
```

### Training the Agent
```bash
python agent.py
```

### Playing Human Version
```bash
python snake_game_human.py
```

### Loading Trained Model
```python
from agent import Agent
from game import SnakeGameAI

agent = Agent()
agent.model.load_state_dict(torch.load('./model/model.pth'))
game = SnakeGameAI()

# Run trained agent
while True:
    state = agent.get_state(game)
    action = agent.get_action(state)
    reward, done, score = game.play_step(action)
    if done:
        break
```

---

*This project serves as an educational demonstration of Deep Q-Learning applied to a classic game environment, showcasing both the potential and limitations of current deep reinforcement learning techniques.*