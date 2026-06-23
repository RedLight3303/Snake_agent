"""
AI Snake Agent — Q-Learning
Trains an agent to play Snake using tabular Q-learning.
Watch it go from random to competent in ~500 episodes.

Run:
  python snake_agent.py          # train + watch
  python snake_agent.py --train  # train only (headless, faster)
  python snake_agent.py --play   # load saved model and watch
"""

import random
import sys
import time
import json
import os
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

GRID = 10               # grid size (GRID x GRID)
MAX_STEPS = 200         # max steps per episode before timeout
EPISODES = 1000
ALPHA = 0.1             # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0     # exploration rate
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MODEL_FILE = "q_table.json"

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
OPPOSITES = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


# ── Game ──────────────────────────────────────────────────────────────────────

@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other):
        return Point(self.x + other[0], self.y + other[1])

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        cx, cy = GRID // 2, GRID // 2
        self.snake = deque([Point(cx, cy), Point(cx - 1, cy), Point(cx - 2, cy)])
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.food = self._place_food()
        self.game_over = False
        return self.get_state()

    def _place_food(self) -> Point:
        occupied = set(self.snake)
        while True:
            p = Point(random.randint(0, GRID - 1), random.randint(0, GRID - 1))
            if p not in occupied:
                return p

    def get_state(self) -> Tuple:
        head = self.snake[0]
        d = self.direction

        def danger(direction) -> int:
            nx, ny = head.x + DIRS[direction][0], head.y + DIRS[direction][1]
            next_p = Point(nx, ny)
            return int(
                nx < 0 or nx >= GRID or ny < 0 or ny >= GRID
                or next_p in set(self.snake)
            )

        # Relative directions: straight, right-turn, left-turn
        turn_right = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
        turn_left  = {UP: LEFT,  LEFT: DOWN,  DOWN: RIGHT, RIGHT: UP}

        danger_straight = danger(d)
        danger_right    = danger(turn_right[d])
        danger_left     = danger(turn_left[d])

        # Food direction relative to head
        food_left  = int(self.food.x < head.x)
        food_right = int(self.food.x > head.x)
        food_up    = int(self.food.y < head.y)
        food_down  = int(self.food.y > head.y)

        return (
            danger_straight, danger_right, danger_left,
            int(d == LEFT), int(d == RIGHT), int(d == UP), int(d == DOWN),
            food_left, food_right, food_up, food_down
        )

    def step(self, action: int) -> Tuple[Tuple, float, bool]:
        """
        action: 0 = straight, 1 = turn right, 2 = turn left
        Returns: (new_state, reward, done)
        """
        turn_right = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
        turn_left  = {UP: LEFT,  LEFT: DOWN,  DOWN: RIGHT, RIGHT: UP}

        if action == 1:
            self.direction = turn_right[self.direction]
        elif action == 2:
            self.direction = turn_left[self.direction]

        head = self.snake[0]
        dx, dy = DIRS[self.direction]
        new_head = Point(head.x + dx, head.y + dy)

        # Collision detection
        body_set = set(self.snake)
        if (new_head.x < 0 or new_head.x >= GRID or
                new_head.y < 0 or new_head.y >= GRID or
                new_head in body_set):
            self.game_over = True
            return self.get_state(), -10.0, True

        self.snake.appendleft(new_head)
        self.steps += 1

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._place_food()
        else:
            self.snake.pop()
            # Small reward for moving toward food
            old_dist = abs(head.x - self.food.x) + abs(head.y - self.food.y)
            new_dist = abs(new_head.x - self.food.x) + abs(new_head.y - self.food.y)
            reward = 0.1 if new_dist < old_dist else -0.1

        done = self.steps >= MAX_STEPS
        return self.get_state(), reward, done


# ── Agent ─────────────────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(self):
        self.q: dict = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.epsilon = EPSILON_START

    def choose_action(self, state: Tuple, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        q_vals = self.q[state]
        max_q = max(q_vals)
        # Break ties randomly
        best = [i for i, v in enumerate(q_vals) if v == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done):
        current_q = self.q[state][action]
        target = reward if done else reward + GAMMA * max(self.q[next_state])
        self.q[state][action] += ALPHA * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path: str):
        data = {str(k): v for k, v in self.q.items()}
        with open(path, "w") as f:
            json.dump({"q_table": data, "epsilon": self.epsilon}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.q = defaultdict(lambda: [0.0, 0.0, 0.0])
        for k, v in data["q_table"].items():
            self.q[eval(k)] = v
        self.epsilon = data.get("epsilon", EPSILON_END)


# ── Training ──────────────────────────────────────────────────────────────────

def train(episodes: int = EPISODES, verbose: bool = True) -> QLearningAgent:
    agent = QLearningAgent()
    game = SnakeGame()
    scores = []

    for ep in range(1, episodes + 1):
        state = game.reset()
        total_reward = 0.0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = game.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        scores.append(game.score)

        if verbose and ep % 100 == 0:
            avg = sum(scores[-100:]) / 100
            best = max(scores[-100:])
            print(f"  Episode {ep:>5} | Avg score: {avg:.2f} | Best: {best} | ε: {agent.epsilon:.3f}")

    agent.save(MODEL_FILE)
    print(f"\n  Training complete. Model saved to {MODEL_FILE}")
    return agent


# ── Visualizer (terminal) ─────────────────────────────────────────────────────

def render(game: SnakeGame):
    head = game.snake[0]
    body = set(list(game.snake)[1:])

    os.system("clear" if os.name == "posix" else "cls")
    print(f"  Score: {game.score}  Steps: {game.steps}")
    print("  +" + "-" * (GRID * 2) + "+")
    for y in range(GRID):
        row = "  |"
        for x in range(GRID):
            p = Point(x, y)
            if p == head:
                row += "O "
            elif p in body:
                row += "# "
            elif p == game.food:
                row += "* "
            else:
                row += ". "
        print(row + "|")
    print("  +" + "-" * (GRID * 2) + "+")


def watch(agent: QLearningAgent, episodes: int = 5, delay: float = 0.05):
    game = SnakeGame()
    for ep in range(episodes):
        state = game.reset()
        print(f"\n  === Episode {ep + 1} ===")
        while True:
            render(game)
            action = agent.choose_action(state, training=False)
            state, _, done = game.step(action)
            time.sleep(delay)
            if done:
                render(game)
                print(f"\n  Final score: {game.score}")
                time.sleep(1)
                break


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--both"

    if mode == "--train":
        print("  Training agent (headless)...\n")
        train()

    elif mode == "--play":
        agent = QLearningAgent()
        if not os.path.exists(MODEL_FILE):
            print(f"  No saved model found ({MODEL_FILE}). Run with --train first.")
            sys.exit(1)
        agent.load(MODEL_FILE)
        agent.epsilon = 0.0
        watch(agent, episodes=3)

    else:
        print("  Training agent...\n")
        agent = train()
        print("\n  Watching trained agent play...\n")
        time.sleep(1)
        agent.epsilon = 0.0
        watch(agent, episodes=3)
