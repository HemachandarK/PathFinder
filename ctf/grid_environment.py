import numpy as np

class GridEnv:
    def __init__(self, size=(5, 5)):
        self.size = size
        self.state = (0, 0)  # Starting position
        self.goal = (size[0] - 1, size[1] - 1)  # Goal position
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        new_x = max(0, min(self.size[0] - 1, x + dx))
        new_y = max(0, min(self.size[1] - 1, y + dy))
        self.state = (new_x, new_y)

        # Reward is 1 for reaching the goal, -1 for each step taken
        reward = 1 if self.state == self.goal else -1
        done = self.state == self.goal

        return self.state, reward, done

    def is_done(self):
        return self.state == self.goal
