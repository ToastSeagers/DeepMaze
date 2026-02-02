import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class MazeEnv:
    def __init__(self, maze_size=(10, 10), mode='static'):
        self.maze_size = maze_size
        self.mode = mode
        self.maze = None
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None
        self.visit_counts = {}
        self.last_visit_step = {}
        self.step_count = 0
        self.prev_pos = None
        self.position_history = deque(maxlen=10)  # Track last 10 positions
        self.reset()

    def reset(self):
        if self.maze is None:
            self.generate_new_maze()
        self.agent_pos = self.start_pos
        self.visit_counts = {} # Reset visit counts
        self.last_visit_step = {}
        self.step_count = 0
        self.visit_counts[self.start_pos] = 1
        self.last_visit_step[self.start_pos] = 0
        self.prev_pos = None
        self.position_history = deque(maxlen=10)
        self.position_history.append(self.start_pos)
        return self.get_local_view()

    def generate_new_maze(self):
        if self.mode == 'static':
             # Simple fixed maze for testing
            self.maze = np.zeros(self.maze_size, dtype=int)
            # Walls
            self.maze[1:4, 1] = 1
            self.maze[1, 1:4] = 1
            self.maze[3, 3:6] = 1
            self.start_pos = (0, 0)
            self.goal_pos = (self.maze_size[0]-1, self.maze_size[1]-1)
        else:
            # Random maze generation loop
            while True:
                self.maze = np.random.choice([0, 1], size=self.maze_size, p=[0.8, 0.2])
                self.maze[0,0] = 0
                self.maze[-1,-1] = 0
                self.start_pos = (0, 0)
                self.goal_pos = (self.maze_size[0]-1, self.maze_size[1]-1)
                
                if self.is_solvable():
                    break

    def is_solvable(self):
        # BFS to check for path
        start = self.start_pos
        goal = self.goal_pos
        queue = [start]
        visited = set()
        visited.add(start)
        
        while queue:
            curr = queue.pop(0)
            if curr == goal:
                return True
            
            r, c = curr
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.maze_size[0] and 
                    0 <= nc < self.maze_size[1] and 
                    self.maze[nr, nc] == 0 and 
                    (nr, nc) not in visited):
                    queue.append((nr, nc))
                    visited.add((nr, nc))
        return False

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = moves[action]
        
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
        
        reward = -0.01 # Small step penalty
        done = False
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.maze_size[0] or 
            new_pos[1] < 0 or new_pos[1] >= self.maze_size[1]):
            reward = -10 # Wall/Bound penalty
            new_pos = self.agent_pos
        # Check walls
        elif self.maze[new_pos] == 1:
            reward = -10
            new_pos = self.agent_pos
        # Check goal
        elif new_pos == self.goal_pos:
            reward = 100
            done = True
        # Back‑track penalty: discourage immediate reversal
        if self.prev_pos is not None and new_pos == self.prev_pos:
            reward -= 2.0
        
        # Smart Loop Prevention & Exploration 
        # Update visit count for the new position
        self.visit_counts[new_pos] = self.visit_counts.get(new_pos, 0) + 1
        visit_count = self.visit_counts[new_pos]
        
        # 0. Exploration Bonus (New Tile!)
        if visit_count == 1:
            # First time visiting this cell in this episode
            # +10.0 counters the step penalties heavily, making exploration the PRIORITY
            reward += 10.0 
        
        # 1. Recency Penalty (Memory)
        # How long has it been since we were last here?
        self.step_count += 1
        if new_pos in self.last_visit_step:
            turns_since_visited = self.step_count - self.last_visit_step[new_pos]
            if turns_since_visited > 0:
                # 2 turns ago -> -5.0
                # 5 turns ago -> -2.0
                # 100 turns ago -> -0.1
                recency_penalty = -10.0 / turns_since_visited
                reward += recency_penalty
        
        self.last_visit_step[new_pos] = self.step_count

        # 2. Quadratic Boredom Penalty (Visiting same spot gets exponentially worse)
        if visit_count > 1:
            # Base quadratic penalty
            penalty = -0.25 * (visit_count ** 2)
            reward += penalty
            # Additional penalty for revisiting beyond first visit
            reward -= 2.0
            
        self.prev_pos = self.agent_pos # Update previous position to current before moving
        self.agent_pos = new_pos
        self.position_history.append(new_pos)  # Track position in history
            
        return self.get_local_view(), reward, done

    def get_local_view(self, view_range=2):
        # 5x5 grid means radius/range of 2
        # Padding the maze to handle edges
        padded_maze = np.full((self.maze_size[0] + 2*view_range, self.maze_size[1] + 2*view_range), -1) # -1 for out of bounds
        
        # Place actual maze in the center
        # We need to represent the goal in the view. Let's say goal is 2.
        # We need to represent the goal in the view. Let's say goal is 2.
        view_maze = self.maze.astype(float) # Convert to float for gradients
        
        # Apply Footprints (Sight)
        # Iterate over visited positions and paint them
        for pos, count in self.visit_counts.items():
            if self.maze[pos] == 0: # Only paint empty spaces
                # Darken based on visit count. Max out at 0.95 (walls are 1.0)
                # 1 visit = 0.3, 2 visits = 0.6, 3+ visits = 0.9 (Radioactive!)
                darkness = min(count * 0.3, 0.95)
                view_maze[pos] = darkness
        
        view_maze[self.goal_pos] = 2.0 
        
        padded_maze[view_range:view_range+self.maze_size[0], view_range:view_range+self.maze_size[1]] = view_maze
        
        # Agent current relative position in padded maze
        curr_r, curr_c = self.agent_pos[0] + view_range, self.agent_pos[1] + view_range
        
        # Extract 5x5
        local_view = padded_maze[curr_r-view_range : curr_r+view_range+1, curr_c-view_range : curr_c+view_range+1]
        
        # Calculate relative goal position (normalized)
        dy = (self.goal_pos[0] - self.agent_pos[0]) / self.maze_size[0]
        dx = (self.goal_pos[1] - self.agent_pos[1]) / self.maze_size[1]
        
        return np.concatenate((local_view.flatten(), [dy, dx]))

    def get_valid_actions(self):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_actions = []
        possible_moves_count = 0
        
        # First pass: check physics (Walls/Bounds)
        for move in moves:
            r, c = self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]
            
            is_valid = True
            # Check bounds
            if r < 0 or r >= self.maze_size[0] or c < 0 or c >= self.maze_size[1]:
                is_valid = False
            # Check walls
            elif self.maze[r, c] == 1:
                is_valid = False
                
            valid_actions.append(is_valid)
            if is_valid:
                possible_moves_count += 1
                
        # Second pass: Exclusion Zone (Snake Tail Avoidance)
        # Block moves to positions in our recent history
        if possible_moves_count > 1 and len(self.position_history) > 0:
            for i, move in enumerate(moves):
                if not valid_actions[i]:  # Already blocked
                    continue
                r, c = self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]
                if (r, c) in self.position_history:
                    # This position is in our tail! Block it.
                    valid_actions[i] = False
                    possible_moves_count -= 1
                    # Stop if we're about to block all moves (trapped)
                    if possible_moves_count <= 1:
                        # Re-enable the last one we blocked (allow escape)
                        valid_actions[i] = True
                        break

        # Additional mask: prevent immediate opposite move (U‑turn)
        if self.prev_pos is not None:
            dr = self.prev_pos[0] - self.agent_pos[0]
            dc = self.prev_pos[1] - self.agent_pos[1]
            opposite_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
            opposite_action = opposite_map.get((dr, dc))
            if opposite_action is not None and valid_actions[opposite_action] and possible_moves_count > 1:
                valid_actions[opposite_action] = False

            # Visit‑bias: prefer unvisited cells when available
            unvisited_indices = []
            for i, move in enumerate(moves):
                if not valid_actions[i]:
                    continue
                r, c = self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]
                if self.visit_counts.get((r, c), 0) == 0:
                    unvisited_indices.append(i)
            if unvisited_indices:
                for i in range(len(valid_actions)):
                    if i not in unvisited_indices:
                        valid_actions[i] = False

            return valid_actions
