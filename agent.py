import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from model import DQN

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay=0.995):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def act(self, state, valid_actions=None):
        # valid_actions: list of booleans [True, False, True, True] for [Up, Down, Left, Right]
        
        if random.random() < self.epsilon:
            if valid_actions:
                possible_indices = [i for i, valid in enumerate(valid_actions) if valid]
                if possible_indices:
                    return random.choice(possible_indices)
            return random.randint(0, self.output_dim - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            
        if valid_actions:
            # Mask invalid actions with negative infinity
            for i, valid in enumerate(valid_actions):
                if not valid:
                    q_values[0][i] = -float('inf')
                    
        return q_values.argmax().item()

    def cache(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        curr_q = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
        expected_q = reward + (1 - done) * self.gamma * next_q
        
        loss = nn.MSELoss()(curr_q, expected_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path)
            
            # Check if it's the new full checkpoint format
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                    print(f"Loaded checkpoint. Epsilon restored to: {self.epsilon:.4f}")
            else:
                # Fallback for old simple weights-only format
                print("Detected legacy model format (weights only).")
                self.policy_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # Should we reset epsilon? If it's legacy, it's just weights. 
                # We'll stick to default init epsilon unless manually set elsewhere.
                
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("Error: Saved model architecture does not match the current 'Dueling DQN' model.")
                print("The neural network size has been increased (128 -> 256 neurons) and structure changed.")
                print("Please start training a fresh model or verify you are loading the correct file.")
            else:
                print(f"RuntimeError loading checkpoint: {e}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
