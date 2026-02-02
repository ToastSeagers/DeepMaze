import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from environment import MazeEnv
from agent import DQNAgent

class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Maze Solver")
        self.root.geometry("1000x800")
        
        self.env = MazeEnv(mode='static')
        self.agent = DQNAgent(input_dim=27, output_dim=4)
        
        self.training = False
        self.testing = False
        self.stop_event = threading.Event()
        self.rewards_history = []
        self.total_episodes_trained = 0
        
        self.setup_ui()
        self.draw_maze()
        
    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=200, bg="#f0f0f0", padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(sidebar, text="Configuration", font=("Arial", 14)).pack(pady=10)
        
        tk.Label(sidebar, text="Episodes:").pack(anchor=tk.W)
        self.episodes_var = tk.StringVar(value="500")
        tk.Entry(sidebar, textvariable=self.episodes_var).pack(fill=tk.X)
        
        tk.Label(sidebar, text="Epsilon Decay:").pack(anchor=tk.W, pady=(5,0))
        self.decay_var = tk.StringVar(value="0.995")
        tk.Entry(sidebar, textvariable=self.decay_var).pack(fill=tk.X)

        tk.Label(sidebar, text="Min Epsilon:").pack(anchor=tk.W, pady=(5,0))
        self.min_epsilon_var = tk.StringVar(value="0.10")
        tk.Entry(sidebar, textvariable=self.min_epsilon_var).pack(fill=tk.X)
        
        tk.Label(sidebar, text="Current Epsilon:").pack(anchor=tk.W, pady=(5,0))
        self.current_epsilon_var = tk.StringVar(value=f"{self.agent.epsilon:.4f}")
        tk.Label(sidebar, textvariable=self.current_epsilon_var, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.random_maze_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sidebar, text="Randomize Maze per Episode", variable=self.random_maze_var).pack(anchor=tk.W, pady=5)
        
        tk.Label(sidebar, text="Controls", font=("Arial", 12)).pack(pady=(20, 5))
        
        self.btn_train = tk.Button(sidebar, text="Start Training", command=self.start_training, bg="#4CAF50", fg="white")
        self.btn_train.pack(fill=tk.X, pady=5)
        
        self.btn_stop = tk.Button(sidebar, text="Stop", command=self.stop_training, bg="#F44336", fg="white", state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=5)
        
        tk.Button(sidebar, text="Generate New Maze", command=self.new_maze).pack(fill=tk.X, pady=5)
        tk.Button(sidebar, text="Run Test (1 Episode)", command=self.run_test).pack(fill=tk.X, pady=5)
        
        self.step_label_var = tk.StringVar(value="Steps: 0")
        tk.Label(sidebar, textvariable=self.step_label_var, font=("Arial", 12, "bold"), fg="blue").pack(pady=5)
        
        tk.Label(sidebar, text="Model Management", font=("Arial", 12)).pack(pady=(20, 5))
        tk.Button(sidebar, text="Save Model", command=self.save_model).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Load Model", command=self.load_model).pack(fill=tk.X, pady=2)
        
        # Main Area
        main_area = tk.Frame(self.root)
        main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Maze Canvas
        self.canvas_size = 400
        self.cell_size = self.canvas_size // self.env.maze_size[0]
        self.canvas = tk.Canvas(main_area, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Plot Area
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Rewards over Episodes")
        self.ax.set_xlabel("Total Episodes")
        self.ax.set_ylabel("Reward")
        self.line, = self.ax.plot([], [])
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=main_area)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        
    def draw_maze(self):
        self.canvas.delete("all")
        rows, cols = self.env.maze_size
        self.cell_size = self.canvas_size // max(rows, cols)
        
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                
                if self.env.maze[r, c] == 1:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                elif (r, c) == self.env.start_pos:
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text="S", fill="red")
                elif (r, c) == self.env.goal_pos:
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text="G", fill="green")
                    
        self.draw_agent()
        
    def draw_agent(self):
        self.canvas.delete("agent")
        r, c = self.env.agent_pos
        x1, y1 = c * self.cell_size + 2, r * self.cell_size + 2
        x2, y2 = x1 + self.cell_size - 4, y1 + self.cell_size - 4
        self.canvas.create_oval(x1, y1, x2, y2, fill="blue", tags="agent")
        
    def new_maze(self):
        self.env.mode = 'random'
        self.env.generate_new_maze()
        self.env.reset()
        self.draw_maze()
        
    def start_training(self):
        self.training = True
        self.stop_event.clear()
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        # self.rewards_history = [] # Don't reset history
        self.update_plot()
        
        # Update parameters
        try:
            self.agent.epsilon_decay = float(self.decay_var.get())
            self.agent.epsilon_end = float(self.min_epsilon_var.get())
        except:
            pass
            
        threading.Thread(target=self.training_loop, daemon=True).start()
        
    def stop_training(self):
        self.stop_event.set()
        self.testing = False
        
    def training_loop(self):
        episodes = int(self.episodes_var.get())
        episodes_run = 0
        
        for e in range(episodes):
            if self.stop_event.is_set():
                break
            
            episodes_run += 1
            
            if self.random_maze_var.get():
                self.env.generate_new_maze()
                
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.act(state, self.env.get_valid_actions())
                next_state, reward, done = self.env.step(action)
                self.agent.cache(state, action, reward, next_state, done)
                self.agent.learn()
                
                state = next_state
                total_reward += reward
                
                if total_reward < -200:
                    break
            
            if e % 10 == 0:
                self.agent.update_target_network()
                current_ep_num = self.total_episodes_trained + e
                print(f"Episode {current_ep_num}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")
                
            self.agent.update_epsilon()
            self.current_epsilon_var.set(f"{self.agent.epsilon:.4f}")
            self.rewards_history.append(total_reward)            
            # Update GUI periodically
            if e % 5 == 0:
                self.root.after(0, self.update_plot)
                self.root.after(0, self.draw_maze) # Show progress
                
                
        self.total_episodes_trained += episodes_run
        self.root.after(0, self.training_finished)
        
    def training_finished(self):
        self.training = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.update_plot()
        print("Training Finished")

    def update_plot(self):
        self.line.set_data(range(len(self.rewards_history)), self.rewards_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.plot_canvas.draw()
        
    def run_test(self):
        if self.training or self.testing:
            return
            
        self.testing = True
        self.testing = True
        self.test_steps = 0
        self.step_label_var.set("Steps: 0")
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        self.env.reset()
        self.draw_maze()
        self.root.after(100, self.step_test)
        
    def step_test(self):
        if not self.testing:
            self.test_finished()
            return

        self.test_steps += 1
        self.step_label_var.set(f"Steps: {self.test_steps}")
        state = self.env.get_local_view()
        # Greedy action
        
        # Hack: temporarily force epsilon low
        old_eps = self.agent.epsilon
        self.agent.epsilon = 0
        action = self.agent.act(state, self.env.get_valid_actions())
        self.agent.epsilon = old_eps
        
        _, _, done = self.env.step(action)
        self.draw_agent()
        
        if not done and self.test_steps < 200:
            self.root.after(200, self.step_test)
        else:
            self.test_finished()

    def test_finished(self):
        self.testing = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        print(f"Test Run Finished. Steps: {self.test_steps}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pth",
                                                 filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if file_path:
            try:
                self.agent.save(file_path)
                print(f"Model saved to {file_path}")
            except Exception as e:
                print(f"Error saving model: {e}")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if file_path:
            try:
                self.agent.load(file_path)
                print(f"Model loaded from {file_path}")
                self.current_epsilon_var.set(f"{self.agent.epsilon:.4f}")
            except Exception as e:
                print(f"Error loading model: {e}")

    def on_canvas_click(self, event):
        if self.training or self.testing:
            return
            
        # Determine cell based on click position
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        
        # Check bounds
        if 0 <= r < self.env.maze_size[0] and 0 <= c < self.env.maze_size[1]:
            # Don't edit Start or Goal
            if (r, c) == self.env.start_pos or (r, c) == self.env.goal_pos:
                return
                
            # Toggle Wall
            if self.env.maze[r, c] == 1:
                self.env.maze[r, c] = 0
            else:
                self.env.maze[r, c] = 1
                
            self.draw_maze()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.mainloop()
