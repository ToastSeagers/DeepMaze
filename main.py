from environment import MazeEnv
from agent import DQNAgent
import matplotlib.pyplot as plt

def main():
    env = MazeEnv(mode='static')
    agent = DQNAgent(input_dim=27, output_dim=4) # 5x5 flattened + 2 goal dir
    
    episodes = 500
    rewards_history = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, env.get_valid_actions())
            next_state, reward, done = env.step(action)
            agent.cache(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if total_reward < -200: # Force stop if stuck
                break
        
        if e % 10 == 0:
            agent.update_target_network()
            print(f"Episode {e}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
        agent.update_epsilon()
        rewards_history.append(total_reward)

    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.savefig('training_plot.png')
    print("Training finished. Plot saved.")

if __name__ == "__main__":
    main()
