import pygame
import random
import numpy as np
import time
from collections import deque

# --- CONFIG ---
MAZE_SIZE = (6, 6)
TILE_SIZE = 100
WIDTH, HEIGHT = MAZE_SIZE[1] * TILE_SIZE, MAZE_SIZE[0] * TILE_SIZE
WHITE, BLACK, BLUE, GREEN, PINK, RED = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 182, 193), (255, 0, 0)
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
REGEN_INTERVAL = 15  # Maze reset time in seconds
TRAIN_MODE_COLOR = (0, 255, 0)  # Green
TEST_MODE_COLOR = (255, 0, 0)    # Red

# --- ENVIRONMENT ---
class MazeEnv:
    def __init__(self):
        self.goal = (MAZE_SIZE[0]-1, MAZE_SIZE[1]-1)  # Goal position - bottom-right
        self.agent_pos = (0, 0)  # Start position - top-left
        self.reset_maze() # creates initial maze layout
    def reset_maze(self):
        while True:
            self.maze = np.zeros(MAZE_SIZE)  # Initialize empty maze
            wall_count = random.randint(5, 10)  # Random number of walls
            for _ in range(wall_count):
                x, y = random.randint(0, MAZE_SIZE[0]-1), random.randint(0, MAZE_SIZE[1]-1)
                # Make sure the start (0, 0) and goal cells are not walled off
                if (x, y) != (0, 0) and (x, y) != self.goal:
                    self.maze[x][y] = 1
            self.maze[self.goal[0]][self.goal[1]] = 0  # Ensure goal is not blocked
            if self.path_exists():
                break
    def path_exists(self):
        # Check if there is a valid path from start to goal
        visited = np.zeros(MAZE_SIZE, dtype=bool)
        queue = deque([self.agent_pos])
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < MAZE_SIZE[0] and 0 <= ny < MAZE_SIZE[1]:
                    if not visited[nx][ny] and self.maze[nx][ny] == 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
        return False
    def move_agent(self, action):
        x, y = self.agent_pos
        moves = {'UP': (x-1, y), 'DOWN': (x+1, y), 'LEFT': (x, y-1), 'RIGHT': (x, y+1)}
        nx, ny = moves[action]
        if 0 <= nx < MAZE_SIZE[0] and 0 <= ny < MAZE_SIZE[1] and self.maze[nx][ny] == 0:
            self.agent_pos = (nx, ny)
    def get_state(self):
        # returns agent's current position
        return self.agent_pos
    def is_terminal(self):
        # returns true if the agent has reached the goal
        return self.agent_pos == self.goal
    def reset_agent(self):
        # restes the agent position to start
        self.agent_pos = (0, 0)

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}             #This dictionary stores Q-values for state-action pairs: Q[state, action]
        self.alpha = alpha            #Learning rate	 - Controls how much new information overrides old (0 to 1).
        self.gamma = gamma            #Discount factor	 - Prioritizes future rewards (closer to 1 = long-term focus).
        self.epsilon = epsilon        #Exploration rate	 - Probability of choosing a random action (exploration).

    def get_q(self, state, action):   #Returns the Q-value for a given (state, action) pair
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):   #This chooses the next action based on an ε-greedy policy
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = [self.get_q(state, a) for a in ACTIONS]
        return ACTIONS[np.argmax(q_values)]
    
    def update(self, state, action, reward, next_state):
        #This is the core learning function
        old_q = self.get_q(state, action) #Get current Q-value
        future_q = max([self.get_q(next_state, a) for a in ACTIONS])  #Estimate future reward
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q) #Compute updated Q-value
        self.q_table[(state, action)] = new_q #store the updated Q-value

# --- DRAW FUNCTIONS ---
def draw(env, screen):
    screen.fill(WHITE)
    for x in range(MAZE_SIZE[0]):
        for y in range(MAZE_SIZE[1]):
            rect = pygame.Rect(y*TILE_SIZE, x*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if env.maze[x][y] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    # Draw goal first (so agent appears on top)
    gx, gy = env.goal
    goal_color = BLUE if env.is_terminal() else GREEN
    pygame.draw.rect(screen, goal_color, pygame.Rect(gy*TILE_SIZE, gx*TILE_SIZE, TILE_SIZE, TILE_SIZE))
    # Draw agent with different color when at goal
    ax, ay = env.agent_pos
    agent_color = RED if env.is_terminal() else PINK  # Red when at goal
    pygame.draw.circle(screen, agent_color, (ay*TILE_SIZE + TILE_SIZE//2, ax*TILE_SIZE + TILE_SIZE//2), TILE_SIZE//4)
    # Add white border to agent when at goal for better visibility
    if env.is_terminal():
        pygame.draw.circle(screen, WHITE, (ay*TILE_SIZE + TILE_SIZE//2, ax*TILE_SIZE + TILE_SIZE//2), TILE_SIZE//4, 2)

def draw_stats(screen, font, episode, reward_total, step_count, train_mode):
    texts = [
        f"Episode: {episode}",
        f"Reward: {reward_total}",
        f"Steps: {step_count}",
        f"Mode: {'TRAIN' if train_mode else 'TEST'}"]
    for i, text in enumerate(texts):
        color = TRAIN_MODE_COLOR if train_mode and i == 3 else TEST_MODE_COLOR if i == 3 else RED
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (10, 10 + i * 30))

# --- MAIN GAME LOOP ---
def run_game(initial_mode=True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Time-Loop Maze with AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
    env = MazeEnv()
    agent = QLearningAgent()
    last_regen_time = time.time()
    episode = 1
    reward_total = 0
    step_count = 0
    train_mode = initial_mode
    running = True

    while running:
        if time.time() - last_regen_time > REGEN_INTERVAL:
            env.reset_maze()
            env.reset_agent()
            last_regen_time = time.time()
            # print(f"[Maze regenerated]")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:  # Press T to toggle mode
                    train_mode = not train_mode
                    print(f"Switched to {'TRAIN' if train_mode else 'TEST'} mode")
                    if not train_mode:
                        # When switching to test mode, disable exploration
                        agent.epsilon = 0.0
        state = env.get_state()
        action = agent.choose_action(state)
        env.move_agent(action)
        next_state = env.get_state()
        reward = 10 if env.is_terminal() else -1
        if train_mode:
            agent.update(state, action, reward, next_state)
        reward_total += reward
        step_count += 1
        if env.is_terminal():
            print(f"Episode {episode} | Reward: {reward_total} | Steps: {step_count}")
            pygame.display.flip()         # Ensure final frame is shown
            time.sleep(1)                 # Pause for 1 second
            episode += 1
            reward_total = 0
            step_count = 0
            env.reset_agent()
        draw(env, screen)
        draw_stats(screen, font, episode, reward_total, step_count, train_mode)
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()

# --- ENTRY POINT ---
if __name__ == "__main__":
    run_game()






# The draw() function:
# Clears the screen.
# Draws the maze grid (walls and open cells).
# Highlights the goal cell.
# Draws the agent (with a color change if it reaches the goal).



# This draw_stats() function is responsible for displaying game statistics and mode info 
# (like episode number, reward, steps, and mode) on the Pygame screen.
# screen: Pygame surface to draw the text on.
# font: a Pygame font object used to render the text.
# episode: current training/test episode number.
# reward_total: total reward accumulated in the current episode.
# step_count: number of steps taken in the episode.
# train_mode: True if in training mode, False if testing.



# The run_game(initial_mode=True) function is the main game loop for your Time-Loop Maze game. 
# It initializes Pygame, creates the maze and agent, handles training/testing, regenerates the 
# maze periodically, and updates the screen with game visuals and stats.





