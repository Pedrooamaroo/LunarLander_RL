import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D

class CustomLunarLander(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.MAX_FUEL_CAPACITY = 1000.0
        self.current_fuel = 0
        
        # THEME COLORS
        self.bg_color = np.array([15, 15, 60], dtype=np.uint8)     # Deep Space Blue
        self.ground_color = np.array([100, 130, 200], dtype=np.uint8) # Steel Blue
        
        # Observation Space Expantion (8 -> 9 dimensions) - fuel level adeed as normalized value [0,1]
        low = np.append(env.observation_space.low, 0.0)
        high = np.append(env.observation_space.high, 1.0)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        #d Reset
        obs, info = self.env.reset(**kwargs)

        # Physics Randomization
        new_gravity = np.random.uniform(-8.0, -12.0)
        self.env.unwrapped.world.gravity = (0, new_gravity)
        self.env.unwrapped.enable_wind = True
        self.env.unwrapped.wind_power = np.random.uniform(5.0, 20.0)
        self.env.unwrapped.turbulence_power = np.random.uniform(0.5, 1.5)
        # Initialize wind_idx and torque_idx
        if not hasattr(self.env.unwrapped, 'wind_idx'):
            self.env.unwrapped.wind_idx = 0
        if not hasattr(self.env.unwrapped, 'torque_idx'):
            self.env.unwrapped.torque_idx = 0

        # Fuel Randomization
        self.current_fuel = np.random.uniform(0.5 * self.MAX_FUEL_CAPACITY, self.MAX_FUEL_CAPACITY)

        # Position Shift
        new_x = np.random.uniform(2.0, 18.0)
        
        lander = self.env.unwrapped.lander
        initial_x, initial_y = lander.position
        
        # Set Position
        lander.position = (float(new_x), float(initial_y))
        lander.Awake = True 
        
        # Refresh Observation
        obs, _, _, _, info = self.env.step(0)

        # Update Info
        info['fuel_remaining'] = self.current_fuel
        info['fuel_percentage'] = (self.current_fuel / self.MAX_FUEL_CAPACITY) * 100
        info['start_x'] = obs[0]
        info['wind_enabled'] = self.env.unwrapped.enable_wind

        # Extend Observation with Fuel
        obs_extended = np.append(obs, self.current_fuel / self.MAX_FUEL_CAPACITY)

        return obs_extended, info

    def step(self, action):
        if self.current_fuel <= 0:
            action = 0
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Fuel Consumption
        fuel_cost = 0
        if action == 2: fuel_cost = 5.0; reward -= 0.1
        elif action == 1 or action == 3: fuel_cost = 1.0
        
        self.current_fuel -= fuel_cost
        
        # Temporal Penalty (encourages faster landing)
        reward -= 0.005
        
        # Dynamic Mass (fuel affects lander weight)
        try:
            self.env.unwrapped.lander.fixtures[0].density = 5.0 + (self.current_fuel / self.MAX_FUEL_CAPACITY)
        except:
            pass
        
        # Terminate if out of fuel
        if self.current_fuel <= 0:
            terminated = True
            reward -= 100  # Heavy penalty for running out of fuel
        
        info['fuel_remaining'] = max(0, self.current_fuel)
        info['fuel_percentage'] = (max(0, self.current_fuel) / self.MAX_FUEL_CAPACITY) * 100
        
        # Extend Observation with Fuel
        obs_extended = np.append(obs, max(0, self.current_fuel) / self.MAX_FUEL_CAPACITY)
        
        return obs_extended, reward, terminated, truncated, info

    # Renderization
    def render(self):
        frame = self.env.render()
        is_black = np.all(frame == [0, 0, 0], axis=-1)
        is_white = np.all(frame > [250, 250, 250], axis=-1)
        frame[is_black] = self.bg_color
        frame[is_white] = self.ground_color
        return frame
    
# Creating environment
def make_custom_env(render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode, continuous=False)
    env = CustomLunarLander(env)
    return env