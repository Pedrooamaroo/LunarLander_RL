"""
Hyperparameters Configurations
"""

GENERAL_CONFIG = {
    "total_timesteps": 2_000_000,       
    "eval_episodes": 100,       
    "success_threshold": 200,           
    "seed": 42,                         
}

# PPO (Proximal Policy Optimization) - Configurations

PPO_SETTINGS_1 = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,              
    "n_steps": 2048,                  
    "batch_size": 64,                 
    "n_epochs": 10,                     
    "gamma": 0.99,                      
    "gae_lambda": 0.95,                 
    "clip_range": 0.2,                  
    "ent_coef": 0.01,                   
    "verbose": 1,
    "seed": 42,
}

PPO_SETTINGS_2 = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,          
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.995,                 
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy_kwargs": dict(net_arch=[256, 256]), 
    "verbose": 1,
    "seed": 42,
}

# DQN (Deep Q-Network) - Configurations

DQN_SETTINGS_1 = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,            
    "buffer_size": 100_000,             
    "learning_starts": 50_000,        
    "batch_size": 32,                 
    "tau": 0.005,                      
    "gamma": 0.99,                     
    "train_freq": 4,                   
    "gradient_steps": 1,                
    "target_update_interval": 10_000,   
    "exploration_fraction": 0.1,        
    "exploration_initial_eps": 1.0,     
    "exploration_final_eps": 0.05,      
    "verbose": 1,
    "seed": 42,
}

DQN_SETTINGS_2 = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 500_000,         
    "learning_starts": 100_000,     
    "batch_size": 128,
    "gamma": 0.995,
    "tau": 0.005, 
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 500,  
    "exploration_fraction": 0.2,    
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": dict(net_arch=[256, 256]), 
    "verbose": 1,
    "seed": 42,
}

# Model names for organization

MODEL_NAMES = {
    "original_ppo_1": "Original Environment - PPO Settings 1",
    "original_ppo_2": "Original Environment - PPO Settings 2",
    "original_dqn_1": "Original Environment - DQN Settings 1",
    "original_dqn_2": "Original Environment - DQN Settings 2",
    "custom_ppo_1": "Custom Environment - PPO Settings 1",
    "custom_ppo_2": "Custom Environment - PPO Settings 2",
    "custom_dqn_1": "Custom Environment - DQN Settings 1",
    "custom_dqn_2": "Custom Environment - DQN Settings 2",
}