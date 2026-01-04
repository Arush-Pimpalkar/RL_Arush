# Overview

 - Implementing simple pendulum using MuJoCo.
 - Added gaussian noise at action using ActionGaussianNoiseWrapper  
 - Tried various methods to attenuate noise at f > 10 hz
 - Compared with a simple SAC trained without noise based rewards 

### Simple SAC (Baseline):
 - Standard Soft Actor-Critic algorithm without any additional penalty terms
 - Trains directly on a noisy environment with Gaussian action noise (SNR = 10)
 - Uses default SAC hyperparameters (learning rate 3e-4, gamma 0.99)
 - **Reward Function:** `R = R_base` (no modifications)

## Methods tried: 

### DeepMind Style (Improved DM):
 - Applies a sigmoid function for energy available over 10 hz. 
 - Using the base reward function showed better results.
 - **Reward Function:** `R = R_base - (λ_penalty × sigmoid_score)`
 
![DeepMind Style Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/deepmind_style_vs_simple_sac.png)

### CombinedPenaltyWrapper:
 - Combines both high-frequency filtering and action rate penalties for better noise attenuation
 - Uses a low-pass filter (5Hz cutoff) to extract high-frequency components via residual method
 
![Combined Penalty Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/combined_penalty_vs_simple_sac.png)

### HighFrequencyPenaltyWrapper:
 - Uses a 2nd-order Butterworth high-pass filter to detect high-frequency action components above 10Hz
 - Applies a quadratic penalty to the filtered high-frequency content
 - Designed to directly penalize frequencies above the desired control bandwidth
 - **Reward Function:** `R = R_base - λ_hf × (u_hf)²` where `u_hf` is high-frequency component

### ActionRatePenaltyWrapper: 
 - Penalizes rapid changes in consecutive actions without using any filtering
 - Simple and computationally efficient approach to reduce high-frequency content
 - Tracks previous action and penalizes the squared difference with current action
 - **Reward Function:** `R = R_base - λ_rate × (u_t - u_{t-1})²` where `λ_rate = 0.1`
 
![Action Rate Penalty Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/action_rate_penalty_vs_simple_sac.png)



## Results

The **Combined Penalty**, **Improved DM**, and **DLS** methods are most effective at reducing high-frequency energy content while maintaining control performance. The DLS method is unique in penalizing state frequencies rather than action frequencies, providing a more direct approach to controlling system response characteristics.

### Overall Comparison
![All Methods Comparison](https://raw.githubusercontent.com/Arush-Pimpalkar/RL_Arush/main/plots/all_methods_comparison.png)