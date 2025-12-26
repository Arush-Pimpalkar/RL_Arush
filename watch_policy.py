import time
from stable_baselines3 import SAC
from pendulum_env import MuJoCoPendulumEnv

# ---------- TRAIN ----------
env = MuJoCoPendulumEnv()

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    verbose=1,
)

model.learn(total_timesteps=40_000)
model.save("sac_pendulum")

# ---------- WATCH ----------
env = MuJoCoPendulumEnv(render_mode="human")
model = SAC.load("sac_pendulum", env=env)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    time.sleep(env.dt)
