import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("/Users/arushpimpalkar/work/timepass/RL_Arush/pendulum.xml")
data = mujoco.MjData(model)

# Optional: initial condition (slight offset)
data.qpos[0] = 0.2   # radians

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Apply torque (try zero first)
        data.ctrl[0] = 0.0  

        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        time.sleep(model.opt.timestep)
