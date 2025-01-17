import os

os.environ["PYOPENGL_PLATFORM"] = "glx"  # Set OpenGL platform before importing genesis
import genesis as gs
import torch
from robot_env import RobotEnv


if __name__ == "__main__":
    device = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(backend=device, logging_level="warning")

    env_dict = {
        "options": {
            "dt": 0.01,
        },
        "objects": {"plane": gs.morphs.Plane()},
    }
    robot_dict = {
        "morph": {
            "file": "xml/franka_emika_panda/panda.xml",
            "pos": (0, 0, 0),
            "euler": (0, 0, 0),
        },
        "joints": {
            "names": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint6",
                "finger_joint1",
                "finger_joint2",
            ],
            "init_config": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
            "kv": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        },
    }

    env = RobotEnv(env_dict, robot_dict)
    env.run_thread()

    while env.running:
        try:
            cmd = input("Enter 'u' or 'l' to move (`q` to quit) : ")
            if cmd.lower() == "u":
                qpos = env.get_qpos()
                qpos[0] += 0.2
                env.set_qpos(qpos)
            elif cmd.lower() == "l":
                qpos = env.get_qpos()
                qpos[0] -= 0.2
                env.set_qpos(qpos)
            elif cmd.lower() == "q":
                break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break

    env.stop_thread()
