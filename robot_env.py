import threading
import genesis as gs
import numpy as np
from genesis import options


class RobotEnv:
    def __init__(self, env_dict: dict, robot_dict: dict):
        """
        Initialize the robot environment.

        Args:
            env_dict (dict): Environment configuration dictionary.
            robot_dict (dict): Robot configuration dictionary.
        """
        # Set up the environment and robot
        self._initialize_env(env_dict)
        self._initialize_robot(robot_dict)
        self.running = False  # Thread running flag

    def _initialize_env(self, env_dict: dict):
        """
        Populate the environment with objects.

        Args:
            env_dict (dict): Environment configuration dictionary.
        """
        self.dt = env_dict["options"]["dt"]
        self.fps = env_dict["options"]["max_FPS"]
        self.substeps = env_dict["options"]["substeps"]
        # Configure options
        viewer_options = options.ViewerOptions(max_FPS=self.fps)
        sim_options = options.SimOptions(dt=self.dt, substeps=self.substeps)
        # Initialize scene
        self.robot = None
        self.scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=sim_options,
            show_viewer=True,
        )

        # Initialize entities (except robots)
        for _, morph in env_dict["objects"].items():
            self.scene.add_entity(morph=morph)

    def _initialize_robot(self, robot_dict: dict):
        """
        Set up the robot using the provided configuration.

        Args:
            robot_dict (dict): Robot configuration dictionary.
        """
        # Add the robot entity to the scene
        self.robot = self.scene.add_entity(gs.morphs.MJCF(**robot_dict["morph"]))
        self.scene.build()

        # Configure robot joint properties
        self.robot.set_dofs_kp(robot_dict["joints"]["kp"])
        self.robot.set_dofs_kv(robot_dict["joints"]["kv"])

        # Cache joint indices for efficient operations
        self.dofs_idx = np.array(
            [
                self.robot.get_joint(name).dof_idx_local
                for name in robot_dict["joints"]["names"][:-2]
            ]
        )

        # Set initial joint positions
        self.robot.control_dofs_position(robot_dict["joints"]["init_config"])

    def set_qpos(self, qpos, idx=None):
        """
        Set joint positions for the robot.

        Args:
            qpos (array-like): Target joint positions.
            idx (array-like, optional): Indices of the joints to set (default: self.dofs_idx).

        Raises:
            IndexError: If lengths of `qpos` and `idx` do not match.
        """
        if idx is None:
            idx = self.dofs_idx

        if len(qpos) != len(idx):
            raise IndexError(
                f"Joint configuration and indices lengths differ ({len(qpos)} != {len(idx)})."
            )

        self.robot.control_dofs_position(qpos, idx)
        print("Joint positions set to:", list(qpos))

    def get_qpos(self, idx=None):
        """
        Get current joint positions of the robot.

        Args:
            idx (array-like, optional): Indices of the joints to retrieve (default: self.dofs_idx).

        Returns:
            np.ndarray: Current joint positions.
        """
        if idx is None:
            idx = self.dofs_idx

        return self.robot.get_qpos()[idx]

    def run_thread(self):
        """
        Start the simulation loop in a separate thread.
        """
        if self.running:
            print("Simulation thread is already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()

    def _simulation_loop(self):
        """
        Main simulation loop.
        """
        while self.running:
            try:
                self.scene.step()
            except gs.GenesisException as e:
                print(f"Simulation interrupted: {e}")
                self.running = False

    def stop_thread(self):
        """
        Stop the simulation loop and join the thread.
        """
        if not self.running:
            print("Simulation is not running.")
            return

        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
            print("Simulation thread stopped.")
