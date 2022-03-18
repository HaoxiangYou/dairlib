from operator import index
import numpy as np
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph, CoulombFriction
from pydairlib.common import FindResourceOrThrow
from pydairlib.cassie.cassie_utils import *
from scipy.spatial.transform import Rotation as R
from pydairlib.multibody import *
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.solvers.snopt import *
from pydrake.geometry import SceneGraph, DrakeVisualizer, HalfSpace, Box
from pydairlib.multibody import MultiposeVisualizer
from drake_to_mujoco_converter import DrakeToMujocoConverter


class DisturbationToCassieStates():
    def __init__(self, drake_sim_dt=5e-5):
        
        # Initialize a Drake object
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, drake_sim_dt)
        AddCassieMultibody(self.plant, self.scene_graph, True,
                           "examples/Cassie/urdf/cassie_v2.urdf", False, False)
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()

        # Create a map from index to a "named" joint

        self.pos_map = makeNameToPositionsMap(self.plant)
        self.vel_map = makeNameToVelocitiesMap(self.plant)
        self.act_map = makeNameToActuatorsMap(self.plant)

        self.num_pos = self.plant.num_positions()

        self.num_vel = self.plant.num_velocities()

        # Adding DrakeToMujocoConverte, used for visualization and convert states back forward between mujoco and drake
        self.drake_to_mujoco_converter = DrakeToMujocoConverter()

        # Initial states we want to disturb in drake
        self.initial_states = None

        # Initialize ik solver
        self.ik_solver = InverseKinematics(self.plant, self.context, with_joint_limits=True)

    def get_initial_states(self, states_in_drake):

        self.initial_states = states_in_drake 
        self.initial_pos = self.initial_states[:self.num_pos]
        self.initial_vel = self.initial_states[-self.num_vel:]

    def disturbation_to_single_position_state(self, state_index, disturbation):

        # Solve the IK for the single leg

        # Check which leg is to be disturbed

        # Check if the other leg is on the ground 

        # Set constraint for disturbed state to equal disturbed value 

        # If the other leg is not on the ground then we need add constraints that the disturbed leg should touched on ground 

        # Otherwise make sure the disturbed leg is above the ground 

        # Set the cost to be the quadratic error of each independent joint(motor). The units are all in radians
        # TODO think of how to combine the spring and other joints' error that may not be units of radians

        # Solve the desired leg

        # Go back to solve entire system 
    
        # First fixed the solved leg with linear equality constrained

        # If the other leg is previous on the ground, give a linear constraint equality constraints that the leg is still on the ground

        # Otherwise make sure the other leg is above the ground

        # Similar to previous solver, where the cost is the difference between the solved states and previous states
        
        value_after_disturbation = self.initial_states[state_index] + disturbation

        self.ik_solver.prog().AddLinearEqualityConstraint(self.ik_solver.q()[state_index], value_after_disturbation)

        Q = self.get_Q(which_leg="left")

        b = -Q @ self.initial_pos

        c = self.initial_pos.T @ Q @ self.initial_pos / 2

        self.ik_solver.prog().AddQuadraticCost(Q, b, c , self.ik_solver.q())

        snopt_solver = SnoptSolver()

        result = snopt_solver.Solve(self.ik_solver.prog())

        q_pos = np.copy(self.initial_pos)

        left_leg_index = self.get_index_for_certain_leg(which_leg="left")

        q_pos[left_leg_index] = result.GetSolution()[left_leg_index]

        import pdb; pdb.set_trace()

        q_vel = np.copy(self.initial_vel)

        return np.hstack((q_pos, q_vel))

    def visualize_left_leg(self, states_in_drake):

        self.drake_to_mujoco_converter.visualize_entire_leg(states_in_drake)

    def get_Q(self, which_leg = "left"):
        """
        Return the Quadratic Cost Matrix for a specified leg
        """
        
        scalar_for_angle = 1

        Q = np.zeros((self.num_pos, self.num_pos))
        
        if which_leg == "left":
            Q[self.pos_map["hip_roll_left"], self.pos_map["hip_roll_left"]] = scalar_for_angle
            Q[self.pos_map["hip_pitch_left"], self.pos_map["hip_pitch_left"]] = scalar_for_angle
            Q[self.pos_map["hip_yaw_left"], self.pos_map["hip_yaw_left"]] = scalar_for_angle
            Q[self.pos_map["knee_joint_left"], self.pos_map["knee_joint_left"]] = scalar_for_angle
            Q[self.pos_map["ankle_joint_left"], self.pos_map["ankle_joint_left"]] = scalar_for_angle

        else:
            Q[self.pos_map["hip_roll_right"], self.pos_map["hipp_roll_right"]] = scalar_for_angle
            Q[self.pos_map["hip_pitch_right"], self.pos_map["hip_pitch_right"]] = scalar_for_angle 
            Q[self.pos_map["hip_yaw_right"], self.pos_map["hip_yaw_right"]] = scalar_for_angle
            Q[self.pos_map["knee_joint_right"], self.pos_map["knee_joint_right"]] = scalar_for_angle
            Q[self.pos_map["ankle_joint_right"], self.pos_map["ankle_joint_right"]] = scalar_for_angle

        return Q

    def get_index_for_certain_leg(self, which_leg="left"):
        """
        Get index corresponding for a single leg 
        """
        index = []
        
        if which_leg == "left":
            for key in self.pos_map:
                if "left" in key:
                    index.append(self.pos_map[key])
        else :
            for key in self.pos_map:
                if "right" in key:
                    index.append(self.pos_map[key])

        return np.array(index, dtype=int)