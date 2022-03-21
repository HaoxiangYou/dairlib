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
                           "examples/Cassie/urdf/cassie_v2_conservative.urdf", False, False)
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
        self.ik_solver = InverseKinematics(self.plant, self.context, with_joint_limits=False)

    def get_initial_states(self, states_in_drake):

        self.initial_states = states_in_drake 
        self.initial_pos = self.initial_states[:self.num_pos]
        self.initial_vel = self.initial_states[-self.num_vel:]

    def add_disturbation_to_left_foot_point_at_z_direction(self,disturbation):
        """
        Add the disturbation to the point where the left foot end up being
        """

        # Get the Initial position of toe by Forward kinematics

        self.plant.SetPositions(self.context, self.initial_pos)

        world_frame = self.plant.world_frame()

        toe_left_frame = self.plant.GetBodyByName("toe_left").body_frame()

        toe_left_anchor_coordinates_in_body_frame = np.array([0.04885482, 0.00394248, 0.01484])

        toe_left_anchor_coordinates_in_world_frame = self.plant.CalcPointsPositions(
                                                    self.context, toe_left_frame, toe_left_anchor_coordinates_in_body_frame, world_frame)

        # Add disturbation to the desired position

        disturbed_toe_left_anchor_coordinates_in_world_frame = np.copy(toe_left_anchor_coordinates_in_world_frame)

        disturbed_toe_left_anchor_coordinates_in_world_frame[2] += disturbation

        # Set disturbed position as a linear equality constraints

        self.ik_solver.AddPositionConstraint(toe_left_frame, toe_left_anchor_coordinates_in_body_frame, world_frame,
                                                disturbed_toe_left_anchor_coordinates_in_world_frame, disturbed_toe_left_anchor_coordinates_in_world_frame)

        # Fixed the body and right leg

        configuration_index_need_to_be_fixed = self.get_pos_fixed_index()

        A = np.eye(configuration_index_need_to_be_fixed.shape[0])

        b = self.initial_pos[configuration_index_need_to_be_fixed]

        delta = np.ones(configuration_index_need_to_be_fixed.shape[0]) * 1e-2

        self.ik_solver.prog().AddLinearConstraint(A, b - delta, b + delta, self.ik_solver.q()[configuration_index_need_to_be_fixed])

        #Add Constraints for rods in left leg

        left_thigh_rod_point_in_body_frame, left_thigh_frame = LeftRodOnThigh(self.plant)
        left_heel_rod_point_in_body_frame, left_heel_frame = LeftRodOnHeel(self.plant)
        
        rod_length = 0.5

        delta = 1e-2

        self.ik_solver.AddPointToPointDistanceConstraint(left_thigh_frame, left_thigh_rod_point_in_body_frame,
                                                        left_heel_frame, left_heel_rod_point_in_body_frame,
                                                        rod_length - delta, rod_length + delta)
        
        # Add the Cost for IK, where the cost is defined as the change of disturbed states and new states.
        # The only things have degree of freedom are joints in left leg which are all units in radians

        Q = np.eye(self.num_pos)

        b = - Q @ self.initial_pos

        c = self.initial_pos.T @ Q @ self.initial_pos / 2

        self.ik_solver.prog().AddQuadraticCost(Q, b, self.ik_solver.q()) 

        # Solve IK

        self.ik_solver.prog().SetInitialGuess(self.ik_solver.q(), self.initial_pos)

        snap_solver = SnoptSolver()

        result = snap_solver.Solve(self.ik_solver.prog())
        is_success = result.is_success()

        q_pos = result.GetSolution()
        q_vel = self.initial_vel

        # Check the feasible of diturbed states
        import pdb; pdb.set_trace()

        return np.hstack((q_pos, q_vel)), is_success

    def visualize_left_leg(self, states_in_drake):

        self.drake_to_mujoco_converter.visualize_entire_leg(states_in_drake)

    def get_pos_fixed_index(self,):
        """
        Get index for parameters need to be fixed
        """
        index = []
        
        key_lists = [
                    "base_qw",
                    "base_qx",
                    "base_qy",
                    "base_qz",
                    "base_x",
                    "base_y",
                    "base_z",
                    "hip_roll_right", 
                    "hip_pitch_right", 
                    "hip_yaw_right",
                    "knee_right",
                    "knee_joint_right",
                    "ankle_joint_right",
                    "ankle_spring_joint_right",
                    "toe_right"
                    ]

        for key in key_lists:
                # Base may also be affected

                index.append(self.pos_map[key])

        return np.array(index, dtype=int)