#include <memory>

#include <gflags/gflags.h>
#include "drake/lcm/drake_lcm.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"

#include "multibody/multibody_utils.h"
#include "examples/Cassie/cassie_utils.h"

namespace dairlib {
using drake::geometry::DrakeVisualizer;
using drake::geometry::HalfSpace;
using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::RevoluteJoint;
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::Simulator;

// Simulation parameters.
DEFINE_bool(floating_base, true, "Fixed or floating base model");

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(time_stepping, false, "If 'true', the plant is modeled as a "
    "discrete system with periodic updates. "
    "If 'false', the plant is modeled as a continuous system.");
DEFINE_double(dt, 1e-4, "The step size to use for compliant, ignored for time_stepping)");

int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  const double time_step = FLAGS_time_stepping ? FLAGS_dt : 0.0;

  MultibodyPlant<double>& plant =
      *builder.AddSystem<MultibodyPlant>(time_step);

  if (FLAGS_floating_base) {
    multibody::addFlatTerrain(&plant, &scene_graph, .8, .8);
  }

  addCassieMultibody(&plant, &scene_graph, FLAGS_floating_base);

  plant.Finalize();

  std::map<std::string, int> actuatorIndexMap =
    multibody::makeNameToActuatorsMap(plant);

  for (auto const& x : actuatorIndexMap) {
    std::cout << x.first  // string (key)
              << ':'
              << x.second  // string's value
              << std::endl;
  }

  auto input_source =
      builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
          Eigen::VectorXd::Zero(plant.num_actuators()));

  builder.Connect(input_source->get_output_port(),
                  plant.get_actuation_input_port());

  builder.Connect(
    plant.get_geometry_poses_output_port(),
    scene_graph.get_source_pose_port(plant.get_source_id().value()));

  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  DrakeVisualizer<double>::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();


  // Create a context for this system:
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());


  plant.GetJointByName<RevoluteJoint>("hip_pitch_left").
      set_angle(&plant_context, .269);
  plant.GetJointByName<RevoluteJoint>("knee_left").
      set_angle(&plant_context, -.644);
  plant.GetJointByName<RevoluteJoint>("ankle_joint_left").
      set_angle(&plant_context, .792);
  plant.GetJointByName<RevoluteJoint>("toe_left").
      set_angle(&plant_context, -M_PI/3);

  plant.GetJointByName<RevoluteJoint>("hip_pitch_right").
      set_angle(&plant_context, .269);
  plant.GetJointByName<RevoluteJoint>("knee_right").
      set_angle(&plant_context, -.644);
  plant.GetJointByName<RevoluteJoint>("ankle_joint_right").
      set_angle(&plant_context, .792);
  plant.GetJointByName<RevoluteJoint>("toe_right").
      set_angle(&plant_context, -M_PI/3);

  if (FLAGS_floating_base) {
    const drake::math::RigidTransformd transform(
        drake::math::RotationMatrix<double>(), Eigen::Vector3d(0, 0, 1.2));
    plant.SetFreeBodyPose(&plant_context, plant.GetBodyByName("pelvis"),
        transform);
  }

  Simulator<double> simulator(*diagram, std::move(diagram_context));

  if (!FLAGS_time_stepping) {
    // simulator.get_mutable_integrator()->set_maximum_step_size(0.01);
    // simulator.get_mutable_integrator()->set_target_accuracy(1e-1);
    // simulator.get_mutable_integrator()->set_fixed_step_mode(true);
    simulator.reset_integrator<drake::systems::RungeKutta2Integrator<double>>(
      FLAGS_dt);
  }

  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(std::numeric_limits<double>::infinity());

  return 0;
}

}  // namespace dairlib

int main(int argc, char* argv[]) {
  return dairlib::do_main(argc, argv);
}
