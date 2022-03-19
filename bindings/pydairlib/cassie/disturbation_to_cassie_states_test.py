from math import radians
from disturbation_to_cassie_states import DisturbationToCassieStates
import numpy as np

def main():

    qstate_gt = np.array(
        [ 0.997335036534076,
          0.9994677976315247,
          0.014342509822522855,
          0.025448715241811547,
          0.014518154225711895,
          -0.06798602851909162,
          0.022089323345553236,
          0.5418787133206028,
          -1.4322107499893533,
          -1.5135615861228986,
          -0.06430447462816609,
          -0.1038198197241002,
          0.27563717282320116,
          -0.8518387062727452,
          -1.4518001640196492,
          0.014593755891002105,
          0.026179535297769667,
          0.133482689872097,
          1.340795890425404,
          -0.21594454877448516,
          -0.06194733767907466,
          -0.8569726562500002,
          0.798564453125,
          1.049652099609375,
          -0.8643951416015626,
          -0.5293399047851562,
          -3.2833984375000007,
          -1.75404296875,
          -1.482635498046875,
          -0.962982177734375,
          1.2090745544433594,
          0.5846620762175988,
          0.2172065010947209,
          -10.793310141486685,
          0.0025646241297463063,
          1.6490772838766856,
          -1.513272047249185,
          0.01227184630308513,
          1.0648223512907422,
          -1.4511458253398166,
          -0.15511701602518552,
          0.8035290690398684,
          0.9643420559888644,
          2.5922085155034535,
          -2.8707355816460116,
          3.6387613895575095,
          0.3826834323650898,
          0.9238795325112867,
          1.0])

    qpos_init_drake = np.hstack((qstate_gt[1:5], np.array([0, 0, qstate_gt[0]]), qstate_gt[5:9], qstate_gt[34], qstate_gt[35], 0.0, qstate_gt[9], qstate_gt[11:15], qstate_gt[37:39], 0.0, qstate_gt[14]))
    qvel_init_drake = np.hstack((qstate_gt[18:21], qstate_gt[15:18], qstate_gt[21:25], qstate_gt[40:42], 0.0, qstate_gt[25], qstate_gt[26:30], qstate_gt[44:46], 0.0, qstate_gt[30]))

    initial_state_in_drake = np.hstack((qpos_init_drake, qvel_init_drake))

    disturbation_generator = DisturbationToCassieStates()

    disturbation_generator.get_initial_states(initial_state_in_drake)

    disturbed_states, is_success = disturbation_generator.add_disturbation_to_left_foot_point_at_z_direction(0)

    disturbation_generator.visualize_left_leg(initial_state_in_drake)

    import pdb; pdb.set_trace()

    disturbation_generator.visualize_left_leg(disturbed_states)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()