from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO



class A1_23Cfg( A1RoughCfg ):

    class init_state( A1RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.805] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'waist_yaw_joint' : 0,
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'left_wrist_roll_joint' : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
           'right_wrist_roll_joint' : 0.
        }
    
    class env(A1RoughCfg.env):
        num_observations = 80
        num_privileged_obs = 83
        num_actions = 23    

    class control(A1RoughCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist_yaw': 250,
                     'shoulder': 100,
                     "elbow": 50,
                     'wrist': 50,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist_yaw': 6,
                     'shoulder': 2,
                     "elbow": 2,
                     'wrist': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(A1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/Arcus_23dof.urdf'
        name = "a1_23dof"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards(A1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.802
        
        class scales(A1RoughCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -.10
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = -20.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

    class sim(A1RoughCfg.sim):
        class physx(A1RoughCfg.sim.physx):
            max_gpu_contact_pairs = 2**24

class A1_23RoughCfgPPO(A1RoughCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
        
    class algorithm(A1RoughCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(A1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'a1_23dof'