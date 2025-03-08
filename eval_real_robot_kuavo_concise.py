
# %%
import time

import cv2
import torch
import dill
import hydra
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_kuavo_Task7_SuZhou import KuavoEnv
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import rospy
input="/home/leju-ali/hx/kuavo/Task8-SuZhou/data/outputs/2025.02.23/19.42.10_train_diffusion_unet_image_Kuavo_Task8_SuZhou_task/checkpoints/latest.ckpt"
output="/home/leju-ali/hx/kuavo/Task8-SuZhou/data/outputs/2025.02.23/19.42.10_train_diffusion_unet_image_Kuavo_Task8_SuZhou_task/checkpoints/output"

vis_camera_idx=1
steps_per_inference=6
frequency=15
command_latency=0.01
OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    # load checkpoint
    steps_per_inference=6
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    rospy.init_node("test")
    with KuavoEnv(
        frequency=10,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=500,

        img_buffer_size=30,
        robot_state_buffer_size=500,

        video_capture_resolution=(640, 480),

        
        ) as env:
            print("waiting for the obs buffer to be ready ......")

            env.obs_buffer.wait_buffer_ready()
            obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            print('##################################################################')
            print(f"{camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}, \n"
                f"{camera_obs_timestamps['img02'][0]:.10f}, {camera_obs_timestamps['img02'][1]:.10f}, \n"
                f"{robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}, \n"
                f"{robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}\n")
            print('##################################################################')
            for k, v in obs.items():
               print(f'{k=}, \t {v.shape=}')
            
            while True:
                # ========= human control loop ==========
                print("skip Human in control!")
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    while True:
                        # =============
                        # get obs
                        # =============
                        
                        obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
                        print('##################################################################')
                        print(f"{camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}, \n"
                            f"{camera_obs_timestamps['img02'][0]:.10f}, {camera_obs_timestamps['img02'][1]:.10f}, \n"
                            f"{robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}, \n"
                            f"{robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}\n")
                        print('##################################################################')
                        # =============
                        # show sense img
                        # =============
                        imgs = []
                        for k, v in obs.items():
                            if "img" in k:
                                img = obs[k][-1]
                                img = cv2.resize(img, (480, 640))
                                img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                imgs.append(img01_bgr)
                        
                        # run inference
                        with torch.no_grad():
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            s = time.time()
                            result = policy.predict_action(obs_dict)
                            inference_time = time.time() - s
                            
                            print('Inference latency:', inference_time)
                          
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
           

                        # execute actions
                        numpy_action = action.to("cpu").numpy()
                        env.exec_actions(
                            actions=numpy_action[::2,:],
                        )
                        print(f"Submitted {len(action)} steps of actions.")
  
                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.close()
                    exit(0)
                print("Stopped.")


# %%
if __name__ == '__main__':
    main()
