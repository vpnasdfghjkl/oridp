
from pathlib import Path
import numpy as np
import torch
import time
import rospy
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
if not torch.cuda.is_available():
     print(f"⚠️⚠️⚠️ running on cpu")
else:
    print(f">-< running on cuda")

from KuavoEnv import KuavoEnv
rospy.init_node("Test obs from KuavoEnv")   
task_name = "Task11_Toy"
with KuavoEnv(
        frequency=10,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=500,
        img_buffer_size=30,
        robot_state_buffer_size=500,
        video_capture_resolution=(640, 480),
        ) as env:
        is_just_img = False
        print("waiting for the obs buffer to be ready ......")
        env.obs_buffer.wait_buffer_ready(just_img=is_just_img)
        obs,_,_,_,_ = env.get_obs(just_img=is_just_img)
        for k, v in obs.items():
            print(f'{k=}, \t {v.shape=}')
        
        while 1:
            obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs(is_just_img)

            print('##################################################################')
            print(f"{camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}, \n"
                f"{camera_obs_timestamps['img02'][0]:.10f}, {camera_obs_timestamps['img02'][1]:.10f}, \n"
            )
            if not is_just_img:
                print(f"{robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}, \n"
                    f"{robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}\n"
                    )
            print('##################################################################')
            
            # =============
            # show sense img
            # =============
            imgs = []
            
            for k, v in obs.items():
                if "img01" in k:
                    img = obs[k][-1]
                    img = cv2.resize(img, (640, 480))
                    img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    imgs.append(img01_bgr)
            concatenated_img = np.hstack(imgs)  # 横向拼接图像
            cv2.imwrite('/home/leju-ali/hx/oridp/check.png', concatenated_img)
            time.sleep(1)