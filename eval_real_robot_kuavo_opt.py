import time
import threading
import signal
import sys
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import imageio
import rospy
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from datetime import datetime

from diffusion_policy.real_world.real_env_kuavo_Task7_SuZhou import KuavoEnv
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Configuration
INPUT_PATH = "/home/leju-ali/hx/kuavo/Task8-SuZhou/data/outputs/2025.02.23/19.42.10_train_diffusion_unet_image_Kuavo_Task8_SuZhou_task/checkpoints/latest.ckpt"
OUTPUT_PATH = "/home/leju-ali/hx/kuavo/Task8-SuZhou/data/outputs/2025.02.23/19.42.10_train_diffusion_unet_image_Kuavo_Task8_SuZhou_task/checkpoints/output"
VIS_CAMERA_IDX = 1
STEPS_PER_INFERENCE = 6
FREQUENCY = 15
COMMAND_LATENCY = 0.01
INFERENCE_STEPS = 16  # DDIM inference iterations

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_policy(ckpt_path):
    """Load policy from checkpoint"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)
    
    if 'diffusion' not in cfg.name:
        raise RuntimeError(f"Unsupported policy type: {cfg.name}")
    
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(torch.device('cuda'))
    policy.num_inference_steps = INFERENCE_STEPS
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    
    return policy, cfg

def get_timestamp():
    """Get current time as formatted string"""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def log_params(**kwargs):
    """Log parameters in a clean format"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

def setup_video_recording(env, cfg, latency):
    """Setup video recording"""
    timestamp = get_timestamp()
    output_video_path = f'./{cfg.task.name}_{timestamp}_speedx1_latency{latency}ms.mp4'
    record_title = f"Latency of [Obs_trans, PredictRet_trans, Motor] =[{latency}ms, {latency}ms, {latency}ms]"
    
    stop_event = threading.Event()
    video_thread = threading.Thread(
        target=env.record_video, 
        args=(output_video_path, 640, 480, 30, True, stop_event, record_title)
    )
    video_thread.start()
    
    # Setup separate video writer for direct frame recording
    writer_path = f'./{cfg.task.name}_{timestamp}_speedx_latency{latency}ms.mp4'
    writer = imageio.get_writer(writer_path, fps=10, codec="libx264")
    
    return stop_event, writer, record_title

def add_text_to_image(image, text):
    """Add text overlay to image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    draw.text((50, 50), text, font=font, fill="green")
    return np.array(pil_img)

def handle_exit(signum, frame, stop_event):
    """Signal handler for clean exit"""
    print("Signal received, saving video and cleaning up...")
    stop_event.set()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    # Load policy
    policy, cfg = load_policy(INPUT_PATH)
    
    # Log configuration parameters
    log_params(
        n_obs_steps=cfg.n_obs_steps,
        steps_per_inference=STEPS_PER_INFERENCE,
        action_offset=0
    )
    
    # Initialize environment
    rospy.init_node("test")
    with KuavoEnv(
        frequency=FREQUENCY,
        n_obs_steps=2,
        video_capture_fps=FREQUENCY,
        robot_publish_rate=500,
        img_buffer_size=30,
        robot_state_buffer_size=500,
        video_capture_resolution=(640, 480),
        output_dir=OUTPUT_PATH,
    ) as env:
        print("Waiting for observation buffer to be ready...")
        time.sleep(1.0)
        env.obs_buffer.wait_buffer_ready()
        
        print("Warming up policy inference")
        latency = 0  # in ms
        
        # Setup video recording
        stop_event, writer, record_title = setup_video_recording(env, cfg, latency)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda sig, frame: handle_exit(sig, frame, stop_event))
        signal.signal(signal.SIGQUIT, lambda sig, frame: handle_exit(sig, frame, stop_event))
        
        print('Ready!')
        try:
            # Start episode
            policy.reset()
            start_delay = 1.0
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay
            print("Started!")
            
            iter_idx = 0
            with open('inference_latency.txt', 'a') as log_file:
                while True:
                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + STEPS_PER_INFERENCE) * (1/FREQUENCY)
                    
                    # Get observations
                    obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
                    
                    # Log timestamps
                    print('=' * 70)
                    print(f"Camera timestamps: {camera_obs_timestamps['img01'][0]:.10f}, {camera_obs_timestamps['img01'][1]:.10f}")
                    print(f"Robot joint timestamps: {robot_obs_timestamps['ROBOT_state_joint'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_joint'][1]:.10f}")
                    print(f"Robot gripper timestamps: {robot_obs_timestamps['ROBOT_state_gripper'][0]:.10f}, {robot_obs_timestamps['ROBOT_state_gripper'][1]:.10f}")
                    print('=' * 70)
                    
                    # Add latency if specified
                    start_point = time.time()
                    time.sleep(latency/1000)
                    
                    # Save and display images
                    Image.fromarray(obs['img01'][0].astype(np.uint8)).save("saved_image_pillow.png")
                    
                    # Prepare images for display
                    display_images = []
                    for k, v in obs.items():
                        if "img" in k:
                            img = obs[k][-1]
                            img = cv2.resize(img, (640, 480))
                            display_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    concatenated_img = np.hstack(display_images)
                    cv2.imshow('Image Stream', concatenated_img)
                    
                    # Add text and save to video
                    img_with_text = add_text_to_image(concatenated_img, record_title)
                    writer.append_data(img_with_text)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Run inference
                    with torch.no_grad():
                        obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                        print("End effector state:", obs_dict_np['agent_pos'][-1])
                        
                        obs_dict = dict_apply(
                            obs_dict_np, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to('cuda')
                        )
                        
                        # Measure inference time
                        inference_start = time.time()
                        result = policy.predict_action(obs_dict)
                        inference_time = time.time() - inference_start
                        
                        print(f'Inference latency: {inference_time:.4f}s')
                        log_file.write(f"Inference latency: {inference_time:.4f} seconds\n")
                        
                        # Extract action
                        action = result['action'][0].detach().cpu().numpy()
                    
                    # Add latency if specified
                    time.sleep(latency/1000)
                    
                    # Execute actions
                    print(f'Total pipeline latency: {time.time() - start_point:.4f}s')
                    env.exec_actions(
                        actions=action[::2],
                        cur_state=obs_dict_np['agent_pos'][-1][:6],
                        start_point=start_point,
                        latency=latency/1000,
                    )
                    print(f"Submitted {len(action)} steps of actions.")
                    
                    iter_idx += 1
                    
        except KeyboardInterrupt:
            print("Interrupted!")
            
        finally:
            writer.close()
            cv2.destroyAllWindows()
            print("Stopped.")

if __name__ == '__main__':
    main()