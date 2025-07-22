import os
import sys
import copy
import numpy as np
from alfworld.agents.modules import generic
from alfworld.agents.agent.text_ppo_agent import TextPPOAgent
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

def play_with_ppo(checkpoint_path=None, problem_path=None):
    # Load config
    config = generic.load_config()
    agent = TextPPOAgent(config)
    agent.eval()
    if checkpoint_path is not None:
        agent.load_pretrained_model(checkpoint_path)
        agent.update_target_net()
        print(f"Đã nạp checkpoint: {checkpoint_path}")
    else:
        print("Không tìm thấy checkpoint, sẽ dùng agent chưa huấn luyện!")

    alfred_env = AlfredTWEnv(config, train_eval="eval")
    env = alfred_env.init_env(batch_size=1)

    # Chọn ngẫu nhiên một nhiệm vụ nếu không chỉ định
    if problem_path is not None:
        obs, infos = env.reset(problem_path)
    else:
        obs, infos = env.reset()
    obs = list(obs)
    infos = dict(infos)
    print("\n--- Bắt đầu episode mới ---")
    print(obs[0])

    done = False
    previous_dynamics = [None]
    step_no = 0
    while not done:
        # Lấy các lệnh hợp lệ
        action_candidates = list(infos["admissible_commands"])
        action_candidates = agent.preprocess_action_candidates([action_candidates])[0]
        # Lấy mô tả nhiệm vụ
        task_desc, obs_str = agent.get_task_and_obs([obs[0]])
        task_desc = agent.preprocess_task(task_desc)[0]
        obs_str = agent.preprocess_observation(obs_str)[0]
        # Agent chọn action
        action, action_idx, logprob, value, current_dynamics = agent.select_action(
            obs_str, task_desc, action_candidates, previous_dynamics[0]
        )
        print(f"\nBước {step_no+1}: Agent chọn lệnh: {action}")
        obs, score, done, infos = env.step([action])
        obs = list(obs)
        print(obs[0])
        previous_dynamics = [current_dynamics]
        step_no += 1
        if done:
            print("\n--- Episode kết thúc ---")
            print(f"Số bước: {step_no}")
            print(f"Điểm số: {score[0]}")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chạy agent PPO đã train trên một nhiệm vụ TextWorld.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Đường dẫn tới file checkpoint .pt')
    parser.add_argument('--problem', type=str, default=None, help='Đường dẫn tới thư mục nhiệm vụ (có initial_state.pddl, traj_data.json)')
    args = parser.parse_args()
    play_with_ppo(args.checkpoint, args.problem) 