import os
import numpy as np
from alfworld.agents.agent.text_ppo_agent import TextPPOAgent
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import alfworld.agents.modules.generic as generic
import yaml

def evaluate_checkpoint(agent, env, num_episodes=20, max_steps=26):
    rewards = []
    successes = []
    steps_list = []
    for ep in range(num_episodes):
        obs, infos = env.reset()
        batch_size = len(obs)
        # Thêm dòng này để khởi tạo observation pool đúng batch size
        agent.observation_pool.reset(batch_size)
        episode_reward = [0.0] * batch_size
        done = [False] * batch_size
        step = 0
        task_desc_strings, observation_strings = agent.get_task_and_obs(list(obs))
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = observation_strings.copy()
        agent.observation_pool.push_first_sight(first_sight_strings)
        action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
        observation_strings = [item + " [SEP] restart" for item in observation_strings]
        prev_rewards = [0.0] * batch_size
        while step < max_steps:
            agent.observation_pool.push_batch(observation_strings)
            most_recent_observation_strings = agent.observation_pool.get()
            actions = []
            for b in range(batch_size):
                action, _, _, _, _ = agent.select_action(
                    most_recent_observation_strings[b],
                    task_desc_strings[b],
                    action_candidate_list[b],
                    None
                )
                actions.append(action)
            obs, _, dones, infos = env.step(actions)
            scores = [float(item) for item in infos["won"]]
            rewards_step = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]
            prev_rewards = scores
            for b in range(batch_size):
                episode_reward[b] += rewards_step[b]
            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, actions)]
            done = [d or bool(x) for d, x in zip(done, dones)]
            if all(done):
                break
            step += 1
        rewards.extend(episode_reward)
        successes.extend([1 if r > 0 else 0 for r in episode_reward])
        steps_list.append(step)
    avg_reward = np.mean(rewards)
    success_rate = np.mean(successes)
    avg_steps = np.mean(steps_list)
    return avg_reward, success_rate, avg_steps

def main():
    with open('configs/eval_config.yaml') as f:
        config = yaml.safe_load(f)
    checkpoint_dir = '/home/nguyendnt/alfworld-1/training'
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    results = []
    for ckpt in checkpoints:
        print(f"Đánh giá checkpoint: {ckpt}")
        agent = TextPPOAgent(config)
        agent.load_pretrained_model(os.path.join(checkpoint_dir, ckpt))
        alfred_env = AlfredTWEnv(config, train_eval="eval_in_distribution")
        env = alfred_env.init_env(batch_size=1)
        avg_reward, success_rate, avg_steps = evaluate_checkpoint(agent, env, num_episodes=20, max_steps=26)
        print(f"Reward TB: {avg_reward:.2f} | Success rate: {success_rate*100:.1f}% | Steps TB: {avg_steps:.1f}")
        results.append((ckpt, avg_reward, success_rate, avg_steps))
    print("\nTổng hợp kết quả:")
    print("| Checkpoint | Reward TB | Success rate | Steps TB |")
    for ckpt, avg_reward, success_rate, avg_steps in results:
        print(f"| {ckpt} | {avg_reward:.2f} | {success_rate*100:.1f}% | {avg_steps:.1f} |")

    # Lưu kết quả ra file CSV
    import csv
    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "Reward TB", "Success rate", "Steps TB"])
        for ckpt, avg_reward, success_rate, avg_steps in results:
            writer.writerow([ckpt, avg_reward, success_rate, avg_steps])
    print("Đã lưu kết quả vào eval_results.csv")

if __name__ == "__main__":
    main()