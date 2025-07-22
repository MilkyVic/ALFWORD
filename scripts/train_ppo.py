import datetime
import os
import copy
import numpy as np
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent.text_ppo_agent import TextPPOAgent
from alfworld.agents.utils.misc import extract_admissible_commands
from alfworld.agents.modules.generic import HistoryScoreCache

def train():
    config = generic.load_config()
    agent = TextPPOAgent(config)
    alfred_env = AlfredTWEnv(config, train_eval="train")
    env = alfred_env.init_env(batch_size=agent.batch_size)

    output_dir = config["general"]["save_path"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)

    best_performance_so_far = 0.0
    json_file_name = agent.experiment_tag.replace(" ", "_")

    if agent.load_pretrained:
        if os.path.exists(output_dir + "/" + agent.load_from_tag + ".pt"):
            agent.load_pretrained_model(output_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while True:
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)
        previous_dynamics = None

        chosen_actions = ["restart"] * batch_size
        prev_step_dones = [0.0] * batch_size
        prev_rewards = [0.0] * batch_size

        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = copy.deepcopy(observation_strings)
        agent.observation_pool.push_first_sight(first_sight_strings)
        action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
        observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, chosen_actions)]

        episode_rewards = [0.0] * batch_size
        episode_dones = [0.0] * batch_size
        step_cache = []

        for step_no in range(agent.max_nb_steps_per_episode):
            agent.observation_pool.push_batch(observation_strings)
            most_recent_observation_strings = agent.observation_pool.get()

            # Chọn action bằng PPO agent
            actions = []
            action_indices = []
            logprobs = []
            values = []
            for b in range(batch_size):
                action, action_idx, logprob, value, current_dynamics = agent.select_action(
                    most_recent_observation_strings[b],
                    task_desc_strings[b],
                    action_candidate_list[b],
                    previous_dynamics[b] if previous_dynamics is not None else None
                )
                actions.append(action)
                action_indices.append(action_idx)
                logprobs.append(logprob)
                values.append(value)
            # Lưu transition vào buffer
            for b in range(batch_size):
                agent.store_transition(
                    most_recent_observation_strings[b],
                    task_desc_strings[b],
                    action_candidate_list[b],
                    action_indices[b],
                    logprobs[b],
                    0.0,  # reward sẽ cập nhật sau
                    0.0,  # done sẽ cập nhật sau
                    values[b],
                    previous_dynamics[b] if previous_dynamics is not None else None
                )
            obs, _, dones, infos = env.step(actions)
            scores = [float(item) for item in infos["won"]]
            dones = [float(item) for item in dones]
            rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]
            prev_rewards = scores
            for b in range(batch_size):
                agent.buffer.rewards[-batch_size + b] = rewards[b]
                agent.buffer.dones[-batch_size + b] = dones[b]
                episode_rewards[b] += rewards[b]
                episode_dones[b] = dones[b]
            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, actions)]
            previous_dynamics = [None] * batch_size  # Đơn giản hóa, có thể lưu dynamics nếu dùng RNN
            if all(dones):
                break
        # Tính advantage và update agent
        agent.finish_path(last_value=0)
        agent.update()
        avg_reward = np.mean(episode_rewards)
        running_avg_game_points.push(avg_reward)
        running_avg_game_steps.push(step_no + 1)
        print(f"Episode {episode_no} | Reward: {avg_reward:.2f} | Steps: {step_no+1}")
        # Lưu model định kỳ
        if agent.report_frequency > 0 and episode_no % agent.report_frequency == 0:
            agent.save_model_to_path(os.path.join(output_dir, f"{json_file_name}_ep{episode_no}.pt"))
        episode_no += 1

if __name__ == "__main__":
    train() 