from flask import Flask, request, jsonify
import os
import sys
import copy
import numpy as np
from alfworld.agents.modules import generic
from alfworld.agents.agent.text_ppo_agent import TextPPOAgent
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

app = Flask(__name__)

@app.route('/ppo_play', methods=['POST'])
def ppo_play():
    data = request.get_json()
    checkpoint_path = data.get('checkpoint')
    problem_path = data.get('problem')
    # Load config và agent
    config = generic.load_config()
    agent = TextPPOAgent(config)
    agent.eval()
    if checkpoint_path:
        agent.load_pretrained_model(checkpoint_path)
        agent.update_target_net()
    alfred_env = AlfredTWEnv(config, train_eval="eval")
    env = alfred_env.init_env(batch_size=1)
    # Reset env
    if problem_path:
        obs, infos = env.reset(problem_path)
    else:
        obs, infos = env.reset()
    obs = list(obs)
    infos = dict(infos)
    log_steps = []
    done = False
    previous_dynamics = [None]
    step_no = 0
    while not done:
        action_candidates = list(infos["admissible_commands"])
        action_candidates = agent.preprocess_action_candidates([action_candidates])[0]
        task_desc, obs_str = agent.get_task_and_obs([obs[0]])
        task_desc = agent.preprocess_task(task_desc)[0]
        obs_str = agent.preprocess_observation(obs_str)[0]
        action, action_idx, logprob, value, current_dynamics = agent.select_action(
            obs_str, task_desc, action_candidates, previous_dynamics[0]
        )
        log_steps.append({
            'step': step_no+1,
            'action': action,
            'obs': obs[0]
        })
        obs, score, done, infos = env.step([action])
        obs = list(obs)
        previous_dynamics = [current_dynamics]
        step_no += 1
        if done:
            break
    result = {
        'steps': log_steps,
        'total_steps': step_no,
        'final_score': float(score[0])
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 