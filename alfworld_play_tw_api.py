from flask import Flask, request, jsonify
import textworld
from textworld.agents import HumanAgent
import textworld.gym
import os
import json
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType
from os.path import join as pjoin
import glob
import random

app = Flask(__name__)

# Biến toàn cục demo cho 1 session
SESSION = {
    'env': None,
    'obs': None,
    'infos': None,
    'done': False
}

def create_env(problem=None):
    if problem is None:
        problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
        problems = [p for p in problems if "movable_recep" not in p]
        if len(problems) == 0:
            raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
        problem = os.path.dirname(random.choice(problems))
    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
    # load state and trajectory files
    pddl_file = os.path.join(problem, 'initial_state.pddl')
    json_file = os.path.join(problem, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)
    with open(domain) as f:
        pddl_domain = f.read()
    with open(grammar) as f:
        grammar_text = f.read()
    grammar_text = add_task_to_grammar(grammar_text, traj_data)
    gamedata = dict(pddl_domain=pddl_domain, grammar=grammar_text, pddl_problem=open(pddl_file).read())
    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    json.dump(gamedata, open(gamefile, "w"))
    expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, score=True, max_score=True, intermediate_reward=True, extras=["expert_plan"])
    env_id = textworld.gym.register_game(gamefile, request_infos, max_episode_steps=1000000, wrappers=[AlfredDemangler(), expert])
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()
    return env, obs, infos

@app.route('/start', methods=['POST'])
def start():
    global SESSION
    env, obs, infos = create_env()
    SESSION['env'] = env
    SESSION['obs'] = obs
    SESSION['infos'] = infos
    SESSION['done'] = False
    return jsonify({'obs': obs, 'done': False})

@app.route('/step', methods=['POST'])
def step():
    global SESSION
    if SESSION['env'] is None or SESSION['done']:
        return jsonify({'error': 'Session not started or already done.'}), 400
    data = request.get_json()
    cmd = data.get('cmd', '')
    obs, score, done, infos = SESSION['env'].step(cmd)
    SESSION['obs'] = obs
    SESSION['infos'] = infos
    SESSION['done'] = done
    return jsonify({'obs': obs, 'done': done, 'score': score})

@app.route('/reset', methods=['POST'])
def reset():
    global SESSION
    if SESSION['env'] is None:
        return jsonify({'error': 'Session not started.'}), 400
    obs, infos = SESSION['env'].reset()
    SESSION['obs'] = obs
    SESSION['infos'] = infos
    SESSION['done'] = False
    return jsonify({'obs': obs, 'done': False})

if __name__ == '__main__':
    app.run(debug=True) 