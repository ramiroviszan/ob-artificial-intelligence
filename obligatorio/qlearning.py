
import numpy as np
import pickle
import os
import re
import datetime


def max_policy(state, env, state_action_values):
    actions = env.get_legal_actions()
    act_index = np.argmax(list(map(lambda a: state_action_values.get(
        (env.flatten_obs(state), a), 0), actions)))
    return actions[act_index]


def policy_runner(strategy, env, layout_name, enable_render=False):
    done = False
    state, _ = env.reset(layout_name=layout_name, enable_render=enable_render)
    reward_sum = 0
    step_count = 0
    while (not(done)):
        action = strategy(state, env)
        state, new_reward, done, info = env.step(action)
        win = info["win"]
        score = info["score"]
        step_count += 1
        reward_sum += new_reward

    return reward_sum, step_count, win, score


class QLearner():

    def __init__(self, discount=0.9, use_softmax=True):
        self.use_softmax = use_softmax

        self.discount = discount
        self.states = set()#values: flatten_obs
        self.state_action_values = dict()#key: (flatten_obs, action), value: q value
        #if needed add here more structures
        #Note: if loading fails after adding more structures, delete old saves

        #Only for info
        self.iterations_total = 0
        self.layout_score = dict()
        self.layout_win = dict()
        self.layout_count = dict()
        self.layout_score_current = dict()
        self.layout_win_current = dict()
        self.layout_count_current = dict()
        self.layout_score_last = dict()
        self.layout_win_last = dict()
        self.layout_count_last = dict()

    def learn_n(self, env, iterations=1000, render_mod=180, layouts=["mediumClassic"], save_mod=50, info_mod=200, verbose=False):

        for i in range(iterations):
            layout = layouts[i % len(layouts)]
            reward_sum, step_count, win, score = self.learn(env, render_mod is not None and i % render_mod == 0,
                                                            layout_name=layout)

            self.update_info(layout, score, win, False)

            self.iterations_total += 1
            if (verbose):
                print(self.iterations_total, i+1, layout.ljust(16),
                      str(reward_sum).rjust(6), str(step_count).rjust(3), win, score)
            if (save_mod is not None and i % save_mod == 0 and i != 0):
                self.save()
            if (info_mod is not None and i % info_mod == 0):
                self.print_info(True)
        self.print_info(True)

    def learn(self, env, enable_render=False, layout_name="mediumClassic"):
        done = False
        reward_sum = 0
        step_count = 0
        state, _ = env.reset(enable_render=enable_render,
                             layout_name=layout_name)
        s_flat = env.flatten_obs(state)

        while (not(done)):
            self.states.add(s_flat)
            raise Exception('TODO: implement')
            a = 0  # choose action
            new_state, r, done, info = env.step(a)
            win = info["win"]
            score = info["score"]
            reward_sum += r
            step_count += 1

        return reward_sum, step_count, win, score

    def get_policy(self):
        state_action_values = dict(self.state_action_values)

        def q_learning_policy(state, env):
            return max_policy(state, env, state_action_values)

        return q_learning_policy

    def run(self, env, layouts=["mediumClassic"], iterations=1000, render_mod=180):
        self.clear_scores()
        policy = self.get_policy()
        for i in range(iterations):
            layout = layouts[i % len(layouts)]
            reward_sum, step_count, win, score = policy_runner(
                policy, env, layout, enable_render=render_mod is not None and i % render_mod == 0)
            self.update_info(layout, score, win, True)
        self.print_info(False)

    def print_info(self, train):
        print(datetime.datetime.now())
        print("Train"if train else "Run", "Info iterations", self.iterations_total, "states", len(
            self.states), "state_actions", len(self.state_action_values), "use_softmax", self.use_softmax)
        print("\t", "|", "Layout".ljust(16), "Iter".rjust(5),
              "|", "Avg".rjust(11), "Avg Cur.".rjust(11), "Avg Last".rjust(11),
              "|", "Wins".rjust(11), "Wins Cur.".rjust(11), "Wins Last".rjust(11), "|",)
        for l in self.layout_score.keys():
            print("\t", "|",
                  l.ljust(16),
                  str(self.layout_count[l]).rjust(5),
                  "|",
                  str(int(self.layout_score[l] /
                          self.layout_count[l])).rjust(11),
                  str(int(self.layout_score_current.get(l, -1) /
                          self.layout_count_current.get(l, 1))).rjust(11),
                  str(int(self.layout_score_last.get(l, -1) /
                          self.layout_count_last.get(l, 1))).rjust(11),
                  "|",
                  (str(round(self.layout_win.get(l, 0)*100 /
                             self.layout_count.get(l, 1), 1))+"%").rjust(11),
                  (str(round(self.layout_win_current.get(l, 0)*100 /
                             self.layout_count_current.get(l, 1), 1))+"%").rjust(11),
                  (str(round(self.layout_win_last.get(l, 0)*100 /
                             self.layout_count_last.get(l, 1), 1))+"%").rjust(11),
                  "|",
                  )
        print()
        self.clear_scores()

    def clear_scores(self):
        self.layout_score_last = self.layout_score_current
        self.layout_count_last = self.layout_count_current
        self.layout_win_last = self.layout_win_current

        self.layout_score_current = dict()
        self.layout_count_current = dict()
        self.layout_win_current = dict()

    def update_info(self, layout, score, win, update_only_current):
        if (not(update_only_current)):
            self.layout_score[layout] = self.layout_score.get(layout, 0)+score
            self.layout_count[layout] = self.layout_count.get(
                layout, 0)+1

        self.layout_score_current[layout] = self.layout_score_current.get(
            layout, 0)+score
        self.layout_count_current[layout] = self.layout_count_current.get(
            layout, 0)+1

        if win:
            if (not(update_only_current)):
                self.layout_win[layout] = self.layout_win.get(
                    layout, 0)+1
            self.layout_win_current[layout] = self.layout_win_current.get(
                layout, 0)+1

    def save(self):
        d = os.path.join('obligatorio', "saves")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'Qlearner ' + str(self.iterations_total) + ' .pkl'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(iteration=None):
        d = os.path.abspath(os.path.join('obligatorio', "saves"))
        os.makedirs(d, exist_ok=True)
        if (iteration is None):
            iterations = sorted(
                map(lambda numbers: numbers[0],
                    filter(lambda numbers: len(numbers) > 0,
                           map(lambda f: [int(s) for s in f.split() if s.isdigit()],
                               filter(lambda f: re.match("Qlearner \d+ .pkl", f) is not None, os.listdir(d))))))
            if (len(iterations) > 0):
                iteration = iterations[-1]

            else:
                return None

        with open(os.path.join(d, 'Qlearner ' + str(iteration) + ' .pkl'), 'rb') as input:
            return pickle.load(input)
