from environment import Environment
from ownagents.ddpg import DDPG
import tensorflow as tf
import numpy as np
from ownagents.ReplayBuffer import ReplayBuffer
from collections import defaultdict
import copy

class HRAgent:
    def __init__(self, num_layers, env, args):
        self.num_layers = num_layers
        self.FLAGS = args.FLAGS

        self.agents = []
        self.create_agents(env, args)
        self.replay_buffers = [ReplayBuffer(max_size=args.replay_buffer_size) for _ in range(num_layers)]
        self.steps_limits = [args.H for _ in range(num_layers)]
        self.H = args.H #for num_layers 2 the H in [20,30]
        self.subgoal_testing_rate = args.subgoal_testing_rate
        self.batch_size = 1024
        self.penalty = -args.H
        self.gamma = args.gamma
        self.training_times = 40

    def create_agents(self, env, args):
        def create_session():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        self.graph = tf.Graph()
        with self.graph.as_default():
            create_session()
            for i in range(self.num_layers):
                self.agents.append(DDPG(self.sess, i, env, args))
        self.init_agent_networks()

    def init_agent_networks(self):
        for agent_i in self.agents:
            agent_i.init_network()

    def evaluate(self, env):
        self.end_goal = env.get_next_goal(True)#Set tesintg flag to true
        self.initial_state = env.reset_sim()
        self.current_state = self.initial_state  # to keep track of state

        # Rollout all levels
        self.total_steps_taken = 0  # The steps taken overall
        self.cumulated_reward = 0
        self.agents_goals = [None for _ in range(self.num_layers)]
        goal_status, max_lay_achieved = self.train_level(self.num_layers - 1, self.initial_state, self.end_goal, env)
        # update the networks from experiences
        return goal_status[self.num_layers - 1], copy.copy(self.total_steps_taken), \
               copy.copy(self.cumulated_reward)

    #todo it would be better to have the cumulated reward as return and not as
    #side effect
    def evaluate_level(self, level, state, goal, env):
        agent_level = self.agents[level]
        s_level = state
        g_level = goal
        self.agents_goals[level] = goal
        for attempt_i in range(self.steps_limits[level]):
            # epsilon-greedy exploration; greedy if subgoal_testing
            a_level = agent_level.step(np.concatenate((s_level, g_level)), explore=False)
            if level > 0:
                goal_status, max_lay_achieved = self.evaluate_level(level - 1, s_level, a_level, env)
            else:
                # performs the action since it is the lowest policy
                next_state = env.execute_action(a_level)  # gym this is different
                self.current_state = next_state
                self.total_steps_taken += 1
                if self.total_steps_taken >= env.max_actions:  # this must be done differently in gym
                    print("Out of actions (Steps: %d)" % self.total_steps_taken)
                goal_status, max_lay_achieved = self.check_goals(env, next_state)

            next_state = self.current_state
            # we are done when a goal was achieved (either own or from higher level) or cannot do more steps
            reward, done, gamma = (0, True, 0) if goal_status[level] else (-1, False, self.gamma)
            if level == 0:
                self.cumulated_reward += reward
            if self.total_steps_taken >= env.max_actions \
                    or (max_lay_achieved is not None and max_lay_achieved >= level):
                return goal_status, max_lay_achieved
            s_level = next_state
        return goal_status, max_lay_achieved

    #this will be called by another class M times (number of episodes)
    def train(self, env, i_episode):
        self.i_episode = i_episode
        self.end_goal = env.get_next_goal(False)
        self.initial_state = env.reset_sim()
        self.current_state = self.initial_state #to keep track of state
        print("Next End Goal: ", self.end_goal)
        print("Very first Initial state: ", self.initial_state)

        #Rollout all levels
        self.total_steps_taken = 0 #The steps taken overall
        self.cumulated_reward = 0
        self.agents_goals = [None for _ in range(self.num_layers)]
        self.maxed_out = [False for _ in range(self.num_layers)]

        goal_status, max_lay_achieved = self.train_level(self.num_layers-1, self.initial_state, self.end_goal, env)
        #update the networks from experiences
        losses = []
        for i in range(self.num_layers):
            self.agents[i].normalizer_update(self.replay_buffers[i].sample_batch(batch_size=self.batch_size))
            l = defaultdict(float)
            for _ in range(self.training_times):
                info = self.agents[i].train(self.replay_buffers[i].sample_batch(batch_size=self.batch_size))
                for k, v in info.items():
                    l[k] += v
            self.agents[i].target_update()
            for k in l.keys():
                l[k] /= self.training_times
            losses.append(l)
        return goal_status[self.num_layers - 1], losses, copy.copy(self.total_steps_taken), \
               copy.copy(self.cumulated_reward)

    def train_level(self, level, state, goal, env, upper_doing_subgoal_test=False):
        print('Training level ', level)
        agent_level = self.agents[level]
        s_level = state
        g_level = goal
        self.agents_goals[level] = goal
        transitions_for_her = ReplayBuffer(max_size=self.steps_limits[level])
        goal_testing_transition = ReplayBuffer(max_size=self.steps_limits[level])# temporal to avoid higdsight goals
                                                                                 # are these ones
        for attempt_i in range(self.steps_limits[level]):
            # epsilon-greedy exploration; greedy if subgoal_testing
            a_level = agent_level.step(np.concatenate((s_level, g_level)), explore=not upper_doing_subgoal_test)
            if level == 0:
                assert (- env.action_bounds <= a_level).all() and (a_level <= env.action_bounds).all()
            else:
                assert (- env.subgoal_bounds_symmetric <= a_level).all() and (a_level <= env.subgoal_bounds_symmetric).all()
            if level > 0:
                if upper_doing_subgoal_test:
                    next_subgoal_test = True
                else:
                    next_subgoal_test = np.random.uniform() <= self.subgoal_testing_rate
                goal_status, max_lay_achieved = self.train_level(level-1, s_level, a_level, env, next_subgoal_test)
            else:
                #performs the action since it is the lowest policy
                next_state = env.execute_action(a_level)#gym this is different
                self.current_state = next_state
                self.total_steps_taken += 1
                if self.total_steps_taken >= env.max_actions:#this must be done differently in gym
                    print("Out of actions (Steps: %d)" % self.total_steps_taken)
                goal_status, max_lay_achieved = self.check_goals(env, next_state)

            if goal_status[level]:
                print('\n')
                if level < self.FLAGS.agents - 1:
                    print("SUBGOAL ACHIEVED")
                print("Episode %d, Layer %d, Attempt %d Goal Achieved" % (
                    self.i_episode, level, attempt_i))
                print("Goal: ", g_level)
                if level == self.FLAGS.agents - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, self.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, self.current_state))

            next_state = env.project_state_to_subgoal(env.sim, self.current_state)
            #HINDSIGHT ACTION TRANSITIONS
            #if level 0 cannot change; if other level we do not change if sublevel achieved its goal (our proposed
            #action is a state) if not achieved we take the achieved state
            # Print if goal from current layer as been achieved
            hindsight_action = a_level if level == 0 or goal_status[level - 1] else next_state
            # for level = 0In addition, the discountrate is set to 0 if the goal has been achieved, but remains γ otherwise
            # γ_i  is set to 0 if a subgoal is tested and missed or if an action achieves the goal, but is otherwise γ f
            reward, done, gamma = (0, True, 0) if goal_status[level] else (-1, False, self.gamma)
            self.replay_buffers[level].add(ob=np.concatenate((s_level, g_level)),
                                           action=hindsight_action,
                                           ob1=np.concatenate((next_state, g_level)),
                                           done=done,
                                           reward=reward,
                                           gamma=gamma)

            #To generate afterwards hindsight goals
            transitions_for_her.add(ob=np.concatenate((s_level, g_level)),
                                           action=hindsight_action,
                                           ob1=np.concatenate((next_state, g_level)),
                                           done=done,
                                           reward=reward,
                                           gamma=gamma)
            #GOAL TESTING
            if level > 0:
                # perhaps parameter maxed out is needed since it can return before if some higher goals was achieved
                if next_subgoal_test and not goal_status[level] and self.maxed_out[level - 1]:
                    #not store yet, to avoid hidsight goals with this
                    goal_testing_transition.add(ob=np.concatenate((s_level, g_level)),
                                           action=a_level,
                                           ob1=np.concatenate((next_state, g_level)),
                                           done=False,
                                           reward=self.penalty,
                                           gamma=0)
            if level == 0:
                self.cumulated_reward += reward

            #we are done when a goal was achieved (either own or from higher level) or cannot do more steps
            if self.total_steps_taken >= env.max_actions \
                    or (max_lay_achieved is not None and max_lay_achieved >= level):
                return goal_status, max_lay_achieved
            s_level = next_state
        if not goal_status[level]:
            self.maxed_out[level] = True
        #create hindsight goals
        self.sample_her(env, level, transitions_for_her, len(g_level), strategy='future', number_extra_goals=3)
        #store goal testing
        transitions_dict = goal_testing_transition.get_all_as_dict()
        self.replay_buffers[level].store_from_dict(transitions_dict)
        return goal_status, max_lay_achieved


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self, env, current_state): #this is a state in the actual world Space S from the environment
        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for _ in range(self.num_layers)]
        max_lay_achieved = None
        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, current_state) # The space S_i for subgoals
        proj_end_goal = env.project_state_to_end_goal(env.sim, current_state) # actual G

        for i in range(self.num_layers):
            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.num_layers - 1:
                # Check dimensions are appropriate
                "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"
                assert len(proj_end_goal) == len(self.agents_goals[i]) == len(env.end_goal_thresholds)

                # Check whether layer i's goal was achieved by checking whether projected
                # state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.agents_goals[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:
                # Check that dimensions are appropriate
                "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"
                assert len(proj_subgoal) == len(self.agents_goals[i]) == len(env.subgoal_thresholds)

                # Check whether layer i's goal was achieved by checking whether projected state
                # is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.agents_goals[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def sample_her(self, env, level, transitions_storage, dim_real_goal, strategy='future', number_extra_goals=3):
        assert isinstance(transitions_storage, ReplayBuffer)
        assert strategy in ['future', 'episode', 'random']
        tr_dict = transitions_storage.get_all_as_dict()

        def get_reward_and_done_and_gamma(hindsight_goal, st1):
            goal_achieved = True
            if level == self.num_layers - 1:
                goal_thresholds = env.end_goal_thresholds
                st1_as_G = env.project_state_to_end_goal(env.sim, st1)
            else:
                goal_thresholds = env.subgoal_thresholds
                st1_as_G = env.project_state_to_end_subgoal(env.sim, st1)
            for i in range(len(hindsight_goal)):
                if np.absolute(st1_as_G[i] - hindsight_goal[i]) > goal_thresholds[i]:
                    goal_achieved = False
                    break
            if goal_achieved:
                return 0, True, 0
            else:
                return -1, False, self.gamma

        for t in range(transitions_storage.size()):
            goals = []
            start = None
            if strategy == 'future':
                start = t
            elif strategy == 'episode' or strategy == 'random':  # in theory here we cannot use episode;
                start = 0
            indices = np.random.randint(start, transitions_storage.size(), number_extra_goals)
            for index in indices:
                obs = tr_dict['obs1'][index]
                achieved_state = obs[-dim_real_goal:]
                if level == self.num_layers - 1:
                    goals.append(env.project_state_to_end_goal(env.sim, achieved_state))
                else:
                    goals.append(env.project_state_to_subgoal(env.sim, achieved_state))
            for hindsight_goal in goals:
                obs = tr_dict['obs'][t]
                st = obs[-dim_real_goal:]
                obs = tr_dict['obs'][t]
                st1 = obs[-dim_real_goal:]
                at = tr_dict['actions'][t]
                reward, done, gamma = get_reward_and_done_and_gamma(hindsight_goal, st1)
                self.replay_buffers[level].add(ob=np.concatenate((st, hindsight_goal)),
                                           action=at,
                                           ob1=np.concatenate((st1, hindsight_goal)),
                                           done=done,
                                           reward=reward,
                                           gamma=gamma)