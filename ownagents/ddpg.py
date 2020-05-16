import numpy as np
import tensorflow as tf
from ownagents.utils.tf_utils import get_vars, Normalizer
import ownagents.tf_util as U
#from algorithm.replay_buffer import goal_based_process


class DDPG:
	def __init__(self, sess, layer_index, env, args):
		self.args = args
		self.sess = sess
		self.layer_index = layer_index

		self.pi_lr = args.pi_lr
		self.q_lr = args.q_lr
		self.polyak = args.polyak
		# for level 0: S_0 = S, A_0 = A, T_0 = T, G_0 = S;
		# for level i: S_i = S, A_i = S, T_i = T~, G_i = S except for most bigger hierarchy (k-1) G_k-1 = G
		if layer_index == 0:
			self.action_space_bounds = env.action_bounds
			self.action_offset = env.action_offset
			self.action_dims = env.action_dim
		else:
			# Determine symmetric range of subgoal space and offset
			self.action_space_bounds = env.subgoal_bounds_symmetric
			self.action_offset = env.subgoal_bounds_offset
			self.action_dims = env.subgoal_dim
		if layer_index == args.num_layers - 1:
			self.goal_dim = env.end_goal_dim
		else:
			self.goal_dim = env.subgoal_dim
		self.state_dim = env.state_dim

		# Set parameters to give critic optimistic initialization near q_init
		self.q_init = -0.067
		self.q_limit = -args.H
		self.q_offset = -np.log(self.q_limit / self.q_init - 1)

		self.create_model()



		self.train_info_pi = {
			'Pi_q_loss': self.pi_q_loss,
			'Pi_l2_loss': self.pi_l2_loss
		}
		self.train_info_q = {
			'Q_loss': self.q_loss
		}
		self.train_info = {**self.train_info_pi, **self.train_info_q}

		self.step_info = {
			'Q_average': self.q_pi
		}

	def create_model(self):
		input_dims = [self.state_dim + self.goal_dim]
		action_dims = [self.action_dims]
		def create_inputs():
			self.raw_obs_ph = tf.placeholder(tf.float32, [None]+input_dims)
			self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+input_dims)
			self.acts_ph = tf.placeholder(tf.float32, [None]+action_dims)
			self.rews_ph = tf.placeholder(tf.float32, [None, 1])
			self.gamma_ph = tf.placeholder(tf.float32, [None, 1])

		def create_normalizer():
			with tf.variable_scope('normalizer_'+str(self.layer_index)):
				self.obs_normalizer = Normalizer(input_dims, self.sess)
			self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
			self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)

		def create_network():
			def mlp_policy(obs_ph):
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					pi_dense1 = tf.layers.dense(obs_ph, 64, activation=tf.nn.relu, name='pi_dense1')
					pi_dense2 = tf.layers.dense(pi_dense1, 64, activation=tf.nn.relu, name='pi_dense2')
					pi_dense3 = tf.layers.dense(pi_dense2, 64, activation=tf.nn.relu, name='pi_dense3')
					pi = tf.layers.dense(pi_dense3, action_dims[0], activation=tf.nn.tanh, name='pi')
					pi = pi * self.action_space_bounds + self.action_offset#!!!!!this is transformation to action needed for not normalized enviroinments
				return pi

			def mlp_value(obs_ph, acts_ph):
				state_ph = tf.concat([obs_ph, acts_ph], axis=1)
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					q_dense1 = tf.layers.dense(state_ph, 64, activation=tf.nn.relu, name='q_dense1')
					q_dense2 = tf.layers.dense(q_dense1, 64, activation=tf.nn.relu, name='q_dense2')
					q_dense3 = tf.layers.dense(q_dense2, 64, activation=tf.nn.relu, name='q_dense3')
					q = tf.layers.dense(q_dense3, 1, name='q')
					q = tf.sigmoid(q + self.q_offset) * self.q_limit#!!!!!this is the clipping from 0 and -H, from the paper
				return q

			with tf.variable_scope('main_'+str(self.layer_index)):
				with tf.variable_scope('policy'):
					self.pi = mlp_policy(self.obs_ph)
				with tf.variable_scope('value'):
					self.q = mlp_value(self.obs_ph, self.acts_ph)
				with tf.variable_scope('value', reuse=True):
						self.q_pi = mlp_value(self.obs_ph, self.pi)

			with tf.variable_scope('target_'+str(self.layer_index)):
				with tf.variable_scope('policy'):
					self.pi_t = mlp_policy(self.obs_next_ph)
				with tf.variable_scope('value'):
					self.q_t = mlp_value(self.obs_next_ph, self.pi_t)

		def create_operators():
			self.pi_q_loss = -tf.reduce_mean(self.q_pi)
			self.pi_l2_loss = self.args.act_l2*tf.reduce_mean(tf.square(self.pi))
			self.pi_optimizer = tf.train.AdamOptimizer(self.pi_lr)
			self.pi_train_op = self.pi_optimizer.minimize(self.pi_q_loss+self.pi_l2_loss,
														  var_list=get_vars('main_'+str(self.layer_index)+'/policy'))

			'''if self.args.clip_return:
				return_value = tf.clip_by_value(self.q_t, self.args.clip_return_l, self.args.clip_return_r)
			else:
				return_value = self.q_t'''
			return_value = self.q_t
				
			discounted = self.gamma_ph * return_value
			target = tf.stop_gradient(self.rews_ph+discounted)
			self.q_loss = tf.reduce_mean(tf.square(self.q-target))
			self.q_optimizer = tf.train.AdamOptimizer(self.q_lr)
			self.q_train_op = self.q_optimizer.minimize(self.q_loss,
														var_list=get_vars('main_'+str(self.layer_index)+'/value'))

			self.target_update_op = tf.group([
				v_t.assign(self.polyak*v_t + (1.0-self.polyak)*v)
				for v, v_t in zip(get_vars('main_'+str(self.layer_index)),
								  get_vars('target_'+str(self.layer_index)))
			])

			self.saver=tf.train.Saver()
			self.init_op = tf.global_variables_initializer()
			self.target_init_op = tf.group([
				v_t.assign(v)
				for v, v_t in zip(get_vars('main_'+str(self.layer_index)),
								  get_vars('target_'+str(self.layer_index)))
			])

		create_inputs()
		create_normalizer()
		create_network()
		create_operators()

	def init_network(self):
		self.sess.run(self.init_op)
		self.sess.run(self.target_init_op)

	def step(self, obs, explore=False, test_info=False):
		#if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
		#	return np.random.uniform(-1, 1, size=self.action_dims)
		# TODO if self.args.goal_based: obs = goal_based_process(obs)#neede?

		# eps-greedy exploration; if eps:act==20 means 20% will be totally random action; else 80%
		if explore and np.random.uniform() <= self.args.eps_act:
			#return np.random.uniform(-1, 1, size=self.action_dims)#!!!!!!this works just with normalized actions
			a = np.random.uniform(-1, 1, size=self.action_dims)
			return a*self.action_space_bounds + self.action_offset#!!!!this maps againto the actioons dimensions!!!!
		feed_dict = {
			self.raw_obs_ph: [obs]
		}
		action, info = self.sess.run([self.pi, self.step_info], feed_dict)
		action = action[0]

		# uncorrelated gaussian exploration
		'''so will work just for normalized actions
		if explore:
			action += np.random.normal(0, self.args.std_act, size=self.action_dims)
		action = np.clip(action, -1, 1)'''
		if explore:
			action += np.random.normal(0, self.args.std_act, size=self.action_dims)#!!!!!!!!!!!!!!
			action = np.clip(action, -self.action_space_bounds, self.action_space_bounds)


		if test_info:
			return action, info
		return action

	def step_batch(self, obs):
		actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
		return actions

	def get_feed_dict(self, batch):
		return {
			self.raw_obs_ph: batch['obs'],
			self.raw_obs_next_ph: batch['obs1'],
			self.acts_ph: batch['actions'],
			self.rews_ph: U.adjust_shape(self.rews_ph, batch['rewards']),
		   	self.gamma_ph: U.adjust_shape(self.gamma_ph, batch['gammas'])
		}

	def train(self, batch):
		feed_dict = self.get_feed_dict(batch)
		info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
		return info

	def train_pi(self, batch):
		feed_dict = self.get_feed_dict(batch)
		info, _ = self.sess.run([self.train_info_pi, self.pi_train_op], feed_dict)
		return info

	def train_q(self, batch):
		feed_dict = self.get_feed_dict(batch)
		info, _ = self.sess.run([self.train_info_q, self.q_train_op], feed_dict)
		return info

	def normalizer_update(self, batch):
		self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs1']], axis=0))

	def target_update(self):
		self.sess.run(self.target_update_op)
