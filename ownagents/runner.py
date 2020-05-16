from ownagents.utils.os_utils import get_arg_parser
from ownagents.hrl_agent import HRAgent

from ownagents.design_agent_and_env2 import design_agent_and_env
from options import parse_options
from ownagents.common import get_args
from ownagents.csv_logger import CSVLogger
import copy


def run():
    FLAGS = parse_options()
    FLAGS.retrain = True # this is just important fot the other agent
    args = get_args()
    args.FLAGS = FLAGS
    other_agent, other_env = design_agent_and_env(FLAGS, first_call=True)
    _, env = design_agent_and_env(FLAGS)
    agent = HRAgent(3, env, args)
    # Begin training
    csvlogger = CSVLogger('HAC_log')
    csvloggerEv = CSVLogger('HAC_Ev_log')
    other_csvlogger = CSVLogger('other_HAC_log')
    other_csvloggerEv = CSVLogger('other_HAC_Ev_log')

    successful_episodes_train = 0
    successful_episodes_evaluate = 0
    other_successful_episodes_train = 0
    other_successful_episodes_evaluate = 0

    # total_steps = num_epochs * num_episode * max_real_steps_in_env
    for epoch in range(10):

        num_episodes = 50

        #train the agent
        for episode in range(num_episodes):

            print("\nepoch %d, Episode %d" % (epoch, episode))

            # Train for an episode
            print('********************************************************************************')
            success, losses, steps_taken, cumulated_reward = agent.train(env, episode)
            d = csvlogger.to_dict_entry(success, losses, steps_taken, cumulated_reward)
            csvlogger.log_to_file(d)

            if success:
                print("epoch %d, Episode %d End Goal Achieved\n" % (epoch, episode))
                successful_episodes_train += 1
            print('-------------------------------------------------------------------------')

            '''other_agent.FLAGS.test = False
            other_success, other_losses, other_steps_taken, other_cumulated_reward = other_agent.train(other_env, episode)
            other_d = other_csvlogger.to_dict_entry(other_success, other_losses, other_steps_taken, other_cumulated_reward)
            other_csvlogger.log_to_file(other_d)

            if other_success:
                print("epoch %d, Episode %d End Goal Achieved\n" % (epoch, episode))
                other_successful_episodes_train += 1'''
            print('********************************************************************************')

        #evaluate; store in different data set
        success, steps_taken, cumulated_reward = agent.evaluate(env)
        d = csvloggerEv.to_dict_entry(success, None, steps_taken, cumulated_reward)
        csvloggerEv.log_to_file(d)
        if success:
            successful_episodes_evaluate +=1

        '''other_agent.FLAGS.test = True
        other_success, _ ,other_steps_taken, other_cumulated_reward = other_agent.train(other_env, episode)
        other_d = other_csvloggerEv.to_dict_entry(other_success, None, other_steps_taken, other_cumulated_reward)
        other_csvloggerEv.log_to_file(other_d)
        if other_success:
            other_successful_episodes_evaluate +=1'''

if __name__ == '__main__':
    run()