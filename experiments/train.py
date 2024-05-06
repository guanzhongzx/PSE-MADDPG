import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random ####自己加的

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm_Zx2023SE", help="name of the scenario script") # simple_tag_Zx #
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=51000, help="number of episodes") #训练次数
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")#maddpg
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")#maddpg
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")#贝尔曼公式折扣因子0.95
    parser.add_argument("--eta", type=float, default=0.99, help="preference discount factor")#####自己加的“偏好向量折扣因子”
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="expname", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../tmp/policy_tt/", help="directory in which training state and model should be saved") #存储路径#0805a6t2e=0.95
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded") #载入数据
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False) # True / True #False #  #
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files_tt/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves_tt/", help="directory where plot data is saved") ###"./learning_curves/"
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False): #  False / True
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries): #
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg'))) #判断为false
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg'))) #判断为false
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark: #载入数据
            print('Loading previous state...')
            U.load_state(arglist.load_dir) #arglist.load_dir = arglist.save_dir

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0

        t_start = time.time()

        print('Starting iterations...')
        #tf.su
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]###维度9=5+4=dim_p+dim_c
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n) #智能体与环境交互 #done_n全False，info_n全n个{}
            print('rew_n= ', rew_n)
            episode_step += 1
            done = all(done_n) ###全False
            terminal = (episode_step >= arglist.max_episode_len)

            if train_step == 0:
                preference_n = np.array(rew_n) ####自己加的
            else:
                preference_n = (1 - arglist.eta) * np.array(rew_n) + arglist.eta * preference_n  ####计算偏好向量
            # print('preference_n= ', preference_n)

            choose_n = [[0] * len(rew_n[0]) for _ in range(env.n)]
            rew_nm = [0] * env.n
            # collect experience
            for i, agent in enumerate(trainers):
                rew_nm[i] = max(rew_n[i])###各agent对应不同目标最大的奖励
                # print('agent_%d get a reward= '% i, rew_nm[i])
                ei_max = np.argmax(preference_n[i])
                choose_n[i][ei_max] = 1 ###生成选择向量
                # rew_i = np.dot(rew_n[i], choose_n[i])
                # print('rew_i= ', rew_i)
                agent.experience(obs_n[i], action_n[i], rew_n[i], choose_n[i], new_obs_n[i], done_n[i], terminal)
                # print('action_n[%d]= ' % i, action_n[i])#
                # print('rew_n[%d]= ' % i, rew_n[i])  #
            print('choose_n= ', choose_n)

            team_agents = []
            for i in range(len(choose_n[0])):
                index_agents = [j for j, choose in enumerate(choose_n) if choose[i] == 1]
                team_agents.append(index_agents)
            print('team_agents: ', team_agents)

            if train_step % 4 == 0:#########自己加的，从其他同类agent学习experience
                for j in range(len(team_agents)):
                    if team_agents[j]==[]:
                        continue
                    else:
                        Ti_max = np.argmax([rew_n[i][j] for _, i in enumerate(team_agents[j])])
                        i_max = team_agents[j][Ti_max]
                        # i_max = np.argmax([rew_n[i][j] for i in range(env.n)])
                        print('For team-%d, agent-%d get the max reawrd. '% (j, i_max))
                        for _, i in enumerate(team_agents[j]):  #
                            agent = trainers[i]
                            # i_rand = random.randint(0, env.n - 1)###随机分享
                            # agent.experience(obs_n[i_rand], action_n[i_rand], rew_n[i_rand], new_obs_n[i_rand], done_n[i_rand], terminal)#
                            agent.experience(obs_n[i_max], action_n[i_max], rew_n[i_max], choose_n[i_max], new_obs_n[i_max], done_n[i_max], terminal)

            obs_n = new_obs_n#更新状态观测

            for i, rew in enumerate(rew_nm):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:###依据terminal重开始
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.2)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver) #
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

    # with tf.Session() as sess:
    #     tf.summary.FileWriter("logs_tt/",sess.graph)

if __name__ == '__main__':
    arglist = parse_args() #定义参数列表并赋值
    train(arglist)
