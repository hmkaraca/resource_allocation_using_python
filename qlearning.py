from dlmodel import *
import shelve
from utils import *
from network import *
import re
import time as t
import copy
from collections import defaultdict


class Qlearning:
    '''
        Q-learning Based Distributed Resource Allocation
    '''

    def __init__(self, alpha=0.5, gama=0.9, A=(25, 20, 15, 10, 5), pmax=30, kesi=0.2, deta_p=2):
        self.net = None
        self.alpha = alpha
        self.gama = gama
        [self.A1, self.A2, self.A3, self.A4, self.A5] = A
        self.pmax = pmax
        self.kesi = kesi
        self.Q = []
        # self.BS_power = net.BS_power
        self.deta_p = deta_p
        self.count = 0
        self.qfnet = DLModel()
        # for i in range(net.expectBS):
        #     self.Q.append(defaultdict(list))

    def compute_reward(self, n, action):
        assert action >= 0 and action < self.net.subcarrier
        if self.net.BS_power[n] <= self.pmax:
            Cnk = self.net.compute_capacity_on_subcarrier(n, action)
            return 1 - np.power(np.e, -Cnk)
        return -1

    def update_Q(self, n, action, bool_p):
        old_state = str(self.net.eachBS_number[n]) + ':' + str(self.power_levels(n))   
        if not (old_state in self.Q[n]):
            self.Q[n][old_state] = np.zeros((self.net.subcarrier,))
        # print(self.Q[n])
        # self.Q[n][state][action] += self.compute_reward(n, action)
        if bool_p == 1:
            self.net.power_subcarrier[n, action] += self.deta_p
            # self.net.BS_power[n] += self.deta_p
        elif bool_p == -1:
            self.net.power_subcarrier[n, action] -= self.deta_p
            # self.net.BS_power[n] -= self.deta_p
        elif bool_p == 0:
            pass
        self.net.BS_power = np.sum(self.net.power_subcarrier, axis=1)
        new_state = str(self.net.eachBS_number[n]) + ':' + str(self.power_levels(n))  
        reward = self.compute_reward(n, action)
        temp_Q = 0
        for j in range(self.net.expectBS):
            if j != n:
                if new_state in self.Q[j]:
                    if self.Q[j][new_state][action] > temp_Q:
                        temp_Q = self.Q[j][new_state][action]
        self.Q[n][old_state][action] = (1 - self.alpha) * self.Q[n][old_state][action] + \
                                       self.alpha * (reward + self.gama * temp_Q)

    def power_levels(self, n):
#        if 0 <= self.net.BS_power[n] < self.pmax - 28:
#            return 0
#        # elif self.net.BS_power[n] >= self.pmax - self.A2 and self.net.BS_power[n] < self.pmax:
#        elif self.net.BS_power[n] < self.pmax - 26:
#            return 1
#        elif self.net.BS_power[n] < self.pmax - 24:
#            return 2
#        elif self.net.BS_power[n] < self.pmax - 22:
#            return 3
#        elif self.net.BS_power[n] < self.pmax - 20:
#            return 4
#        elif self.net.BS_power[n] <= self.pmax - 18:
#            return 5
#        elif self.net.BS_power[n] < self.pmax - 16:
#            return 6
#        elif self.net.BS_power[n] < self.pmax - 14:
#            return 7
#        elif self.net.BS_power[n] < self.pmax - 12:
#            return 8
#        elif self.net.BS_power[n] < self.pmax - 10:
#            return 9
#        elif self.net.BS_power[n] <= self.pmax - 8:
#            return 10
#        elif self.net.BS_power[n] < self.pmax - 6:
#            return 11
#        elif self.net.BS_power[n] < self.pmax - 4:
#            return 12
#        elif self.net.BS_power[n] <= self.pmax - 2:
#            return 13
#        else:
#            return 14
        if 0 <= self.net.BS_power[n] < self.pmax - self.A1:
            return 0
        # elif self.net.BS_power[n] >= self.pmax - self.A2 and self.net.BS_power[n] < self.pmax:
        elif self.net.BS_power[n] < self.pmax - self.A2:
            return 1
        elif self.net.BS_power[n] < self.pmax - self.A3:
            return 2
        elif self.net.BS_power[n] < self.pmax - self.A4:
            return 3
        elif self.net.BS_power[n] < self.pmax - self.A5:
            return 4
        elif self.net.BS_power[n] <= self.pmax:
            return 5
        else:
            return 5

    def choose_action(self, n, qfnet_outs, C=0.7, kesi=0.2):
        prob = np.abs(np.random.rand())
        if prob <= kesi:  # exploration
            action = np.random.randint(0, self.net.subcarrier)
        else:  # exploitation
            temp_action = np.zeros(self.net.subcarrier)
            for i in range(self.net.subcarrier):
                for j in range(self.net.expectBS):
                    state = str(self.net.eachBS_number[j]) + ':' + str(self.power_levels(j))
                    if j != n:  
                        if state in self.Q[j]:  
                            # print('state', state)
                            # print('Q: ', self.Q[j])
                            # print(self.Q[j][state][i])
                            temp_action[i] += self.Q[j][state][i]
            temp_action = C * qfnet_outs + (1 - C) * temp_action
            action = np.argmax(temp_action)
        temp = self.net.power_subcarrier[n, action]
        dis_arg = np.argsort(self.net.BS_BS_distance[n])
        capacity_add = 0
        capacity_minus = 0
        if self.net.BS_power[n] < self.pmax - self.deta_p:
            self.net.power_subcarrier[n, action] += self.deta_p
            for n_station in dis_arg[:5]:  
                capacity_add += self.net.compute_capacity_on_subcarrier(n_station, action)
        else:  
            self.net.power_subcarrier[n, action] = temp
        self.net.power_subcarrier[n, action] = temp
        if self.net.power_subcarrier[n, action] > self.deta_p:
            self.net.power_subcarrier[n, action] -= self.deta_p
            for n_station in dis_arg[:5]:  
                capacity_minus += self.net.compute_capacity_on_subcarrier(n_station, action)
        else:  
            self.net.power_subcarrier[n, action] = temp
        self.net.power_subcarrier[n, action] = temp
        if capacity_add > capacity_minus:
            return action, 1  
        elif capacity_add < capacity_minus:
            return action, -1  
        elif capacity_add == capacity_minus and capacity_add == 0:
            return action, 0  
        else:
            return action, 1  

    def remove_data(self):
        before_datas = os.listdir(Tools.qf_data)
        if len(before_datas) != 0:
            for data in before_datas:
                os.remove(Tools.qf_data + '/' + data)

    def save_data(self, filename, epoch_data):
        before_datas = os.listdir(Tools.qf_data)
        if len(before_datas) > 6:  
            tim = []
            for i in range(len(before_datas)):
                tim.append(int(before_datas[i].split('_')[2]))
            del_data = str(np.min(tim))
            for i in range(len(before_datas)):
                de_dat = re.search(r'.*_' + del_data + r'_.*', before_datas[i])
                if de_dat is not None:
                    os.remove(Tools.qf_data + '/' + de_dat.group())
        try:
            datas = shelve.open(Tools.qf_data + '/' + filename, writeback=True, flag='c')
            data = np.array(datas['epoch_data'])
            datas['epoch_data'] = np.vstack((data, epoch_data))
        except KeyError:
            datas['epoch_data'] = epoch_data
        finally:
            # print(filename, epoch_data.shape)
            datas.close()

    def save_qlmodel(self, ql_model, ql_model_path):
        filename = 'qlmodel_' + str(int(t.time())) + '_.db'
        ql_models = os.listdir(ql_model_path)
        if len(ql_models) >= 6:  
            tim = []
            for i in range(len(ql_models)):
                tim.append(int(ql_models[i].split('_')[1]))
            del_data = str(np.min(tim))
            for i in range(len(ql_models)):
                de_dat = re.search(r'.*_' + del_data + r'_.*', ql_models[i])
                if de_dat is not None:
                    os.remove(ql_model_path + '/' + de_dat.group())
        try:
            datas = shelve.open(ql_model_path + '/' + filename, writeback=True, flag='c')
            datas['ql_model'] = ql_model
        except KeyError:
            print('ql_model')
        finally:
            datas.close()

    def load_qlmodel(self, ql_model_path):
        ql_models = os.listdir(ql_model_path)
        if len(ql_models) == 0:
            return None
        tim = []
        for i in range(len(ql_models)):
            tim.append(int(ql_models[i].split('_')[1]))
        try:
            datas = shelve.open(ql_model_path + '/qlmodel_' + str(np.max(tim)) + '_.db', writeback=True, flag='c')
            ql_model = datas['ql_model']
            return ql_model
        except KeyError:
            return None
        finally:
            datas.close()

    def start(self, model_num, epoch=100, iteration=400, only_q=True):
        self.net = Network()
        self.net.start()
        if only_q:
            ql_model = self.load_qlmodel(Tools.ql_model)
        else:
            ql_model = self.load_qlmodel(Tools.ql_model_pro)
        if ql_model is not None:
            self.Q = ql_model
        else:
            for i in range(self.net.BSs):
                self.Q.append(defaultdict(list))

        epoch_time = t.time()
        self.count = 0
        epoch_data_filename = model_num + '_x_' + str(int(epoch_time)) + '_.bd'
        epoch_label_filename = model_num + '_y_' + str(int(epoch_time)) + '_.bd'
        flag = 0
        best_model = self.qfnet.load_best_model()
        while self.count < epoch:
            outs = np.zeros((1, self.qfnet.layers_units[0]))
            if not (only_q) and (best_model is not None):
                print('Qlearing')
                epoch_data = Tools.get_epoch_data(self.net)
                outs = self.qfnet.predict(best_model, epoch_data)
            iter = 0
            subcarrier_power_old = self.net.power_subcarrier
            print('subcarrier_power: \n', subcarrier_power_old)
            # sub_flag = 0
            while iter < iteration:
                print('*' * 50)
                print('epoch: ', self.count, ' iter: ', iter)
                subcarrier_power_old = self.net.power_subcarrier
                epoch_data = Tools.get_epoch_data(self.net)
                for i in range(self.net.BSs):
                    qfnet_out = outs[0, i * self.net.subcarrier: i * self.net.subcarrier + self.net.subcarrier]
                    if self.count * iter <= 30:
                        C = 0.8
                    elif self.count * iter <= 60:
                        C = 0.6
                    elif self.count * iter <= 100:
                        C = 0.4
                    else:
                        C = 0.2
                    action, bool_p = self.choose_action(i, qfnet_out, C=C)
                    self.update_Q(i, action, bool_p)
                subcarrier_power_new = self.net.power_subcarrier
                self.net.BS_power = np.sum(subcarrier_power_new, axis=1)
                outs_capacity = self.net.compute_capacity(
                    subcarrier_power_new, self.net.BS_power
                )
                if only_q:
                    with open(Tools.qfnet_data + '/' + model_num + '_only_c.txt', 'a') as only:
                        only.write(str(outs_capacity) + ',')
                else:
                    with open(Tools.qfnet_data + '/' + model_num + '_dqn_c.txt', 'a') as only:
                        only.write(str(outs_capacity) + ',')
                    epoch_label = subcarrier_power_new.flatten() - subcarrier_power_old.flatten()  
                    self.save_data(epoch_data_filename, epoch_data)
                    self.save_data(epoch_label_filename, epoch_label)
                # if np.sum((subcarrier_power_new - subcarrier_power_old) == 0) > self.net.expectBS * 2/3:  
                #                    sub_flag += 1
                #                    if sub_flag >=5:
                #                        break
                iter += 1
            self.count += 1
            subcarrier_power_new = self.net.power_subcarrier
            if np.sum(subcarrier_power_new - subcarrier_power_old) < 2:
                flag += 1
                if flag > 5:
                    break
            self.net.update()
        if only_q:
            self.save_qlmodel(self.Q, Tools.ql_model)
        else:
            self.save_qlmodel(self.Q, Tools.ql_model_pro)

    def ql_evaluate(self, q_model, net, iteration=400):
        self.Q = q_model
        self.net = copy.deepcopy(net)
        subcarrier_power_old = self.net.power_subcarrier.flatten()
        iter = 0
#        while iter < iteration:
        subcarrier_power = []
        subcarrier_power.append(subcarrier_power_old)
        while True:
            outs = np.zeros((1, self.qfnet.layers_units[0]))
            print(' iter: ', iter)

            for i in range(self.net.expectBS):
                qfnet_out = outs[0, i * self.net.subcarrier: i * self.net.subcarrier + self.net.subcarrier]
                action, bool_p = self.choose_action(i, qfnet_out, kesi=0)
                if bool_p == 1 and self.net.BS_power[i] < self.pmax - self.deta_p:
                    self.net.power_subcarrier[i, action] += self.deta_p
                    #self.net.BS_power[i] += self.deta_p
                elif bool_p == -1 and self.net.power_subcarrier[i, action] > self.deta_p:
                    self.net.power_subcarrier[i, action] -= self.deta_p
                    #self.net.BS_power[i] -= self.deta_p
                self.net.BS_power = np.sum(self.net.power_subcarrier, axis=1)
            iter += 1
            subcarrier_power_new = self.net.power_subcarrier.flatten()
            print(self.net.BS_power)
            print(np.mean(subcarrier_power_new - subcarrier_power[-1]))
            if np.mean(subcarrier_power_new - subcarrier_power[-1]) < 0.001:
                break
            subcarrier_power.append(subcarrier_power_new)
        return subcarrier_power_new - subcarrier_power_old
