import os
from keras.losses import mean_squared_error
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


class Tools:
    qfnet_data = 'DQN_data_xxx'
    qf_data = qfnet_data + '/qf_data'
    qf_image = qfnet_data + '/qf_image'
    qf_model = qfnet_data + '/qf_model'
    ql_model_pro = qfnet_data + '/ql_model_pro'
    ql_model = qfnet_data + '/ql_model'

    @classmethod
    def create_dirs(cls):
        if not (os.path.exists(cls.qfnet_data)):
            os.mkdir(cls.qfnet_data)
        if not (os.path.exists(cls.qf_data)):
            os.mkdir(cls.qf_data)
        if not (os.path.exists(cls.qf_image)):
            os.mkdir(cls.qf_image)
        if not (os.path.exists(cls.qf_model)):
            os.mkdir(cls.qf_model)
        if not (os.path.exists(cls.ql_model_pro)):
            os.mkdir(cls.ql_model_pro)
        if not (os.path.exists(cls.ql_model)):
            os.mkdir(cls.ql_model)

    @staticmethod
    def adjust_loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + 1 / K.mean(y_pred, axis=-1, keepdims=True)
        # return mean_squared_error(y_true, y_pred)

    @staticmethod
    def compute_capacity_use_tf(y_pred, net):
        '''计算总个网络的吞吐量'''
        interfence = net.compute_interference()
        # power_subcarrier = self._compute_power()
        interfence_p = np.dot(interfence, y_pred)  
        BS_power = K.sum(y_pred, axis=-1, keepdims=True)
        return net.B * K.log(1 + np.abs(np.sum(BS_power, axis=-1) + 1e-8) / (np.sum(interfence_p) + net.N0))

    @staticmethod
    def my_metrics(y_true, y_pred):
        return K.mean(1 / (K.abs(y_pred - y_true) + 1))

    @staticmethod
    def get_epoch_data(net):
        bs_ue_number = net.eachBS_number.flatten()
        CSI = net.CSI.flatten()
        subcarrier_power = net.power_subcarrier.flatten()
        epoch_data = np.hstack((bs_ue_number, subcarrier_power, CSI))
        epoch_data = epoch_data.reshape((1, 14880))
        return epoch_data


def is_gpu_available():
    is_gpu = tf.test.is_gpu_available(True)
    return is_gpu


def get_available_gpus():
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    gpu_name = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_name)  


def set_gpu():
    if is_gpu_available():
        gpus = get_available_gpus()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(['{i}'.format(i=a) for a in range(gpus)]) 
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  
        session = tf.Session(config=config)
        KTF.set_session(session)


def str_map_float(str_array):
    nums = []
    for strs in str_array:
        nums.append(float(strs))
    return nums
