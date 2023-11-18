import time
import socket
import collections
import numpy as np
import pandas as pd
from json import loads
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import raw_to_waves, calculate_sum_waves


def live_plot(data_dict):
    plt.cla()
    for label, data in data_dict.items():
        if data:
            x, y = zip(*data)
            plt.plot(x, y, label=label)
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.xlabel('Epoch')
    plt.ylabel('Amplitude')


data_plt = collections.defaultdict(list)
client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
client_socket.connect(('127.0.0.1', 13854))
time.sleep(0.1)

SAMPLE_RATE = 512
data_raw = []
file_path = 'data_characters_classification.csv'

label = 0
number_of_samples = 40
duration = 3  # seconds

data_train = []
data_sample = []

def save_data():
    # save data_train to csv file
    df = pd.DataFrame(data_train, index=None)
    df.to_csv(file_path, index=False, header=False, mode='a')

def get_waves(i):
    client_socket.sendall(bytes('{"enableRawOutput":true,"format":"Json"}'.encode('ascii')))
    data_set = client_socket.recv(1000)
    data_set = (str(data_set)[2:-3].split(r'\r'))

    for data in data_set:
        try:
            json_data = loads(data)
            temp_data = json_data['rawEeg']
            if temp_data:
                data_raw.append(temp_data)
            if len(data_train) == number_of_samples:
                save_data()
                data_train.clear()
                exit()
            if len(data_raw) > SAMPLE_RATE:
                waves_data = data_raw[:SAMPLE_RATE]
                data_raw.clear()
                waves_data = raw_to_waves(waves_data)
                sum_waves = calculate_sum_waves(waves_data["gamma"], waves_data["beta"])
                data_plt["resonance"] = sum_waves
                x, y = zip(sum_waves)
                data_sample.extend(y)
                if len(data_sample) / SAMPLE_RATE >= duration:
                    np_data = np.array(data_sample)
                    np_data = np_data.reshape(-1, 512) # (3,512)
                    np_data = np_data.mean(axis=0) # (1,512)
                    np_data = np.around(np_data, 3)
                    np_data = np_data.tolist()
                    np_data.append(label) # (1,513)
                    data_train.append(np_data)
                    data_sample.clear()
        except Exception as e:
            # print(e)
            continue
    live_plot(data_plt)


fig = plt.figure(figsize=(6, 2))
plt.grid(True)
ani = FuncAnimation(fig, get_waves, interval=100)
plt.show()
