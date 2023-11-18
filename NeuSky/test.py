import time
import socket
import collections
import numpy as np
from json import loads
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import raw_to_waves, calculate_sum_waves
from train import MindWaveClassify


def live_plot(data_dict):
    plt.cla()
    for label, data in data_dict.items():
        if data:
            x, y = zip(*data)
            plt.plot(x, y, label=label)
    plt.legend(loc='center left')  # the plot evolves to the right
    plt.xlabel('epoch')
    plt.ylabel('Amplitude')


data_plt = collections.defaultdict(list)
client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
client_socket.connect(('127.0.0.1', 13854))
time.sleep(0.1)

SAMPLE_RATE = 512
duration = 1  # seconds
data_raw = []
data_sample = []

model = MindWaveClassify()

DATA_LABEL = ["unknown", "A", "B"]

def get_waves(i):
    client_socket.sendall(bytes('{"enableRawOutput":true,"format":"Json"}'.encode('ascii')))
    data_set = client_socket.recv(1000)
    data_set = (str(data_set)[2:-3].split(r'\r'))
    # print(data_set)

    for data in data_set:
        try:
            json_data = loads(data)
            temp_data = json_data['rawEeg']
            if temp_data:
                data_raw.append(temp_data)
            print(len(data_raw), SAMPLE_RATE)
            if len(data_raw) > SAMPLE_RATE:
                waves_data = data_raw[:SAMPLE_RATE]
                data_raw.clear()
                waves_data = raw_to_waves(waves_data)
                sum_waves = calculate_sum_waves(waves_data["gamma"], waves_data["beta"])
                data_plt["resonance"] = sum_waves
                x, y = zip(*sum_waves)
                data_sample.extend(y)
                if len(data_sample) / SAMPLE_RATE >= duration:
                    np_data = np.array(data_sample)
                    np_data = np_data.reshape(-1, 512)
                    np_data = np_data.mean(axis=0)
                    np_data = np.around(np_data, 3)
                    pred_label = model.predict(np_data)
                    max_label = np.argmax(pred_label, axis=1)
                    # if max_label[0] != 0 and pred_label[0][max_label[0]] > 0.5:
                    print(f"Predict: {DATA_LABEL[max_label[0]]}, confidence: {pred_label[0][max_label[0]]}")
                    data_sample.clear()
        except Exception as e:
            print(e)
            continue
    live_plot(data_plt)


fig = plt.figure(figsize=(6, 2))
plt.grid(True)
ani = FuncAnimation(fig, get_waves, interval=100)
plt.show()
