import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import csv
import os, sys

def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]
    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def smoothExponential(data, weight):
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--env', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    data_dir = os.path.join(args.dir, args.env)
    data_list = []
    for file in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, file)):
            continue

        policy_name = os.path.splitext(file)[0]
        policy_csv = pd.read_csv(os.path.join(data_dir,file))
        data_list.append((policy_name, policy_csv))
    print("found {} records".format(len(data_list)))

    color_list = ['#0075DC','#191919','#00998F','#50EBEC','#FFA405','#FF0180']

    fig = go.Figure()
    for i in range(len(data_list)):
        policy, record = data_list[i][0], data_list[i][1]
        fig.add_trace(go.Scatter(x = record['Step'], y = smoothExponential(record['Value'],0.99),name=policy, marker=dict(color=color_list[i])))

    fig.update_layout(
        title="Training Performance",
        xaxis_title="Episodes",
        yaxis_title="Total Reward",
        legend_title="Vision Input",
        font=dict(
            family="Arial",
            size=20,
            color="Black"
        ),
        plot_bgcolor="rgb(255,255,255)",
        xaxis = dict(
        tickmode = 'array',
        tickvals = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        ticktext = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']
        )
    )
    fig.show()
