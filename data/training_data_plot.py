import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import csv

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

if __name__ == "__main__":
    dqn_csv = pd.read_csv('./lunarlander/dqn.csv')
    ddpg_csv = pd.read_csv('./lunarlander/ddpg.csv')
    td3_csv = pd.read_csv('./lunarlander/td3.csv')
    sac_csv = pd.read_csv('./lunarlander/sac.csv')
    ppo_csv = pd.read_csv('./lunarlander/ppo.csv')
    vpg_csv = pd.read_csv('./lunarlander/vpg.csv')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dqn_csv['Step'], y = smoothExponential(dqn_csv['Value'],0.99),name='DQN', marker=dict(color='#0075DC')))
    fig.add_trace(go.Scatter(x = ddpg_csv['Step'], y = smoothExponential(ddpg_csv['Value'],0.99),name='DDPG', marker=dict(color='#191919')))
    fig.add_trace(go.Scatter(x = td3_csv['Step'], y = smoothExponential(td3_csv['Value'],0.99),name='TD3', marker=dict(color='#00998F')))
    fig.add_trace(go.Scatter(x = sac_csv['Step'], y = smoothExponential(sac_csv['Value'],0.99),name='SAC', marker=dict(color='#50EBEC')))
    fig.add_trace(go.Scatter(x = ppo_csv['Step'], y = smoothExponential(ppo_csv['Value'],0.99),name='PPO', marker=dict(color='#FFA405')))
    fig.add_trace(go.Scatter(x = vpg_csv['Step'], y = smoothExponential(vpg_csv['Value'],0.99),name='VPG', marker=dict(color='#FF0180')))
    fig.update_layout(
        title="LunarLander RL Training Performance",
        xaxis_title="Episodes",
        yaxis_title="Total Reward",
        legend_title="PPO Policies",
        font=dict(
            family="Arial",
            size=20,
            color="Black"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()
