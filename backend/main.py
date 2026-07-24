# Copyright 2025 Paul Richter

# info (paul): the new from august 2025

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import yfinance as yf
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import torch

import os
import torch
from torch import nn
import time

#19072026 from torch.utils.data import DataLoader
# from torchvision import datasets
#19072026 import torchvision.datasets as datasets
#19072026 import torchvision.transforms as transforms

# from torchvision import datasets, transforms

learning_rate = 1e-2#23082025 1e-3
batch_size = 64
epochs = 5

standard_chunck = 0.01

input_len = 100  # 30062025 4000#30#15082024 8#8
output_len = 2  # 15082024 8#2
merge = 1  # 1

stock_label = "DOW"#"INL"#"VW"#"INL"#"NVDA"#"DJI"
initial_cash = 0.1

class Data:
    times = None
    vals = None

# info (paul): "fund": class, which represents, how much stock of which asset is owned in style: (stock-label: amount)
class Fund:

    dic = None

    def __init__(self):
        self.dic = {"cash": 0.}


stock_period = '10y'
stock_interval = '1d'#'1w'

def read_stock(stock_name, num_vals, norm=False, merge=1):
    # info (paul): "merge": take the mean of this number of values
    #       each time to make a more generalized smoother curve

    #07082025 stock_name = "DJI"

    # data = yf.Ticker(stock_name).history(period='20y', interval='1m',)
    # data = yf.Ticker(stock_name).history(start='2012-01-01', end='2022-01-01', interval="1y")
    data = yf.download(stock_name, period=stock_period, interval=stock_interval)

    # info (paul): convert data object to list of keys (times) and vals (stock prices)
    data_open = data['Open']
    keys_l_stamps = list(data_open[stock_name].keys())
    vals_l = list(data_open.values)
    keys_l = [el.timestamp() / 86400 for el in keys_l_stamps]

    # info (paul): reduce number of values in keys and vals by dividing into bundles and replacing
    #       each bundle by mean value
    keys_merged, vals_merged = merge_vals(keys_l, vals_l, merge)
    keys_l, vals_l = keys_merged[:num_vals], vals_merged[:num_vals]

    # info (paul):
    if num_vals >= 0:
        keys_curtailed, vals_curtailed = keys_l[:num_vals], vals_l[:num_vals]
    else:
        keys_curtailed, vals_curtailed = keys_l, vals_l

    # info (paul): normalize (scale: 0, max), if norm==True:
    if norm:
        # keys_max = max(keys_curtailed)
        vals_max = max(vals_curtailed)

        # keys_curtailed = [el/keys_max]
        vals_curtailed = [el / vals_max for el in vals_curtailed]

    return keys_curtailed, vals_curtailed


def merge_vals(keys_l, vals_l, merge):
    # info (paul): merge values, so that each bundle of values is replaced
    #           by its mean value, i.e. "[1, 2.5, 4]" becomes "[2.5]"
    len_keys = len(keys_l)
    bundles = int(len_keys / merge)

    keys_merged = []
    vals_merged = []

    for bundle in range(bundles):
        mean_x, mean_y = mean_from_bundle(keys_l, vals_l, bundle, merge)
        keys_merged.append(mean_x)
        vals_merged.append(mean_y)

    return keys_merged, vals_merged

def mean_from_bundle(keys_l, vals_l, bundle, merge):
    sum_x = 0
    sum_y = 0

    for i in range(merge):
        sum_x += keys_l[merge * bundle + i]
        sum_y += vals_l[merge * bundle + i]

    mean_x = sum_x / merge
    mean_y = sum_y / merge

    return mean_x, mean_y

def mean():
    _ = 1 + 1


def read_stock_1():
    # info (paul): try to read finance
    data = yf.Ticker("NVDA").history(period='30y')

    data_open = data['Open']
    keys_l = list(data_open.keys())
    vals_l = list(data_open.values)
    plt.plot(keys_l, vals_l)
    #19072026 plt.show()
    _ = 1 + 1


def f_linear(x, a, b):
    result = [a * x_el + b for x_el in x]
    return result

def lin_prog(data):
    # info (paul): linear progression

    times_past = data.times_past
    times = data.times
    vals = data.vals_past

    popt, pcov = sp.optimize.curve_fit(f_linear, times_past, vals)

    plt.plot(times_past, vals, '.')
    plt.plot(times, f_linear(times, *popt))  # , 'g--',
    # label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.show()
    _ = 1 + 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_len)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(
        fig1,
        ax1,
        data,
        model,
        loss_fn,
        optimizer,
        fund=None,
        fund_linear=None,
        progress_callback=None,
):
    """

    :return:
    """

    stock_history = list(data.keys())

    # size = len(dataloader.dataset)
    size = len(stock_history)
    model.train()

    losses = []
    losses_simple = []

    cash_history = []
    cash_history_linear = []

    overperform_simple = []
    overperform_linear = []

    # for batch, (X, y) in enumerate(dataloader):
    for i, stock_history_vals in enumerate(stock_history):
        # info (paul): "key" is the last 100 values of the stock;
        #              data[key] is kind of the next value of the stock;
        #


        X = stock_history_vals
        y = data[stock_history_vals]
        batch = i

        pred = model(X)
        loss = loss_fn(pred, y)
        loss_simple = torch.sum(torch.abs(pred - y))

        # info (paul): Trade
        fund = trade_simple(X, model, fund)
        net_worth = compute_net_worth(fund, price=X[len(X)-1])
        cash_history.append(net_worth)

        # info (paul): Trade
        fund_linear = trade_linear(X, model, fund_linear)
        net_worth_linear = compute_net_worth(fund_linear, price=X[len(X) - 1])
        cash_history_linear.append(net_worth_linear)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0 or i == size - 1:  # batch%1==0:
            loss_val, current = loss.item(), batch * batch_size + len(X)
            print("\n loss: " + str(loss_val) + "; current: " + str(current) + "; size: " + str(size))
            losses.append(loss_val)
            losses_simple.append(loss_simple)

        stock_history_l = stock_history_vals.tolist()
        pred_l = pred.tolist()

        # plt.plot(range(0, len(key_k)), key_k, color="blue")
        # plt.plot(range(len(key_k), len(key_k) + len(pred_l)), pred_l, color="orange")
        # plt.show()
        _ = 1 + 1

        plot_len_val = min(len(stock_history_vals), 100)  # A16072025 20:

        # info (paul): compute overperformance
        start_key = stock_history_l[0]
        stock_growth = stock_history_l[-1]/start_key

        overperform_simple.append(((cash_history[-1]/cash_history[0])/stock_growth).item())
        overperform_linear.append(((cash_history_linear[-1]/cash_history_linear[0])/stock_growth).item())

        if progress_callback is not None:
            # Send every tenth step and always send the final step.
            if i % 2 == 0 or i == size - 1:
                cash_value = cash_history[-1]
                cash_linear_value = cash_history_linear[-1]

                if isinstance(cash_value, torch.Tensor):
                    cash_value = cash_value.detach().cpu().item()

                if isinstance(cash_linear_value, torch.Tensor):
                    cash_linear_value = (
                        cash_linear_value.detach().cpu().item()
                    )

                progress_callback({
                    "step": i + 1,
                    "total": size,
                    "loss": float(loss.item()),
                    "loss_simple": float(
                        loss_simple.detach().cpu().item()
                    ),
                    "cash": float(cash_value),
                    "cash_linear": float(cash_linear_value),
                    "overperform_simple": float(
                        overperform_simple[-1]
                    ),
                    "overperform_linear": float(
                        overperform_linear[-1]
                    ),
                })

        if False:  # Ai > len(keys) - plot_len_val:
            # Amanage_plot(fig1, ax1, key_l, pred_l, y, losses[(len(keys) - plot_len_val):],
            # A            losses_simple[(len(keys) - plot_len_val):])
            manage_plot(fig1, ax1, stock_history_l, pred_l, y, losses[:],
                        losses_simple[:], cash_history, cash_history_linear, overperform_simple, overperform_linear)
            _ = 1 + 1

    # 16072025 #plt.plot(losses)
    # 16072025 plt.show()

    return {
        "losses": [float(value) for value in losses],
        "losses_simple": [
            float(value.detach().cpu().item())
            for value in losses_simple
        ],
        "cash_history": [
            float(value.detach().cpu().item())
            if isinstance(value, torch.Tensor)
            else float(value)
            for value in cash_history
        ],
        "cash_history_linear": [
            float(value.detach().cpu().item())
            if isinstance(value, torch.Tensor)
            else float(value)
            for value in cash_history_linear
        ],
        "overperform_simple": [
            float(value) for value in overperform_simple
        ],
        "overperform_linear": [
            float(value) for value in overperform_linear
        ],
    }

def test_loop(fig1, ax1, data, model, loss_fn, fund):
    """

    :return:
    """

    keys = list(data.keys())
    size = len(keys)
    model.eval()

    # size = len(dataloader.dataset)
    num_batches = 1  # len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        # for X, y in data:
        for i, key in enumerate(keys):
            X = key
            y = data[key]
            batch = i

            pred = model(X)

            trade_1_stock(X, model, fund)

            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            key_k = key.tolist()
            pred_l = pred.tolist()

            manage_plot(fig1, ax1, key_k, pred_l, y)
            _ = 1 + 1

    test_loss /= num_batches


def manage_plot(fig1, ax1, key, pred, y_vals, losses=None, losses_simple=None, cash_history=None, cash_history_linear=None,
        overperform_simple=None, overperform_linear=None):
    # ax1.clf()
    ax1.plot(range(0, len(key)), key, color="blue", label='past stocks')
    ax1.plot(range(len(key), len(key) + len(pred)), pred, color=(1.0, 0.85, 0.7, 1.0), label='predictions')
    ax1.plot(range(len(key), len(key) + len(pred)), y_vals, color="orange", label='ref')
    if losses != None:
        ax1.plot(range(len(key) + len(pred) - len(losses_simple), len(key) + len(pred)),
                 [el.detach().numpy() for el in losses_simple], color=(0.8, 0.2, 0.2, 1.0), label='loss')

    if cash_history != None:
        ax1.plot(range(len(key) + len(pred) - len(cash_history), len(key) + len(pred)), cash_history, color="black", label="simple")
    if cash_history_linear != None:
        ax1.plot(range(len(key) + len(pred) - len(cash_history_linear), len(key) + len(pred)), cash_history_linear, color="gray", label="linear")

    if overperform_simple != None:
        ax1.plot(range(len(key) + len(pred) - len(cash_history_linear), len(key) + len(pred)), overperform_simple,
                 color="gray", label="over simple", linestyle="--")
    if overperform_linear != None:
        ax1.plot(range(len(key) + len(pred) - len(cash_history_linear), len(key) + len(pred)), overperform_linear,
                 color="gray", label="over linear", linestyle="--")

    ax1.legend()
    #19072026 fig1.canvas.draw()
    #19072026 fig1.canvas.flush_events()
    #19072026 time.sleep(0.1)  # (2)
    plt.cla()

    # ax1.show(block=False)
    _ = 1 + 1
    _ = 1 + 1

def to_list(vals_0_pre):
    vals_0 = [el[0] for el in vals_0_pre]
    return vals_0


def init_linear(keys_0):
    # info (paul): a: slope, b: offset
    vals_0 = f_linear(keys_0, 1. / (max(keys_0) - min(keys_0)), 0.)

    return vals_0


def smoothen_curve(vals_0):
    # info (paul): smoothen curve by taking the mean, also intending
    #       to be in direction "200-day-curve, 50-day-curve" stuff

    vals = []
    span = 1#10
    for i in range(len(vals_0)):
        start_idx = max(0, i - span)
        end_idx = min(len(vals_0), i + span)
        new_val = np.mean(vals_0[start_idx:end_idx])
        vals.append(new_val)

    return vals


def find_training_data(input_len, ys_len, merge=-1, input_stock=None):
    # info (paul): obtaining data
    # Input:
    # Output: train_els, testels: Each a dictionary, where .keys() yields a list of
    #       Lists. Each element is  a list of the most recent stock values (or the n-th-most recent)

    # trains = 10000
    train_els = {}
    test_els = {}

    # info (paul): get the manual stocks
    keys_all = []
    vals_all = []
    abbrs = ["DJI", "DJI"]
    if input_stock != None:
        abbrs = [input_stock]
    for abbr in abbrs:
        keys_0, vals_0_pre = read_stock(abbr, -1, norm=True, merge=merge)  # NVDA
        vals_0 = to_list(vals_0_pre)

        # info (paul): overwrite with simple values
        vals_0 = smoothen_curve(vals_0)
        # vals_0 = init_linear(keys_0)

        keys_all.append(keys_0)  # 16072025 , keys_1, keys_2]
        vals_all.append(vals_0)  # 16072025 , vals_1, vals_2]

    # info (paul): number of training iterations
    trains = len(keys_0) - input_len - ys_len

    # info (paul): generate random offsets
    import random
    g_cpu = torch.Generator()
    g_cpu.manual_seed(20)
    rands = 6 * torch.rand(trains, generator=g_cpu)  # i*0.1

    # info (paul): prepare the training data as list of input, output blocks
    for abbr_idx in range(len(keys_all)):
        for train_idx in range(trains):
            offset = rands[train_idx]
            keys = keys_all[abbr_idx]
            vals = vals_all[abbr_idx]

            xs_raw = init_from_stock(0 + train_idx, input_len + train_idx, offset, keys_l=keys, vals_l=vals)
            ys_raw = init_from_stock(input_len + train_idx, input_len + ys_len + train_idx, offset, keys_l=keys,
                                     vals_l=vals)
            xs = torch.FloatTensor(xs_raw)
            ys = torch.FloatTensor(ys_raw)
            train_els.update({xs: ys})
            if train_idx % 100 == 0 or train_idx == trains - 1:
                print(f"Preparing training data: {train_idx + 1}/{trains}")

    xs_raw = init_from_stock(0, input_len, 0.1 * (trains + 1), keys_l=keys, vals_l=vals)
    ys_raw = init_from_stock(input_len, input_len + ys_len, 0.1 * (trains + 1), keys_l=keys, vals_l=vals)

    xs = torch.FloatTensor(xs_raw)
    ys = torch.FloatTensor(ys_raw)
    test_els.update({xs: ys})



    return train_els, test_els


def init_lin(min_val, max_val, offset):
    xs = [i + offset for i in range(min_val, max_val)]
    return xs


def init_sq(min_val, max_val, offset):
    xs = [offset + i * i for i in range(min_val, max_val)]
    return xs


def init_cu(min_val, max_val, offset):
    xs = [offset + i * i - 0.05 * i * i * i for i in range(min_val, max_val)]
    return xs


def init_from_stock(min_val, max_val, offset, keys_l=None, vals_l=None):
    if keys_l == None:
        keys, vals = read_stock("NVDA", max_val)
    else:
        keys, vals = keys_l, vals_l

    xs = [vals[i] for i in range(min_val, max_val)]
    return xs


def sq_loss_func(pred, truth):
    return torch.subtract(pred, truth)


def correlate(stock1, stock2):
    # info (paul): find the correlation distribution between two stock histories

    # info (paul): prepare data
    keys1 = list(stock1.keys())
    vals1 = [el[0].item() for el in list(stock1.values())]
    keys2 = list(stock2.keys())
    vals2 = [el[0].item() for el in list(stock2.values())]

    # info (paul): get diffgerence curves
    diff1 = []
    diff2 = []
    diff3 = []
    for idx in range(min([len(vals1)-1, len(vals2)-2])):
        diff1.append(vals1[idx + 1] - vals1[idx])
        diff2.append(vals2[idx + 1] - vals2[idx])
        diff3.append(vals2[idx + 2] - vals2[idx+1])



    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for idx in range(len(diff1)):
        ax1.plot(diff1[idx], diff3[idx], '.', color='blue')

    pearson = sp.stats.pearsonr(diff1, diff3)
    print("pearson: " + str(pearson))

    plt.show()
    _ = 1+1


def self_correlate(stock):

    prod1 = []
    for i in range(len(stock)-2):
        diff1 = stock[i+1] - stock[i]
        diff2 = stock[i+2] - stock[i+1]
        prod1.append(diff1 * diff2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    #for idx in range(len(prod1)):
    ax1.plot(list(range(len(stock))), stock, '.', color='orange', ms=1)
    ax1.plot(list(range(len(prod1))), prod1, '.', color='blue', ms=1)

    plt.show()

    return prod1

def net_1(progress_callback=None):
    # info (paul): params, network
    net1 = NeuralNetwork(input_len, output_len)
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net1.parameters(), lr=learning_rate)

    # info (paul): get data
    #23082025 training_data1, test_data = find_training_data(input_len, output_len, merge=merge,
    #23082025     input_stock="NVDA")
    training_data2, test_data = find_training_data(input_len, output_len, merge=merge,
        input_stock=stock_label)

    # info (paul): set up fund
    fund = Fund()
    fund.dic["cash"] = initial_cash
    fund.dic.update({stock_label: 0.})

    fund_linear = Fund()
    fund_linear.dic["cash"] = initial_cash
    fund_linear.dic.update({stock_label: 0.})

    # info (paul): the correlation stuff
    if False:
        correlate(training_data1, training_data2)

        training1_list = [el[0].item() for el in list(training_data1.values())]
        training2_list = [el[0].item() for el in list(training_data2.values())]

        self_correlate(training1_list)#not now, other time

    # info (paul): train, eval
    #19072026 plt.ion()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    result = train_loop(
        fig1,
        ax1,
        training_data2,
        net1,
        loss_fn,
        optimizer,
        fund,
        fund_linear,
        progress_callback=progress_callback,
    )

    plt.close(fig1)

    return {
        "stock": stock_label,
        "epochs": epochs,
        **result,
    }

# info (paul): simulate trading:

def compute_net_worth(fund, price=None):
    # info (paul): compute the fund, assuming there is only one value

    keys = fund.dic.keys()

    net_worth = fund.dic["cash"]

    for key in keys:
        if key != "cash":
            net_worth += fund.dic[key]*price

    return net_worth

def buy(fund, stock_label, price=None, perc=0.1):
    amount = perc#perc*fund[key]
    fund.dic["cash"] -= amount*price
    fund.dic[stock_label] += amount
    return fund

def sell(fund, stock_label, price=None, perc=0.1):
    amount = perc #perc*fund.dic[key]
    fund.dic["cash"] += amount*price
    fund.dic[stock_label] -= amount
    return fund

def trade_simple(price_history, model, fund):
    # info (paul): easy strategy to trade, if there is only one stock;
    #       We assume, that the fund has only one relevant stock

    keys = list(fund.dic.keys())
    key = keys[1]

    price_now = price_history[len(price_history)-1]
    pred = model(price_history)
    profit = pred[0].item() - price_now.item()

    if profit > 0:
        fund = buy(fund, key, price=price_now.item(), perc=standard_chunck)
    if profit < 0:
        fund = sell(fund, key, price=price_now.item(), perc=standard_chunck)
    return fund

def trade_linear(price_history, model, fund):
    # info (paul): easy strategy to trade, if there is only one stock;
    #       We assume, that the fund has only one relevant stock

    keys = list(fund.dic.keys())
    key = keys[1]

    price_now = price_history[len(price_history) - 1]
    pred = model(price_history)
    profit = pred[0].item() - price_now.item()

    if profit > 0:
        fund = buy(fund, key, price=price_now.item(), perc=10*standard_chunck*profit)
    if profit < 0:
        fund = sell(fund, key, price=price_now.item(), perc=10*standard_chunck*profit)
    return fund

def trade_2_stock(price_history, model, fund):

    _ = 1+1
    _ = 1 + 1
    _ = 1+1

if __name__ == '__main__':
    net_1()

    print("\n net_1 done")
    # 11082024 keys, vals = read_stock("NVDA", 100)
    # 11082024
    # 11082024 data = Data()
    # 11082024 data.times_past = np.array(keys)
    # 11082024 data.times = np.append(data.times_past, 16500)#np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # 11082024 data.vals_past = np.array(vals)
    # 11082024 #read_stock_1()
    # 11082024
    # 11082024 lin_prog(data)