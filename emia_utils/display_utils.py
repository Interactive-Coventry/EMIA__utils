import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from PIL import ImageDraw, ImageFont

from .process_utils import get_bus_arrival_index, exponential_smoothing, double_exponential_smoothing, edit_bus_arrivals


def plot_current_time_vs_diff(df):
    df_preprocessed = edit_bus_arrivals(df)

    timestamps1 = df_preprocessed.index
    timestamps2 = df_preprocessed['diffs']

    idx = get_bus_arrival_index(df_preprocessed)
    timestamps3 = timestamps1[idx]
    timestamps4 = timestamps2[idx]

    plot_title = 'Delay from Previous Estimation on Expected Arrival'

    fig, ax = plt.subplots()
    ax.plot(timestamps1, timestamps2)
    plt.plot(timestamps3, timestamps4, 'ro', label='bus change')

    ax.set_xlabel("Current Time")
    ax.set_ylabel("Difference from previous Estimated Arrival (sec)")
    ax.xaxis_date()
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.title.set_text(plot_title)
    plt.show()


def plot_current_time_vs_estimated_arrival(df):
    timestamps1 = df.index
    timestamps2 = df['est_arrival_time']

    idx = get_bus_arrival_index(df)
    timestamps3 = timestamps1[idx]
    timestamps4 = timestamps2[idx]

    plot_title = 'Current time x Estimated Arrival'

    fig, ax = plt.subplots()
    ax.plot(timestamps1, timestamps2)
    plt.plot(timestamps3, timestamps4, 'ro', label='bus change')

    ax.set_xlabel("Current Time")
    ax.set_ylabel("Estimated Arrival Time")
    ax.xaxis_date()
    ax.yaxis_date()
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    for label in ax.get_yticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.title.set_text(plot_title)
    plt.show()


def plot_consequtive_day_data(df):
    timestamps1 = df.index
    df = df.fillna(method='ffill')
    columns = df.columns

    plot_title = 'Delay from Previous Estimation on Expected Arrival'

    fig, ax = plt.subplots(figsize=(17, 8))
    [ax.plot(timestamps1, df[x].values, label=x) for x in columns]

    ax.set_xlabel("Current Time")
    ax.set_ylabel("Difference from previous Estimated Arrival (sec)")
    ax.xaxis_date()
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.title.set_text(plot_title)
    plt.show()


def plot_series_and_moving_average(xvals, yvals, moving_vals, plot_title='', xlabel='', ylabel=''):
    start_idx = (len(yvals) - len(moving_vals))

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.plot(xvals, yvals, label='measured')
    ax.plot(xvals[start_idx:], moving_vals, label='moving average')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis_date()
    ax.legend(loc="best")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.title.set_text(plot_title)
    plt.grid(True)
    plt.show()


def plot_exponential_smoothing(xvals, yvals, alphas):
    fig = plt.figure(figsize=(17, 8))
    ax = plt.gca()
    plt.plot(xvals.values, yvals.values, label="Actual")
    for alpha in alphas:
        plt.plot(xvals.values, exponential_smoothing(yvals, alpha), label="Alpha {}".format(alpha))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.set_xlabel("Time")
    ax.set_ylabel("Diff from Previous Estimated Arrival (in seconds)")

    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);


def plot_double_exponential_smoothing(xvals, yvals, alphas, betas):
    plt.figure(figsize=(17, 8))
    ax = plt.gca()
    plt.plot(xvals.values, yvals.values, label="Actual")
    for alpha in alphas:
        for beta in betas:
            plt.plot(xvals.values, double_exponential_smoothing(yvals, alpha, beta)[:-1],
                     label="Alpha {}, beta {}".format(alpha, beta))

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

    ax.set_xlabel("Time")
    ax.set_ylabel("Diff from Previous Estimated Arrival (in seconds)")

    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)


def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    import statsmodels.tsa.api as smt

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, method='ywm')
        plt.tight_layout()


def plot_seasonal_decomposition(yvals, period):
    import statsmodels.api as sm

    seas_d = sm.tsa.seasonal_decompose(yvals, model='add', period=period)
    fig = seas_d.plot()
    fig.set_figheight(4)
    plt.show()


def plot_predict_model(yvals, model, pred_length):
    from statsmodels.graphics.tsaplots import plot_predict
    # plot_predict(model, dynamic=False)
    fig, ax = plt.subplots()
    ax = yvals.plot(ax=ax)
    plot_predict(model, len(yvals), len(yvals) + pred_length, ax=ax,
                 dynamic=False)
    plt.show()


def plot_keras_models_performance(val_performance, test_performance, metric_index, metric_name, target_column=''):
    print('\n\nPerformance evaluation\n')
    x = np.arange(len(test_performance))
    width = 0.3

    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in test_performance.values()]

    plt.ylabel(f'{metric_name} [{target_column}, normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=test_performance.keys(),
               rotation=45)
    _ = plt.legend()
    plt.title(f'Performance results ({metric_name})')
    plt.show()

    print(f'Test performance ({metric_name})')
    for name, value in test_performance.items():
        print(f'{name:12s}: {value[metric_index]:0.4f}')

def plot_mse_per_latent_dimension(latent_dims, val_scores, test_scores):
    fig = plt.figure(figsize=(6,4))
    plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16, label='Valid')
    plt.plot(latent_dims, test_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="b", markersize=16, label='Test')

    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.minorticks_off()
    #plt.ylim(0,100)
    plt.show()


def show_anomaly_detection_on_image(image, pred, anomaly_label):
    label = "anomaly" if pred == anomaly_label else "normal"
    color = (0, 0, 255) if pred == anomaly_label else (0, 255, 0)

    I1 = ImageDraw.Draw(image)
    fnt = ImageFont.truetype("arial.ttf", size=40)
    I1.text((10, 25), label, font=fnt, fill=color)
    plt.imshow(image)
    plt.show()


