import numpy as np

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import Range1d


def inspect_prediction(X, target, prediction, offset=0, window=500):
    m = target.mean()
    s = target.std()

    X = X[offset:offset + window]
    target = target[offset:offset + window]
    prediction = prediction[offset:offset + window]

    # Define Bokeh figure
    y_max = (max(X.max(), target.max(), prediction.max()) - m) / s * 2
    y_min = (min(X.min(), target.min(), prediction.min()) - m) / s * 2
    x_range = Range1d(offset, offset + 96, bounds=(offset, offset + window))
    y_range = Range1d(y_min, y_max, bounds=(y_min, y_max))

    p = figure(plot_width=900, plot_height=500,
               x_range=x_range, y_range=y_range)

    # Add line plots
    p.line(np.arange(offset, offset + window), (target - m) / s,
           color='orangered', width=2, legend_label='Target')
    p.line(np.arange(offset, offset + window), (prediction - m) / s,
           color='royalblue', width=2, legend_label='Prediction')
    for i in range(len(X[0])):
        p.line(np.arange(offset, offset + window), X[:, i], color='steelblue',
               alpha=0.03, width=2, legend_label='Reservoir state')

    show(p)
