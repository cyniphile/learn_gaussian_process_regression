import dash
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
from dash import html, dcc
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
import scipy as sp


def pairwise_rbf(xa, xb, sigma=5):
    sq_norm = (-0.5/sigma**2) * sp.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)
    
layout = go.Layout(title='Samples from a 100-D Multivariate Gaussian with new covariance')
fig_double = go.FigureWidget(make_subplots(rows=1, cols=2, subplot_titles=('Function Samples', 'Covariance Matrix')))
xa = np.linspace(-1, 1, 100).reshape(1, -1).T
cov = pd.DataFrame(pairwise_rbf(xa, xa))
cov_map = go.Heatmap(z=np.rot90(cov), showscale=False)
fig_double.add_trace(cov_map, row=1, col=2)

# Define app
app = JupyterDash(__name__)
scatter = go.Scatter(mode='markers')
# scatter2 = go.Scatter(x =xa.T[0], y=xa.T[0], mode='markers')
layout = go.Layout(title='Samples from a Multivariate Gaussian at Real-Valued Indices')
fig_double.add_trace(scatter, row=1, col=1)
# fig_double.add_trace(scatter, row=1, col=1)


# Define app layout
app.layout = html.Div([
    html.Button("New Sample", id="new-sample-btn", n_clicks=0, className="btn btn-success"),
    html.Button("Clear", id="clear-btn", n_clicks=0, className="btn btn-danger"),
    dcc.Slider(
        id='slider-width', min=.1, max=5, 
        value=0.5, step=0.1),
    dcc.Graph(id="plot", figure=fig_double),
])

# Define app callback
@app.callback(
    Output("plot", "figure"),
    Input("new-sample-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    Input("slider-width", "value"), 
    State("plot", "figure")
)
def update_figure(new_clicks, clear_clicks, sigma, fig):
    global cov
    if "new-sample-btn" in dash.callback_context.triggered[0]["prop_id"]:
        print(sigma)
        y = np.random.multivariate_normal(mean=np.zeros(100), cov=cov, size=100)
        fig['data'] += [{'type': 'scatter', 'x': xa.T[0], 'y': y.T[0], 'mode': 'lines', 'xaxis': 'x', 'yaxis': 'y', 'name': f"l = {sigma}"}]
    elif "clear-btn" in dash.callback_context.triggered[0]["prop_id"]: 
        fig['data'] = fig['data'][:1]
        fig['data'] += [{'type': 'scatter', 'mode': 'lines', 'xaxis': 'x', 'yaxis': 'y'}]
    elif "slider-width" in dash.callback_context.triggered[0]["prop_id"]:
        cov = pairwise_rbf(xa, xa, sigma=sigma)
        fig['data'][0] = {'z': np.rot90(cov), 'type': 'heatmap', 'xaxis': 'x2', 'yaxis': 'y2', 'showscale': False}
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(port='8061', debug=True)