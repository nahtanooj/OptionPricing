import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Graphics:
    def __init__(self):
        """
        Initialize the Graphics class.
        """
        pass

    def add_trace(self, fig, x, y, name, color, dash=None):
        """
        Add a trace to a Plotly figure.

        Arguments:
            fig: The Plotly figure object.
            x: X-axis values.
            y: Y-axis values.
            name: Name of the trace (for the legend).
            color: Color of the trace.
            dash: Line style (e.g., 'dash' for dashed lines).
        """
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            line=dict(color=color, width=2, dash=dash)
        ))

    def set_layout(self, fig, title, xaxis_title, yaxis_title):
        """
        Set the layout of a Plotly figure.

        Arguments:
            fig: The Plotly figure object.
            title: Title of the plot.
            xaxis_title: Title of the x-axis.
            yaxis_title: Title of the y-axis.
        """
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            template="plotly_white",
            showlegend=True
        )

    def print_convergence_graph(self, mu_k, lower_bound, upper_bound):
        """
        Plot the sliding mean, lower bound, and upper bound on the same graph.

        Arguments:
            mu_k: Array of sliding means.
            lower_bound: Array of lower bounds of the confidence interval.
            upper_bound: Array of upper bounds of the confidence interval.
        """
        # Create a Plotly figure
        fig = go.Figure()

        # Add traces for each array
        self.add_trace(fig, np.arange(1, len(mu_k) + 1), mu_k, 'Sliding Mean (μ_k)', 'blue')
        self.add_trace(fig, np.arange(1, len(lower_bound) + 1), lower_bound, 'Lower Bound', 'red', dash='dash')
        self.add_trace(fig, np.arange(1, len(upper_bound) + 1), upper_bound, 'Upper Bound', 'green', dash='dash')

        # Set the layout
        self.set_layout(fig, "Monte Carlo Simulation: Sliding Mean and Confidence Interval", "Number of Simulations", "Value")

        # Show the plot
        fig.show()

    def plot_stock_chart(self, df, ticker):
        """
        Plot the historical stock prices.

        Arguments:
            historical_prices_df: Pandas DataFrame containing historical prices.
            ticker: Ticker symbol of the asset.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df, label='Price', color='blue')
        plt.title(f'{ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()