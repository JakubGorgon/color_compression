import plotly.graph_objects as go
import pandas as pd

camera_settings = dict(
    up=dict(x=0, y=0, z=1),          # Define 'up' direction
    center=dict(x=0, y=0, z=-0.25),      # Focal point of the camera
    eye=dict(x=1.25, y=1.25, z=1.25)    # Move camera closer for zoom, adjust for full view
)


def plot_3d_scatter(df):
    # Create a 3D scatter plot for the given dataframe
    fig = go.Figure()

    # Add scatter plot for the RGB data
    fig.add_trace(go.Scatter3d(
        x=df['r'],
        y=df['g'],
        z=df['b'],
        mode='markers',
        marker=dict(
            size=3,
            color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in zip(df['r'], df['g'], df['b'])],
            opacity=0.8
        ),
        text=['R: {}<br>G: {}<br>B: {}'.format(r, g, b) for r, g, b in zip(df['r'], df['g'], df['b'])],
        hoverinfo='text'  
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            camera = camera_settings
        ),
        width = 400,
        height = 400,
        margin=dict(l=0, r=0, t=0, b=0)  # Reduce margins around the plot

        
    )

    return fig
    

def plot_3d_scatter_compressed(img_before, img_after):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=img_before['r'],
        y=img_before['g'],
        z=img_before['b'],
        mode='markers',
        marker=dict(
            size=3,
            color=['rgb({}, {}, {})'.format(r, g, b) for r, g, b in zip(img_after['r'], img_after['g'], img_after['b'])],
            opacity=0.8
        )
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            camera = camera_settings
        ),
        width = 400,
        height = 400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

    