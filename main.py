from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import numpy as np
import matplotlib.pyplot as plt
import os

app = FastAPI()

# Directory to save images
IMAGE_DIR = "generated_plots"
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_plot(z1, z2, w1, w2, c, filename="plot.png"):
    # Define the function
    def f(x, y):
        term1 = np.sqrt(np.abs((x - z1)**2 - (y - z2)**2))
        term2 = np.sqrt(np.abs((x - w1)**2 - (y - w2)**2))
        return term1 + term2 - c
    
    # Define x and y grid
    x_vals = np.linspace(-0.7, 0.7, 400)
    y_vals = np.linspace(-0.7, 0.7, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    # Plot the contour
    plt.figure(figsize=(6, 6))
    plt.contour(X, Y, Z, levels=[0], colors='red')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.grid(False)
    
    # Save the plot
    filepath = os.path.join(IMAGE_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    return filepath

@app.get("/generate_plot")
def get_plot(
    z1: float = Query(0.2, description="Z1 value"),
    z2: float = Query(0.2, description="Z2 value"),
    w1: float = Query(0.2, description="W1 value"),
    w2: float = Query(0.2, description="W2 value"),
    c: float = Query(0.31, description="C value")
):
    filename = "plot.png"
    filepath = generate_plot(z1, z2, w1, w2, c, filename)
    return FileResponse(filepath, media_type="image/png")
