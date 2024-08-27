# # import pygmt
# # import io
# # import base64
# # import tempfile

# # def plot(code_string):
# #     # Define a dictionary to store the generated variables during exec()
# #     local_vars = {}
    
# #     # Execute the passed code string in a controlled local context
# #     exec(code_string, {"pygmt": pygmt}, local_vars)
    
# #     # Extract the figure object from local_vars
# #     fig = local_vars.get("fig")

# #     if fig is None:
# #         raise ValueError("The code string must define a 'fig' object.")
    
# #     # Save the figure to a temporary file
# #     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
# #         fig.savefig(tmpfile.name)
        
# #         # Read the file content into a buffer
# #         with open(tmpfile.name, "rb") as f:
# #             img_base64 = base64.b64encode(f.read()).decode("utf-8")

# #     return img_base64

# # # The code string to be executed
# # code_string = """
# # region = [-82, -34, -56, 14]
# # grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
# # fig = pygmt.Figure()
# # fig.basemap(region=region, projection="M6i", frame=True)
# # fig.grdimage(grid=grid, cmap="geo", shading=True)
# # fig.grdcontour(grid=grid, levels=500, annotation="500+f8p")
# # fig.coast(shorelines=True, water="skyblue")
# # """

# # # Generate the base64-encoded image using the code string
# # image_base64 = plot(code_string)
# # print(image_base64)




# import pygmt
# # Load the global relief data at a specified resolution (e.g., 05m, 01m, 30s, 15s)
# # grid = pygmt.datasets.load_earth_relief(resolution="05m", region=[-125, -65, 24, 50])  # Example region: USA
# # fig = pygmt.Figure()

# # # Set the region and projection of the map
# # fig.basemap(region=[-125, -65, 24, 50], projection="M6i", frame=True)

# # # Plot the topographic data as a color grid
# # fig.grdimage(grid=grid, cmap="geo", shading=True)

# # # Add contour lines
# # fig.grdcontour(grid=grid, interval=500, annotation="1000+f8p,Helvetica-Bold,black", pen="1p,black")

# # # Add a color bar to represent elevation
# # fig.colorbar(frame=["x+lElevation (m)", "y+lm"])



# grid = pygmt.datasets.load_earth_relief(resolution="30s", region=[-125, -66.5, 24.396, 49.384])
# fig = pygmt.Figure()
# fig.grdimage(grid=grid, cmap="geo", shading=True)
# fig.grdcontour(grid=grid, levels=500, annotation="1000+f6p")
# fig.colorbar(frame='af+l"Elevation (m)"')
# fig.basemap(frame=True)
# fig.text(text="Topographic Map of the Contiguous United States", position="JTC", font="18p,Helvetica-Bold", offset="0/1.0c")

# # Display the map
# fig.show()


# import pygmt
# region = [73, 135, 18, 54]
# grid = pygmt.datasets.load_earth_relief(resolution="05m", region=region)
# fig = pygmt.Figure()
# fig.basemap(region=region, projection="M6i", frame=True)
# fig.grdimage(grid=grid, cmap="geo", shading=True)
# fig.grdcontour(grid=grid, interval=500, annotation="1000+f8p,Helvetica-Bold,black", pen="1p,black")
# fig.colorbar(frame=["x+lElevation (m)", "y+lm"])
# fig.text(x=105, y=35, text="China", font="22p,Helvetica-Bold,black", justify="CM")


# import pygmt
# region = [77, 92, 26, 38]
# grid = pygmt.datasets.load_earth_relief(resolution="05m", region=region)
# fig = pygmt.Figure()
# fig.basemap(region=region, projection="M6i", frame=True)
# fig.grdimage(grid=grid, cmap="geo", shading=True)
# fig.grdcontour(grid=grid, interval=500, annotation="1000+f8p,Helvetica-Bold,black", pen="1p,black")
# fig.colorbar(frame=["x+lElevation (m)", "y+lm"])
# fig.text(x=86.9250, y=27.9881, text="Mount Everest", font="12p,Helvetica-Bold,white", justify="CM")
# fig.text(x=81.5167, y=30.3756, text="Nanda Devi", font="12p,Helvetica-Bold,white", justify="CM")
# fig.text(x=88.1475, y=27.7025, text="Kangchenjunga", font="12p,Helvetica-Bold,white", justify="CM")
# fig.text(x=86.8333, y=28.0016, text="Lhotse", font="12p,Helvetica-Bold,white", justify="CM")
# fig.text(x=88.1464, y=27.7668, text="Makalu", font="12p,Helvetica-Bold,white", justify="CM")

import os
import io
from io import BytesIO
import openai
import base64
import requests
from PIL import Image
import tkinter as tk
from PIL import ImageTk
from flask import Flask, request, jsonify, render_template, Response
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
import pygmt
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

def plot2(code_string):
    # Define a dictionary to store the generated variables during exec()
    local_vars = {}
    
    try:
        # Execute the passed code string in a controlled local context
        exec(code_string, {"pygmt": pygmt}, local_vars)
        
        # Extract the figure object from local_vars
        fig = local_vars.get("fig")

        # if fig is None:
        #     raise ValueError("The code string must define a 'fig' object.")

        # Save the figure to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.savefig(tmpfile.name)
            
            # Read the file content into a buffer
            with open(tmpfile.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        # Handle exceptions and return an error message
        print(f"An error occurred: {e}")
        img_base64 = None
    
    return img_base64


code_string = """
import pygmt
region = [77, 92, 26, 38]
grid = pygmt.datasets.load_earth_relief(resolution="05m", region=region)
fig = pygmt.Figure()
fig.basemap(region=region, projection="M6i", frame=True)
fig.grdimage(grid=grid, cmap="geo", shading=True)
fig.grdcontour(grid=grid, levels=500, annotation="1000+f8p,Helvetica-Bold,black", pen="1p,black")
fig.colorbar(frame=["x+lElevation (m)", "y+lm"])
fig.text(x=86.9250, y=27.9881, text="Mount Everest", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=81.5167, y=30.3756, text="Nanda Devi", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=88.1475, y=27.7025, text="Kangchenjunga", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=86.8333, y=28.0016, text="Lhotse", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=88.1464, y=27.7668, text="Makalu", font="12p,Helvetica-Bold,white", justify="CM")
"""

# print(plot2(code_string))


import numpy as np
import matplotlib.pyplot as plt
# Constants
mu_0 = 4 * np.pi * 10**-7  # T·m/A, permeability of free space
N = 800  # number of turns
L = 0.4  # length of solenoid in meters
r = 0.05  # radius of solenoid in meters
n = N / L  # turns per unit length
# Time array for one complete cycle
t = np.linspace(0, 0.02, 1000)  # 1000 points between 0 and 0.02 seconds (one cycle)
# Current as a function of time
I_t = 5 * np.cos(100 * np.pi * t)
# Magnetic field inside the solenoid
B_t = mu_0 * n * I_t
# Induced electric field (E = -dB/dt)
dB_dt = np.gradient(B_t, t)
E_t = -r * dB_dt  # E = -r * (dB/dt) for a solenoid with circular cross-section
# Plotting the magnetic field and induced electric field
plt.figure(figsize=(14, 6))
# Plot the magnetic field
plt.subplot(1, 2, 1)
plt.plot(t, B_t, label=r'$B(t)$')
plt.title('Magnetic Field Inside the Solenoid')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field (T)')
plt.grid(True)
plt.legend()
# Plot the induced electric field
plt.subplot(1, 2, 2)
plt.plot(t, E_t, label=r'$E(t)$', color='r')
plt.title('Induced Electric Field')
plt.xlabel('Time (s)')
plt.ylabel('Electric Field (V/m)')
plt.grid(True)
plt.legend()
plt.tight_layout()



plt.show()