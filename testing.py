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


import pygmt
region = [77, 92, 26, 38]
grid = pygmt.datasets.load_earth_relief(resolution="05m", region=region)
fig = pygmt.Figure()
fig.basemap(region=region, projection="M6i", frame=True)
fig.grdimage(grid=grid, cmap="geo", shading=True)
fig.grdcontour(grid=grid, interval=500, annotation="1000+f8p,Helvetica-Bold,black", pen="1p,black")
fig.colorbar(frame=["x+lElevation (m)", "y+lm"])
fig.text(x=86.9250, y=27.9881, text="Mount Everest", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=81.5167, y=30.3756, text="Nanda Devi", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=88.1475, y=27.7025, text="Kangchenjunga", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=86.8333, y=28.0016, text="Lhotse", font="12p,Helvetica-Bold,white", justify="CM")
fig.text(x=88.1464, y=27.7668, text="Makalu", font="12p,Helvetica-Bold,white", justify="CM")
