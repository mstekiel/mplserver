# Live inspector of plots made by matplotlib.

Saving the script file will triger replotting and updating the plot in the window of the application.

The absolute requirement in the script file is the `plot` method with signature:
`plot() -> mpl.Figure`

### Usage

`python mpl_wrapper.py script_file.py`

## Examples

`python mpl_wrapper.py magnetic_textures.py`

![](.\skyrmions.png)