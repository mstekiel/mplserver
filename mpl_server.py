import sys
import os
import importlib
import traceback
import logging
import logging.config

logger = logging.getLogger('dj')

def setup_logging():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                # 'format': '%(levelname)s: %(message)s'
                'format': '[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s.%(msecs)03d: %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S%z'
            }
        },
        'handlers': {
            'stdout': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'root': {
                'level': 'INFO',
                'handlers': ['stdout']
            },
            'matplotlib': {
                'level': 'INFO'
            }
        }
    }

    logging.config.dictConfig(logging_config)

from PyQt5 import QtCore

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar # type: ignore
from matplotlib.backends.qt_compat import QtWidgets # type: ignore
import matplotlib.pyplot as plt


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, watched_filename):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)

        path, filename = os.path.split(watched_filename)
        script_name, _ = os.path.splitext(filename)
        self.watched_module = importlib.import_module(script_name)
        
        self.setup_widgets()

    def setup_widgets(self):
        self.fig = self.import_plot()
        self.canvas = FigureCanvas(self.fig)
        self.navigationtoolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.navigationtoolbar)
        self.layout.addWidget(self.canvas)


    def clear_widgets(self):
        plt.close(self.fig) # to avoid overflow of figures

        # Clear all, as there aro some hidden widgets on top op canvas and navbar
        while self.layout.count():  
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def file_changed(self, path):
        logger.info(f'Script changed, replotting: {path}')
        # Will maybe have to wait 15 ms, if another signal was not sent replot -> no, works fine
        # TODO Adapt figsize changes to window size changes 
        self.clear_widgets()
        self.setup_widgets() 

    def import_plot(self):
        try:
            self.reload_module(self.watched_module)
            plot = getattr(self.watched_module, 'plot')

            fig = plot()
            return fig
        except Exception as e:
            logger.error(traceback.format_exc())
            return self.fig

    def reload_module(self, module):
        # ignore: numpy?
        # crashes: pathlib
        importlib.reload(module)

        non_reloadable_modules = ['importlib', 'numpy']
        for name in dir(module):
            if isinstance(submodule := getattr(module, name), type(importlib)):
                if not submodule.__name__ in non_reloadable_modules:
                    logger.debug(f'Reloading submodule: {submodule}')
                    importlib.reload(submodule)


if __name__ == "__main__":
    setup_logging()

    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)


    # setup_filewatcher()
    watched_filename = f'{os.getcwd()}\{sys.argv[1]}'
    # watched_filename = r'C:\Users\Stekiel\Documents\GitHub\mplplotter\plot_something.py'

    logger.info(f'Starting the file watcher {watched_filename}')
    script_watcher = QtCore.QFileSystemWatcher([watched_filename])
    app = ApplicationWindow(watched_filename)
    script_watcher.fileChanged.connect(app.file_changed)


    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()