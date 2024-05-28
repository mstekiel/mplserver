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
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets # type: ignore # in qt_compat a proper version is imported based on installed Qt and can't be rocognized properly
import matplotlib.pyplot as plt

# typechecking
from matplotlib.figure import Figure
from types import ModuleType


class ApplicationWindow(QtWidgets.QMainWindow):
    # mpl figure
    fig: Figure
    canvas: FigureCanvas
    navigationtoolbar: NavigationToolbar

    # watched file
    path: str
    watched_module: ModuleType

    # Qt types
    # ignored as the mpl backed misbehaves with providing types
    # _main = QtWidgets.QWidget()
    # layout: QtWidgets.QVBoxLayout

    def __init__(self, watched_filename):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)

        path, filename = os.path.split(watched_filename)
        script_name, _ = os.path.splitext(filename)

        sys.path.append(path)
        self.watched_module = importlib.import_module(script_name)

        self.setup_widgets()

    def setup_widgets(self):
        self.fig = self.import_plot()
        
        # Standard resolution/dpi for mpl is 100
        self.resize([100*x for x in self.fig.get_size_inches()])

        self.canvas = FigureCanvas(self.fig)
        self.navigationtoolbar = NavigationToolbar(self.canvas, self)

        self.layout.addWidget(self.navigationtoolbar)
        self.layout.addWidget(self.canvas)


    def clear_widgets(self):
        plt.close(self.fig) # to avoid overflow of figures

        # Clear all, as there aro some hidden widgets on top op canvas and navbar
        # https://stackoverflow.com/a/10067548
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
            submodule = getattr(module, name)
            if isinstance(submodule, type(importlib)):
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

    watched_filename = f'{os.getcwd()}\{sys.argv[1]}'

    logger.info(f'Starting the file watcher {watched_filename}')
    script_watcher = QtCore.QFileSystemWatcher([watched_filename])
    app = ApplicationWindow(watched_filename)
    script_watcher.fileChanged.connect(app.file_changed)


    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()