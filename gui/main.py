import sys
from PyQt6.QtWidgets import QApplication
from gui import SatelliteSimulatorGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SatelliteSimulatorGUI()
    ex.show()
    sys.exit(app.exec())

