import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qtcore import *
import cv2

class MainWindow(Widget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.VBL = QVBoxLayout()
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        self.CancelBTN = QPushButton("Cancel")
        self.VBL.addWidget(self.CancelBTN)
        self.setLayout(self.VBL)

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
