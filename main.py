from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
from skimage.morphology import *
from skimage.util import *
import sys
import cv2 as cv
import numpy as np
import imutils
from pathlib import Path

class mainwindow(QMainWindow):
    def __init__(self):
        super(mainwindow, self).__init__()
        self.windowWIDTH = 1950
        self.windowHEIGHT = 1050
        self.setGeometry(0, 0, self.windowWIDTH, self.windowHEIGHT)
        self.setWindowTitle("Simple Object Detector")
        self.setWindowIcon(QIcon("icon.png"))
        self.imageloaded = False
        self.initUI()
        self.mainclass = processingClass()

    def initUI(self):
        self.center()
        self.Mainmenubar()
        self.allLabel()
        self.allButton()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.background = QWidget(self)

    def Mainmenubar(self):
        menuBar = self.menuBar()

        filemenu = menuBar.addMenu('&File')

        file_open = QAction("Open", self)
        file_open.triggered.connect(self.browsefiles)
        file_open.setShortcut("CTRL+O")

        file_exit = QAction("Exit", self)
        file_exit.triggered.connect(self.exitApp)

        filemenu.addAction(file_open)
        filemenu.addAction(file_exit)

    def allButton(self):
        openImage = QPushButton(self.background)
        openImage.setText("Open Image...")
        openImage.move(1300, 100)
        openImage.clicked.connect(self.browsefiles)

        calculateButton = QPushButton(self.background)
        calculateButton.setText("Calculate")
        calculateButton.move(1300, 150)
        calculateButton.clicked.connect(self.calculateResult)

        inputlabel = QLabel(self)
        inputlabel.setText("Reference size")
        inputlabel.adjustSize()
        inputlabel.move(1310, 220)

        self.inputsize = QLineEdit(self)
        self.inputsize.move(1410, 215)
        self.inputsize.resize(42, 25)
        reg_ex = QRegExp("[0-9]+")
        number_validator = QRegExpValidator(reg_ex, self.inputsize)
        self.inputsize.setValidator(number_validator)
        self.inputsize.setPlaceholderText("In cm")

        contourlabel = QLabel(self)
        contourlabel.setText("Contour area")
        contourlabel.adjustSize()
        contourlabel.move(1320, 255)

        self.contourinput = QLineEdit(self)
        self.contourinput.move(1410, 250)
        self.contourinput.resize(55, 25)
        number_validator1 = QRegExpValidator(reg_ex, self.contourinput)
        self.contourinput.setValidator(number_validator1)
        self.contourinput.setPlaceholderText("In pixel ")

        blurlabel = QLabel(self)
        blurlabel.setText("Filter size")
        blurlabel.adjustSize()
        blurlabel.move(1340, 290)

        self.blurbox = QSpinBox(self)
        self.blurbox.setRange(1, 29)
        self.blurbox.setValue(7)
        self.blurbox.setSingleStep(2)
        self.blurbox.move(1410, 285)
        self.blurbox.adjustSize()
        self.blurbox.lineEdit().setReadOnly(True)

        self.resetbutton = QPushButton(self.background)
        self.resetbutton.setText("Reset all")
        self.resetbutton.move(1500, 225)
        self.resetbutton.setFixedSize(60, 24)
        self.resetbutton.clicked.connect(self.resetFunction)

    def allLabel(self):
        self.nextButton = [QPushButton(self.background), QPushButton(self.background), QPushButton(self.background), QPushButton(self.background)]
        for i, btn in enumerate(self.nextButton):
            btn.setText(">")
            btn.hide()
            btn.clicked.connect(lambda _, i = i: self.nextImage(i, 1))
        self.backButton = [QPushButton(self.background), QPushButton(self.background), QPushButton(self.background), QPushButton(self.background)]
        for i, btn in enumerate(self.backButton):
            btn.setText("<")
            btn.hide()
            btn.clicked.connect(lambda _, i = i: self.nextImage(i, -1))
        v_lay = QVBoxLayout()
        h_lay = QHBoxLayout()
        label_layer0 = QVBoxLayout()
        label_layer1 = QVBoxLayout()
        label_layer2 = QVBoxLayout()
        label_layer3 = QVBoxLayout()
        button_layer0 = QHBoxLayout()
        button_layer1 = QHBoxLayout()
        button_layer2 = QHBoxLayout()
        button_layer3 = QHBoxLayout()
        self.labelImage = QLabel(self)
        self.labelImage.setAlignment(Qt.AlignHCenter)
        self.labelImage1 = QLabel(self)
        self.labelImage2 = QLabel(self)
        self.labelImage3 = QLabel(self)

        self.originalLabel = QLabel(self)
        self.originalLabel.setText("ORIGINAL IMAGE")
        self.originalLabel.setFont(QFont('Comic Sans MS', 10))
        self.originalLabel.adjustSize()
        self.originalLabel.move(895, 432)
        self.originalLabel.hide()

        self.cannyLabel = QLabel(self)
        self.cannyLabel.setText("Result of Canny")
        self.cannyLabel.setFont(QFont('Comic Sans MS', 12))
        self.cannyLabel.adjustSize()
        self.cannyLabel.move(215, 889)
        self.cannyLabel.hide()

        self.prewittLabel = QLabel(self)
        self.prewittLabel.setText("Result of Prewitt")
        self.prewittLabel.setFont(QFont('Comic Sans MS', 12))
        self.prewittLabel.adjustSize()
        self.prewittLabel.move(895, 889)
        self.prewittLabel.hide()

        self.sobelLabel = QLabel(self)
        self.sobelLabel.setText("Result of Sobel")
        self.sobelLabel.setFont(QFont('Comic Sans MS', 12))
        self.sobelLabel.adjustSize()
        self.sobelLabel.move(1575, 889)
        self.sobelLabel.hide()

        label_layer0.addWidget(self.labelImage)
        button_layer0.addStretch(1)
        button_layer0.addWidget(self.backButton[3])
        button_layer0.addWidget(self.nextButton[3])
        button_layer0.addStretch(1)
        label_layer0.addLayout(button_layer0)

        label_layer1.addWidget(self.labelImage1)
        button_layer1.addWidget(self.backButton[0])
        button_layer1.addWidget(self.nextButton[0])
        label_layer1.addLayout(button_layer1)

        label_layer2.addWidget(self.labelImage2)
        button_layer2.addWidget(self.backButton[1])
        button_layer2.addWidget(self.nextButton[1])
        label_layer2.addLayout(button_layer2)

        label_layer3.addWidget(self.labelImage3)
        button_layer3.addWidget(self.backButton[2])
        button_layer3.addWidget(self.nextButton[2])
        label_layer3.addLayout(button_layer3)

        h_lay.addLayout(label_layer1)
        h_lay.addStretch(1)
        h_lay.addLayout(label_layer2)
        h_lay.addStretch(1)
        h_lay.addLayout(label_layer3)
        h_lay.setAlignment(Qt.AlignHCenter)
        v_lay.addLayout(label_layer0)
        v_lay.addStretch(1)
        v_lay.addLayout(h_lay)
        v_lay.addStretch(3)
        v_lay.setAlignment(Qt.AlignTop)
        self.background.setLayout(v_lay)
        self.setCentralWidget(self.background)

    def nextImage(self, edgeNo, imageNo):
        if edgeNo == 0:
            self.mainclass.cannyindex += imageNo
            if self.mainclass.cannyindex > 2:
                self.mainclass.cannyindex = 0
            elif self.mainclass.cannyindex < 0:
                self.mainclass.cannyindex = 2
            edgeType = self.mainclass.cannyindex
            result = self.mainclass.cannyresult
            label = self.labelImage1
        elif edgeNo == 1:
            self.mainclass.prewittindex += imageNo
            if self.mainclass.prewittindex > 2:
                self.mainclass.prewittindex = 0
            elif self.mainclass.prewittindex < 0:
                self.mainclass.prewittindex = 2
            edgeType = self.mainclass.prewittindex
            result = self.mainclass.prewittresult
            label = self.labelImage2
        elif edgeNo == 2:
            self.mainclass.sobelindex += imageNo
            if self.mainclass.sobelindex > 2:
                self.mainclass.sobelindex = 0
            elif self.mainclass.sobelindex < 0:
                self.mainclass.sobelindex = 2
            edgeType = self.mainclass.sobelindex
            result = self.mainclass.sobelresult
            label = self.labelImage3
        else:
            self.mainclass.originalindex += imageNo
            if self.mainclass.originalindex > 3:
                self.mainclass.originalindex = 0
            elif self.mainclass.originalindex < 0:
                self.mainclass.originalindex = 3
            edgeType = self.mainclass.originalindex
            result = self.mainclass.originalresult
            label = self.labelImage
        label.setPixmap(QPixmap.fromImage(result[edgeType]))
        label.adjustSize()

    def loadImage(self):
        image = cv.imread(self.browse)
        self.image = cv.resize(image, (550, 350))
        self.backupimage = self.image.copy()
        image_RGBA = cv.cvtColor(self.image, cv.COLOR_BGR2BGRA)
        qtImg = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.labelImage.setPixmap(QPixmap.fromImage(qtImg))
        self.labelImage.adjustSize()
        self.mainclass.cannyindex = 2
        self.mainclass.prewittindex = 2
        self.mainclass.sobelindex = 2
        self.mainclass.originalindex = 3

    def browsefiles(self):
        self.browse, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "All Images (*.png && *.jpg && *.bmp && *.tiff);;PNG (*.png);;JPG(*.jpg);;BMP(*.bmp);;TIFF(*.tiff)")
        if self.browse == (""):
            return
        else:
            self.imageName = Path(self.browse).resolve().stem
            self.loadImage()
            self.imageloaded = True

    def exitApp(self):
        qApp.quit()

    def calculateResult(self):
        dist_in_cm = self.inputsize.text()
        if dist_in_cm == "" or dist_in_cm == "0":
            dist_in_cm = 4
        contoursize = self.contourinput.text()
        if contoursize == "" or contoursize == "0":
            contoursize = 100
        blursize = self.blurbox.value()
        if self.imageloaded:
            self.originalLabel.show()
            self.cannyLabel.show()
            self.prewittLabel.show()
            self.sobelLabel.show()
            self.mainclass.processImages(self.image, dist_in_cm, contoursize, blursize)
            self.labelImage1.setPixmap(QPixmap.fromImage(self.mainclass.cannyresult[self.mainclass.cannyindex]))
            self.labelImage1.adjustSize()
            self.labelImage2.setPixmap(QPixmap.fromImage(self.mainclass.prewittresult[self.mainclass.prewittindex]))
            self.labelImage2.adjustSize()
            self.labelImage3.setPixmap(QPixmap.fromImage(self.mainclass.sobelresult[self.mainclass.sobelindex]))
            self.labelImage3.adjustSize()
            self.labelImage.setPixmap((QPixmap.fromImage(self.mainclass.originalresult[self.mainclass.originalindex])))
            self.labelImage.adjustSize()
            self.backButton[3].setFixedSize(int(self.labelImage.width()/2 - 5), 30)
            self.nextButton[3].setFixedSize(int(self.labelImage.width()/2 - 5), 30)
            for btn in self.backButton:
                btn.show()
            for btn in self.nextButton:
                btn.show()

    def resetFunction(self):
        self.inputsize.clear()
        self.contourinput.clear()
        self.blurbox.setValue(7)

class processingClass:
    def __init__(self):
        self.cannyindex = 2
        self.prewittindex = 2
        self.sobelindex = 2
        self.originalindex = 3
        self.originalresult = [None, None, None, None]
        self.cannyresult = [None, None, None]
        self.prewittresult = [None, None, None]
        self.sobelresult = [None, None, None]
        self.kernel = np.ones((3, 3))

    def processImages(self, img, dist_in_cm, contoursize, blursize):
        self.image = img
        image_RGBA = cv.cvtColor(self.image.copy(), cv.COLOR_BGR2BGRA)
        self.originalresult[2] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        grayImg = cv.cvtColor(self.image.copy(), cv.COLOR_BGR2GRAY)
        image_RGBA = cv.cvtColor(grayImg.copy(), cv.COLOR_BGR2BGRA)
        self.originalresult[0] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.grayblurred = cv.GaussianBlur(grayImg.copy(), (blursize, blursize), 0)
        image_RGBA = cv.cvtColor(self.grayblurred.copy(), cv.COLOR_BGR2BGRA)
        self.originalresult[1] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.canny(dist_in_cm, contoursize)
        self.prewitt(dist_in_cm, contoursize)
        self.sobel(dist_in_cm, contoursize)
        self.hariscorner(grayImg)

    def canny(self, dist_in_cm, contoursize):
        cannyImg = cv.Canny(self.grayblurred.copy(), 50, 100)
        self.canny0 = cannyImg
        image_RGBA = cv.cvtColor(cannyImg, cv.COLOR_BGR2BGRA)
        self.cannyresult[0] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        (thresh, output2) = cv.threshold(self.canny0.copy(), 70, 255, cv.THRESH_BINARY)
        image = img_as_float(output2)
        skeleton_lee = skeletonize(image > 0, method='lee')
        canny2 = img_as_ubyte(skeleton_lee)
        image_RGBA = cv.cvtColor(canny2, cv.COLOR_BGR2BGRA)
        self.cannyresult[1] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.contours(canny2, 0, dist_in_cm, contoursize)

    def prewitt(self, dist_in_cm, contoursize):
        kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernelX1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernelY1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewittX = cv.filter2D(self.grayblurred.copy(), -1, kernelX)
        prewittY = cv.filter2D(self.grayblurred.copy(), -1, kernelY)
        prewittX1 = cv.filter2D(self.grayblurred.copy(), -1, kernelX1)
        prewittY1 = cv.filter2D(self.grayblurred.copy(), -1, kernelY1)
        prewittCombined0 = cv.bitwise_or(prewittX, prewittY)
        prewittCombined1 = cv.bitwise_or(prewittX1, prewittY1)
        prewittCombined = cv.bitwise_or(prewittCombined0, prewittCombined1)
        image_RGBA = cv.cvtColor(prewittCombined, cv.COLOR_BGR2BGRA)
        self.prewittresult[0] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        (thresh, output2) = cv.threshold(prewittCombined.copy(), 70, 255, cv.THRESH_BINARY)
        image = img_as_float(output2)
        skeleton_lee = skeletonize(image > 0, method='lee')
        prewitt2 = img_as_ubyte(skeleton_lee)
        image_RGBA = cv.cvtColor(prewitt2, cv.COLOR_BGR2BGRA)
        self.prewittresult[1] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.contours(prewitt2, 1, dist_in_cm, contoursize)

    def sobel(self, dist_in_cm, contoursize):
        sobelX = cv.Sobel(self.grayblurred.copy(), cv.CV_64F, 1, 0)
        sobelY = cv.Sobel(self.grayblurred.copy(), cv.CV_64F, 0, 1)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobelCombined = cv.bitwise_or(sobelX, sobelY)
        image_RGBA = cv.cvtColor(sobelCombined, cv.COLOR_BGR2BGRA)
        self.sobelresult[0] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        (thresh, output2) = cv.threshold(sobelCombined.copy(), 70, 255, cv.THRESH_BINARY)
        image = img_as_float(output2)
        skeleton_lee = skeletonize(image > 0, method='lee')
        sobel2 = img_as_ubyte(skeleton_lee)
        image_RGBA = cv.cvtColor(sobel2, cv.COLOR_BGR2BGRA)
        self.sobelresult[1] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        self.contours(sobel2, 2, dist_in_cm, contoursize)

    def contours(self, edge, index, dist_in_cm, contoursize):
        cnts = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        original = self.image.copy()
        contourSize1 = int(contoursize)
        cnts = [x for x in cnts if cv.contourArea(x) >= contourSize1]
        for c in cnts:
            if cv.contourArea(c) < contourSize1:
                continue
            ref_object = cnts[0]
            box = cv.minAreaRect(ref_object)
            box = cv.boxPoints(box)
            box = np.array(box, dtype = "int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            dist_in_pixel = euclidean(tl, tr)
            pixel_per_cm = dist_in_pixel / int(dist_in_cm)
            for cnt in cnts:
                box = cv.minAreaRect(cnt)
                box = cv.boxPoints(box)
                box = np.array(box, dtype = "int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                cv.drawContours(original, [box.astype("int")], -1, (0, 0, 255), 2)
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
                mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
                wid = euclidean(tl, tr) / pixel_per_cm
                ht = euclidean(tr, br) / pixel_per_cm
                cv.putText(original, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(original, "{:.1f}cm".format(ht), (int(mid_pt_vertical[0] + 10), int(mid_pt_vertical[1])), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        image_RGBA = cv.cvtColor(original, cv.COLOR_BGR2BGRA)
        if index == 0:
            self.cannyresult[2] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        elif index == 1:
            self.prewittresult[2] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)
        else:
            self.sobelresult[2] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)

    def hariscorner(self, grayImg):
        original = self.image.copy()
        gray = np.float32(grayImg)
        dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv.dilate(dst, None)
        original[dst > 0.01 * dst.max()] = [255, 0, 255]
        image_RGBA = cv.cvtColor(original, cv.COLOR_BGR2BGRA)
        self.originalresult[3] = QImage(image_RGBA.data, image_RGBA.shape[1], image_RGBA.shape[0], QImage.Format_ARGB32)

def main():
    app = QApplication(sys.argv)
    win = mainwindow()
    win.show()
    stylesheet = """
                    QLineEdit{
                        font-style: italic;
                    }
                    QLineEdit::text{
                        font-style: italic;
                    }
                    QMainWindow{
                        background-color: darkgray;
                    }
                    {
                    """
    app.setStyleSheet(stylesheet)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()