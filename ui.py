# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from time import sleep
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pytest import Item
from model import model
import threading
import time
import math
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        MainWindow.setMaximumSize(QtCore.QSize(1920, 1080))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # gird button layout dùng để chứa button 
        self.gridbutton = QtWidgets.QWidget(self.centralwidget)
        self.gridbutton.setGeometry(QtCore.QRect(0, 0, 1920, 100))
        self.gridbutton.setObjectName("gridbutton")
        self.gridbutton.setStyleSheet("background-color: #ffffff;")
        # button lấy đầu vào
        self.btdirinput = QtWidgets.QPushButton(self.gridbutton)
        self.btdirinput.setText("chọn đầu vào")
        self.btdirinput.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.btdirinput.clicked.connect(self.btinputevent)
        #button lưu đầu ra
        self.btsave = QtWidgets.QPushButton(self.gridbutton)
        self.btsave.setText("Lưu")
        self.btsave.setGeometry(QtCore.QRect(120, 10, 100, 30))
        self.btsave.clicked.connect(self.btsaveevent)
        # button chuyển ảnh
        self.btchuyen = QtWidgets.QPushButton(self.gridbutton)
        self.btchuyen.setText("Chuyển")
        self.btchuyen.setGeometry(QtCore.QRect(230, 10, 100, 30))
        self.btchuyen.clicked.connect(self.btchuyenevent)
        #button xóa ảnh đã chọn
        self.btxoa = QtWidgets.QPushButton(self.gridbutton)
        self.btxoa.setText("Xóa")
        self.btxoa.setGeometry(QtCore.QRect(340, 10, 100, 30))
        self.btxoa.mousePressEvent = self.removeevent
        # self.cbb = QtWidgets.QComboBox(self.gridbutton)
        # self.cbb.setGeometry(QtCore.QRect(450,10,100,30))
        # list_model = ["DACS","VGG19","resnet52","densenet121"]
        # self.cbb.addItems(list_model)
        # self.cbb.currentTextChanged.connect(self.changecbbevent)
        # subwindow dùng để phóng to ảnh chọn
        self.subwindow = None
        # thanh progress bar dùng để thể hiện quá trình chuyển đổi ảnh
        self.pbt = QtWidgets.QProgressBar(self.gridbutton)
        self.pbt.setGeometry(QtCore.QRect(50, 40, 1000, 30))
        self.pbt.setValue(0)
        self.lbchuyen = QtWidgets.QLabel(self.gridbutton)
        self.lbchuyen.setGeometry(QtCore.QRect(1100, 40, 300, 30))
        self.lbchuyen.setVisible(False)
        # dialog dùng để hiển thị lựa chọn khi bấm nút lấy ảnh đầu vào và lưu ảnh đầu ra
        self.msb = None
        # flag dùng để kiểm tra xem ảnh đã được chuyển chưa
        # nếu bằng 0 thì là chưa chuyển
        # nếu bằng 1 thì ảnh đã chuyển và grid đang hiện ảnh đầu vào
        # nếu bằng 2 thì ảnh đã chuyển và grid đang hiện ảnh đã chuuyển
        self.chuyen = 0
        # danh sách chọn index ảnh cần xóa, khi bấm nút xóa thì nó sẽ xóa ảnh có index trong danh sách này
        self.ds_chon = []
        # tạo thanh scroll cho grid
        self.gridWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridWidget.setGeometry(QtCore.QRect(0, 100, 1920, 880))
        self.gridWidget.setObjectName("gridWidget")
        
        self.gridLayout_1 = QtWidgets.QVBoxLayout(self.gridWidget)
        self.scroll = QtWidgets.QScrollArea(self.gridWidget)
        self.gridLayout_1.addWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        self.scrollWidget = QtWidgets.QWidget(self.scroll)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollWidget)
        self.gridLayout_2.setAlignment(Qt.AlignTop)
        self.gridLayout_2.setContentsMargins(5,5,5,5)
        # set event phóng to ảnh khi double click vào ảnh
        self.scrollWidget.mouseDoubleClickEvent = self.mouspressevent
        # set event chọn ảnh cần xóa khi click vào ảnh
        self.scrollWidget.mousePressEvent = self.mouseclickevent
        self.scroll.setWidget(self.scrollWidget)
        # khởi tạo model
        self.model = model()
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def changecbbevent(self, event):
        """
        <code>changecbbevent</code> is a function that takes two arguments, <code>self</code> and
        <code>event</code>. 
        <code>self</code> is a reference to the object that the function is being called on. 
        <code>event</code> is a reference to the event that triggered the function. 
        The function prints the current index of the combobox.
        
        :param event: QEvent
        """
        print(self.cbb.currentIndex())
    # event khi bấm nút chuyển
    def btchuyenevent(self, event):
        """
        If the user clicks the button, the function will start a thread to run the predict function. 
        
        If the user clicks the button again, the function will replace the images in the grid layout
        with the output images. 
        
        If the user clicks the button again, the function will replace the images in the grid layout
        with the input images. 
        
        The predict function is as follows: 
        
        # Python
        def predict(self):
                self.model.predict()
                self.chuyen = 2
        
        :param event: The event that triggered the function
        """
        if(self.chuyen == 0):
            self.t = threading.Thread(target=self.predict)
            self.t.daemon = 1
            self.t.start()
        elif self.chuyen == 1:
            for i in range(len(self.model.img_inputs)):
                widget = self.gridLayout_2.itemAt(i).widget()
                item  = widget.layout().itemAt(0)
                widget.layout().removeItem(item)
                img_out = self.model.outputs[i]
                img_out = cv2.resize(img_out,(256,256))
                self.add_img(widget, img_out)
                self.chuyen = 2
        elif self.chuyen == 2:    
            for i in range(len(self.model.img_inputs)):
                widget = self.gridLayout_2.itemAt(i).widget()
                item  = widget.layout().itemAt(0)
                widget.layout().removeItem(item)
                img_out = self.model.img_inputs[i]
                img_out = cv2.resize(img_out,(256,256))
                self.add_img(widget, img_out)
                self.chuyen = 1
    # event xóa ảnh
    def removeevent(self, event):
        """
        It removes the selected images from the grid and the model
        
        :param event: The event that triggered the callback
        """
        self.ds_chon.sort(reverse=True)
        self.remove_widget()
        for i in self.ds_chon:
            self.model.img_inputs.pop(i)
            self.model.list_file.pop(i)
            if len(self.model.outputs) > i:
                self.model.outputs.pop(i)
        self.ds_chon.clear()
        self.add_grid()
    # event lưu ảnh
    def btsaveevent(self, event):
        """
        It saves the data in the model to a file
        
        :param event: The event that was triggered
        """
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Open Directory")
        self.model.save(filepath)
        self.msb = QtWidgets.QDialog()
        self.msb.setWindowFlag(Qt.FramelessWindowHint)
        self.msb.setWindowTitle("Thông báo")
        self.msb.setGeometry(QtCore.QRect(800, 500, 400, 100))
        self.msb.setStyleSheet("background-color: rgb(255, 255, 255);")
        dllayout = QtWidgets.QVBoxLayout(self.msb)
        self.msb.setLayout(dllayout)
        lb = QtWidgets.QLabel()
        #lb.setGeometry(QtCore.QRect(100, 10, 300, 100))
        lb.setAlignment(Qt.AlignCenter)
        dllayout.addWidget(lb)
        if(len(self.model.outputs) == 0):
            lb.setText("Chưa có dữ liệu để lưu")
        else:
            lb.setText("Đã lưu thành công")
        lb.setFont(QtGui.QFont("Times", 14, QtGui.QFont.Bold))
        dlbt = QtWidgets.QPushButton()
        dlbt.setText("OK")
        dlbt.clicked.connect(self.msb.close)
        dllayout.addWidget(dlbt)
        self.msb.exec_()
    # hàm chọn widget để xóa
    def mouseclickevent(self, event):
        """
        If the user clicks on a widget, change its background color to blue.
        
        :param event: The event object that was passed to the event handler
        """
        ps = event.pos()
        id = self.get_id(ps)
        item = self.gridLayout_2.itemAt(id)
        if item != None:
            if id in self.ds_chon:
                item = item.widget()
                item.setStyleSheet("background-color: #ffffff;")
                self.ds_chon.remove(id)
            else:
                item = item.widget()
                item.setStyleSheet("background-color: #33E0FF;")
                self.ds_chon.append(id)
            
    # lấy id của widget       
    def get_id(self,ps):
        """
        It takes a point (ps) and returns the id of the tile that the point is in
        
        :param ps: the position of the mouse
        :return: The id of the image that was clicked.
        """
        psx= ps.x()
        psy = ps.y()
        idy = math.floor(psy/307)
        idx = math.floor(psx/310)
        id = idy*6+idx
        return id
    # event khi nhấn chuột trái vào một widget trong grid
    def mouspressevent(self, event):
        """
        It creates a subwindow to display the image of the selected widget.
        
        :param event: The event object that was generated by the mouse press
        """
        ps = event.pos()
        id = self.get_id(ps)
        widget = self.gridLayout_2.itemAt(id)
        if widget != None:
            widget = widget.widget()
            # tạo một dialog để hiển thị ảnh của widget được chọn
            self.createsubwindow(id)
    # tạo một dialog để hiển thị ảnh của widget được chọn trong grid       
    def createsubwindow(self, id):
        """
        It creates a new window, adds a widget to it, and then adds two images to the widget
        
        :param id: the index of the image in the list of images
        """
        self.subwindow = None
        self.subwindow = QtWidgets.QMainWindow()
        self.subwindow.setWindowTitle("Hinh anh")
        self.subwindow.setGeometry(QtCore.QRect(100, 100, 1200, 600))
        
        subwidget = QtWidgets.QWidget(self.subwindow)
        subwidget.setGeometry(QtCore.QRect(0, 0,1200, 600))
        gridsubwidget = QtWidgets.QHBoxLayout()
        subwidget.setLayout(gridsubwidget)
        img1 = self.model.img_inputs[id]
        img1 = cv2.resize(img1,(512,512))
        self.add_img(gridsubwidget, img1)
        if len(self.model.outputs) > id:
            img2 = self.model.outputs[id]
            img2 = cv2.resize(img2,(512,512))
            self.add_img(gridsubwidget, img2)
        self.subwindow.setCentralWidget(subwidget)
        self.subwindow.show()      
    # event khi nhấn chuột vào nút chọn đầu vào
    def btinputevent(self, event):
        """
        It creates a dialog box with two buttons, one for selecting an image and one for selecting a
        folder.
        
        :param event: The event that was triggered
        """
        self.msb = QtWidgets.QDialog()
        self.msb.setGeometry(QtCore.QRect(800, 500, 400, 100))
        self.msb.setWindowTitle("Thông báo")
        dllayout = QtWidgets.QGridLayout(self.msb)
        self.msb.setLayout(dllayout)
        dllb = QtWidgets.QLabel()
        dllb.setText("chọn loại thư mục đầu vào")
        dllb.setFont(QtGui.QFont("Times", 14))
        dllayout.addWidget(dllb, 0, 0,1,2)
        dlbtanh = QtWidgets.QPushButton()
        dlbtanh.setText("ảnh")
        dlbtanh.clicked.connect(self.btanhclickevent)
        dllayout.addWidget(dlbtanh, 1, 0 )
        dlbtthumuc = QtWidgets.QPushButton()
        dlbtthumuc.setText("thư mục")
        dlbtthumuc.clicked.connect(self.btthumucclickevent)
        dllayout.addWidget(dlbtthumuc, 1, 1)
        self.msb.exec_()
    # chọn ảnh
    def btanhclickevent(self, event):
        """
        It opens a file dialog, gets the filepaths, and then calls a function to display the images.
        
        :param event: The event that triggered the slot
        """
        self.msb.close()
        filepath = QtWidgets.QFileDialog.getOpenFileNames(self.centralwidget, "Open File")
        for i in range(len(filepath[0])):
            self.model.get_img(filepath[0][i])
        self.remove_widget()
        self.add_grid()
        self.chuyen = 0
       
    # chọn thư mục để lấy ảnh   
    def btthumucclickevent(self, event):
        """
        It opens a file dialog, gets the file path, and then calls a function to remove the current
        widget and add a new one
        
        :param event: The event that triggered the slot
        """
        self.msb.close()
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Open Directory")
        self.model.get_inputs(filepath)
        self.remove_widget()
        self.add_grid()
        self.chuyen = 0
        
    # hàm xóa widget trong grid   
    def remove_widget(self):
        """
        While there are widgets in the layout, remove the first widget and delete it
        """
        while(True):
            n_w = self.gridLayout_2.count()
            if n_w ==0:
                break
            item = self.gridLayout_2.itemAt(0)
            self.gridLayout_2.removeItem(item)
            item.widget().deleteLater()
    # hàm này không biết làm gì
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
    # hàm tạo widget trong grid
    def create_widget(self, img1):
        """
        It creates a widget with a layout and adds an image to it.
        
        :param img1: the image to be displayed
        :return: A widget with a layout and an image.
        """
        widget = QtWidgets.QWidget()
        widget.setFixedSize(QtCore.QSize(300, 300))
        widget.setStyleSheet("background-color: #ffffff;")
        widget.setObjectName("widget")
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)

        self.add_img(widget, img1)
        return widget
    # hàm thêm ảnh vào widget trong grid
    def add_img(self, widget, img1):
        """
        It takes an image, converts it to a QPixmap, and adds it to a QWidget.
        
        :param widget: the widget that the image will be added to
        :param img1: numpy array
        """
        layout = widget.layout()
        limg = QtWidgets.QLabel()
        size = img1.shape
        limg.setFixedSize(QtCore.QSize(size[0],size[1]))
        pixmap = self.array2qpixmap(img1)
        limg.setPixmap(pixmap)
        layout.addWidget(limg)
    # hàm chuyển đổi ảnh thành bit map để hiện trong widget   
    def array2qpixmap(self, array):
        """
        It takes a numpy array and returns a QPixmap
        
        :param array: The array to be converted to a QPixmap
        :return: A QPixmap object.
        """
        
        img_data = array.data
        stride = array.strides[0]
        size = array.shape
        qimage = QtGui.QImage(img_data, size[0], size[1], stride, QtGui.QImage.Format_RGB888)
        pixmap=QtGui.QPixmap(qimage)
        return pixmap
    # hàm thêm widget vào grid
    def add_grid(self, ):
        """
        It adds widgets to a grid layout
        """
        n_img = len(self.model.img_inputs)
        w = n_img//6 if n_img%6 == 0 else n_img//6+1
        for i in range(w):
            h = 6 if n_img - i*6 > 6 else n_img - i*6
            for j in range(h):
                img_1 = self.model.img_inputs[i*6+j]
                img_1 = cv2.resize(img_1,(256,256))
                widget = self.create_widget(img_1)
                widget.setObjectName("widget_"+str(i*6+j))
                self.gridLayout_2.addWidget(widget, i, j)
        num_img = str(len(self.model.img_inputs)) + " ảnh"
        self.lbchuyen.setVisible(True)
        self.lbchuyen.setText(num_img)
    # hàm chuyển đổi ảnh thành mùa đông dùng multi thread
    def predict(self):
        """
        It takes a list of images, and for each image, it calls the model.predict() function.
        """
        
        self.btchuyen.setEnabled(False)
        self.btsave.setEnabled(False)
        self.btdirinput.setEnabled(False)
        self.btxoa.setEnabled(False)
        input = np.asarray(self.model.img_inputs)
        self.model.outputs.clear()
        for i in range(len(input)):
            string = "Đang chuyển đổi ảnh thành mùa đông: " + str(i+1) + "/" + str(len(input))
            self.lbchuyen.setText(string)
            self.model.predict(input[i])
            dem = int(i/len(input)*100)
            # self.pbt.setValue(dem)
            
        # self.pbt.setValue(100)
        self.btchuyen.setEnabled(True)
        self.btsave.setEnabled(True)
        self.btdirinput.setEnabled(True)
        self.btxoa.setEnabled(True)
        string = "Đã chuyển đổi xong " + str(len(input)) + " ảnh"
        self.lbchuyen.setText(string)
        self.chuyen = 1
        
# hàm main
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.add_grid()
    MainWindow.show()
    
    sys.exit(app.exec_())
