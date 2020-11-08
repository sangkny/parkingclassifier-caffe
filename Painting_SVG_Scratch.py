import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

class CWidget(QWidget):

    def __init__(self):
        super().__init__()
        # overall form
        formbox = QHBoxLayout() # it is not shown yet
        self.setLayout(formbox) # now form box is set

        # left, right layout boxes
        left = QVBoxLayout()
        right = QVBoxLayout()

        # -->  left box components <--
        # create GroupBox 1 and set it to the left layout
        gb = QGroupBox('Drawing Type') # 그리기 종류
        left.addWidget(gb)
        # layout inside gb
        box = QVBoxLayout()
        gb.setLayout(box)
        # radio buttons on gb1
        text = ['line', 'curve', 'rectangle', 'ellipse']
        self.radiobtns = [] # empty yet

        for i in range(len(text)):
            self.radiobtns.append(QRadioButton(text[i], self))
            self.radiobtns[i].clicked.connect(self.radioClicked)
            box.addWidget(self.radiobtns[i])

        self.radiobtns[0].setChecked(True)
        self.drawType = 0

        # group box 2 for pen settings
        gb = QGroupBox('Pen settings')
        left.addWidget(gb)          # insert into left VBoxlayout
        grid = QGridLayout()
        gb.setLayout(grid)          # attach to Groupbox

        label=QLabel('line thickness')
        grid.addWidget(label, 0, 0)  # attach label at (0,0) location on grid widget
        self.combo = QComboBox()
        grid.addWidget(self.combo, 0, 1)
        for i in range(1, 21):
            self.combo.addItem(str(i))

        label = QLabel('line color')
        grid.addWidget(label, 1,0)
        self.pencolor = QColor(0,0,0)
        self.penbtn = QPushButton()
        self.penbtn.setStyleSheet('background-color: rgb(0,0,0)')
        self.penbtn.clicked.connect(self.showColorDlg)
        grid.addWidget(self.penbtn,1,1)

        # group box 3 for brush
        gb = QGroupBox('Brush settings')
        left.addWidget(gb)
        # prepare horizontal component arrangement
        hbox= QHBoxLayout()
        gb.setLayout(hbox)
        label = QLabel('brush color')
        hbox.addWidget(label)
        self.brushcolor = QColor(255,255,255)
        self.brushbtn = QPushButton()
        self.brushbtn.setStyleSheet('background-color: rgb(255,255,255)')
        self.brushbtn.clicked.connect(self.showColorDlg)
        hbox.addWidget(self.brushbtn)

        left.addStretch(1) # gab after left panel (layout)

        # -->  right layout components <--
        # Graphic view on right-sided Box in the main layout location (right layout box)
        self.view = CView(self)
        right.addWidget(self.view)

        # == > left, right boxes docking to main form box < ==
        # ==------------------------------------------------==
        formbox.addLayout(left)
        formbox.addLayout(right)
        formbox.setStretchFactor(left, 0)
        formbox.setStretchFactor(right, 1)
        self.setGeometry(100,100,800,500) # view location and size setting.

    # member function definition, implementations
    def radioClicked(self):
        for i in range(len(self.radiobtns)):
            if self.radiobtns[i].isChecked():
                self.drawType = i
                # print('draw type: %s'%i)
                break

    def showColorDlg(self):
        # color selection dialogue
        color = QColorDialog.getColor()
        sender = self.sender()
        # check if the color is valid or not
        if sender == self.penbtn and color.isValid():
            self.pencolor = color
            self.penbtn.setStyleSheet('background-color: {}'.format(color.name()))
        else:
            self.brushcolor = color
            self.brushbtn.setStyleSheet('background-color: {}'.format(color.name()))

# QGraphicsView display QGraphicsScene
class CView(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # object lists
        self.items = []
        self.start = QPointF()  # starting point
        self.end = QPointF()    # end point

        self.setRenderHint(QPainter.HighQualityAntialiasing)

    def mousePressEvent(self, e):
        print('mouse button pressed!')
        if e.button() == Qt.LeftButton:
            self.start = e.pos()
            self.end = e.pos()

    def mouseReleaseEvent(self, e):
        print('mouse button released!')
        if e.button() == Qt.LeftButton:
            pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex())  # common for all drawtypes
            if self.parent().drawType == 0: # line type
                self.items.clear()
                line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())

                self.scene.addLine(line, pen) # draw a line at final
                print('items #: {}'.format(len(self.items)))
            # drawType ==1 # curve : noting to do here
            if self.parent().drawType == 2 or self.parent().drawType == 3: # rect or ellipse
                brush = QBrush(self.parent().brushcolor)
                self.items.clear()
                rect = QRectF(self.start, self.end)
                if(self.parent().drawType ==2):
                    self.scene.addRect(rect, pen, brush)
                else:
                    self.scene.addEllipse(rect, pen, brush)


        #print(' scene items #:{}'.format(len(self.scene.items())))

    def mouseMoveEvent(self, e):
        print('mouse is moving!!')
        # e.buttons() returns integer value while e.button returns Qt.Nobutton when moving
        if e.buttons() & Qt.LeftButton:
            self.end = e.pos()

        pen = QPen(self.parent().pencolor, self.parent().combo.currentIndex())
        # draw a line
        if self.parent().drawType == 0:
            # remove the previous line
            if len(self.items) > 0:
                self.scene.removeItem(self.items[-1])
                del(self.items[-1])

            # add current line
            line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
            self.items.append(self.scene.addLine(line, pen))
        # draw a curve
        if self.parent().drawType == 1:
            # Using path
            path = QPainterPath()
            path.moveTo(self.start)
            path.lineTo(self.end)
            self.scene.addPath(path, pen)

            #using Line
            # line = QLineF(self.start.x(), self.start.y(), self.end.x(), self.end.y())
            # self.scene.addLine(line, pen)

            # set the end points to the starting points
            self.start = e.pos()

        # draw a rect
        if self.parent().drawType == 2:
            brush = QBrush(self.parent().brushcolor)
            if(len(self.items)> 0) :
                self.scene.removeItem(self.items[-1])
                del(self.items[-1])

            rect = QRectF(self.start, self.end)
            self.items.append(self.scene.addRect(rect, pen, brush))
        # draw a circle
        if self.parent().drawType == 3:
            brush = QBrush(self.parent().brushcolor)
            if(len(self.items)>0):
                self.scene.removeItem(self.items[-1])
                del(self.items[-1])
            rect = QRectF(self.start, self.end)
            self.items.append(self.scene.addEllipse(rect, pen, brush))

    def moveEvent(self, e): # even when form size is changed !!
        print('moveEvent is occurred!!')
        rect = QRectF(self.rect())
        rect.adjust(0,0,-2,-2)              # when form size changes, the scroll will be disappeared due to [-2,-2]
        self.scene.setSceneRect(rect)
        print(rect)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    w.show()
    sys.exit(app.exec_())