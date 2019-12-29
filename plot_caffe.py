# display caffe log
"""## this py code can be used as csv file reader.
# 0. in the caffe, create log file with caffe train -solver /path/to/solver.prototxt 2>&1 | tee caffe.log
# 1. after training, type './tools/extra/parse_log.py /path/to/caffe.log /path/to/ourdir/
# 2. then edit this file to save
## --------------------------------------------------
## to plot train/test acc/loss
## you need to phase: TRAIN/TEST in the input/outs for configureation file such as ive_3.prototxt
## However, you need only TEST phase at the end of the file when converting caffemodel to bin file
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

IsPlotSave = True
TrainTestBoth = False
FileName = 'caffe-log-plot-20191228-brightness_0.4_Test'
dirPath = './20191228_data'
logFileName = 'train_br_040'
outFile = os.path.join(dirPath,logFileName)
if TrainTestBoth:
    train_log = pd.read_csv(str(outFile + ".log.train")) # when test only exists, train_log == test_log
    test_log = pd.read_csv(str(outFile + ".log.test"))
else:
    train_log = pd.read_csv(str(outFile + ".log.test")) # when test only exists, train_log == test_log
    test_log = pd.read_csv(str(outFile + ".log.test"))

_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
plot0, = ax1.plot(train_log["NumIters"], train_log["accuracy"], 'b')
plot1, = ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
plot2, = ax1.plot(test_log["NumIters"], test_log["accuracy"], 'g.')
plot3, = ax1.plot(test_log["NumIters"], test_log["loss"], 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')

plt.legend([plot0, plot1, plot2, plot3], ['Train Acc', 'Train loss', 'Test Acc', 'Test loss'])

plt.savefig('{}.png'.format(FileName))
plt.show()