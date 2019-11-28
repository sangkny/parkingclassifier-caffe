# display caffe log
import pandas as pd
import matplotlib.pyplot as plt

IsPlotSave = True
FileName = 'caffe-log-plot'
train_log = pd.read_csv("./caffe.log.train")
test_log = pd.read_csv("./caffe.log.test")
_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
plot1, = ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
plot2, = ax1.plot(test_log["NumIters"], test_log["accuracy"], 'g')
plot3, = ax1.plot(test_log["NumIters"], test_log["loss"], 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')

plt.legend([plot1, plot2, plot3], ['Train loss', 'Test Acc', 'Test loss'])

plt.savefig('{}.png'.format(FileName))
plt.show()