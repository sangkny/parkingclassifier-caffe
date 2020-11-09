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

logisIn_high = True
IsPlotSave = True
TrainTestBoth = True
TrainTestLossConstrain = True # train loss < test loss for determining maximum test accuracy
Train_batch_Size = str(512) # training batch_size
if logisIn_high:
    FileName = 'train_20201107_lr0001_ga0001_b512_50000_HighGpu'
    dirPath = './20200826_3chs_data_br04'
    logFileName = 'train_20201107_lr0001_ga0001_b512_50000_HighGpu'
else:
    FileName = 'train_20200826_40x32_4_lr0005_v3_3chs_br04_60000'
    dirPath = './20200826_3chs_data_br04'
    logFileName = 'train_20200826_40x32_4_lr0005_v3_3chs_br04_60000'

outFile = os.path.join(dirPath,logFileName)
if TrainTestBoth:
    train_log = pd.read_csv(str(outFile + ".log.train")) # when test only exists, train_log == test_log
    test_log = pd.read_csv(str(outFile + ".log.test"))
else:
    train_log = pd.read_csv(str(outFile + ".log.test")) # when test only exists, train_log == test_log
    test_log = pd.read_csv(str(outFile + ".log.test"))

# search max acc
import numpy as np
if(train_log.get('accuracy') is None):
    print('Train accuracy does not exist')
else:
    log_num, log_acc = list(train_log["NumIters"]), list(train_log["accuracy"])
    idx = np.argmax(log_acc)
    txtstr = 'Train log max acc: {} at {} with idx {}\n '.format(round(log_acc[idx],5),int(log_num[idx]), idx)
    print(txtstr)

log_num, log_acc = list(test_log["NumIters"]), list(test_log["accuracy"])
log_test_loss, log_train_loss = list(test_log["loss"]), list(train_log["loss"])
idx = -1 # for maximum accuracy position
if TrainTestLossConstrain:
    AccIdx = np.argsort(log_acc)[::-1] # sort accuracy indices descending order
    o_idx = np.argmax(log_acc)
    for accidx in range(len(AccIdx)):
        if(log_test_loss[AccIdx[accidx]]>=log_train_loss[AccIdx[accidx]]):
            idx = AccIdx[accidx]
            break
    if (o_idx != idx):
        print('Original max idx: {} at {} ({}/{}), and final max idx: {} at {}({}/{}) \n'.format(o_idx, int(log_num[o_idx]), round(log_acc[o_idx],5), round(log_test_loss[o_idx],5), idx, int(log_num[idx]), round(log_acc[idx],5),round(log_test_loss[idx],5)))
else:
    idx = np.argmax(log_acc)
txtstr = 'Test max acc/loss:{}/{} at {} with idx {} , Training Err:{} (b:{})\n'.format(round(log_acc[idx],5),round(log_test_loss[idx],5),int(log_num[idx]), idx, round(log_train_loss[idx],5), Train_batch_Size)
print(txtstr)


fig, ax1 = plt.subplots(figsize=(15, 10))

#put text in the figure
size = fig.get_size_inches()*fig.dpi # plot size in pixels
center = size/2 # center pixel
print(center)
fig.text((center[0]/(2*1500)),(center[1]/(1000)),txtstr, fontsize=15)  # (x,y) normalized ratio
                                                    # can be fig.text((0.5),(0.5),txtstr)
ax2 = ax1.twinx()
if train_log.get('accuracy') is None:
    plot0, = ax1.plot(test_log["NumIters"], test_log["accuracy"], 'b') # replace
else:
    plot0, = ax1.plot(train_log["NumIters"], train_log["accuracy"], 'b')
plot1, = ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4)
plot2, = ax1.plot(test_log["NumIters"], test_log["accuracy"], 'g.')
plot3, = ax1.plot(test_log["NumIters"], test_log["loss"], 'r')

ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')

plt.legend([plot0, plot1, plot2, plot3], ['Train Acc', 'Train loss', 'Test Acc', 'Test loss'])
plt.title(txtstr)

plt.savefig('{}.png'.format(FileName))
plt.show()
