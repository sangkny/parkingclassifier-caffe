# parkingclassifier-caffe

Produced files (components) from caffe-related training data
and it can be used inside the Caffe

0. develop a pytorch model and convert the model into caffe's files using pytorch2caffe project for easy architecture development
0. I assume that caffe-related files including .prototxt(s) (_solver.prototxt} and _model.prototxt) and resides in the source root (ex. /workspace/) 
1. first edit and run list_files.sh in the source directory (ex. /workspace/parkingclassifier-caffe/20190812_data/)
2. edit create_pk_class.sh and run it to generate database files
3. train the data after editting a solver file.

# model files
lenet32x40_1 : 20190820 12phase lenet32x40_1
lenet32x40_2 : 20191125 6 phase 10000 samples training 14->28: acc: 99.5
lenet32x40_3 : 20191126 b pahse 10000 samples training 50->28: acc: 

# model file confirmation for the given system
1. ./build/tools/ive_tool_caffe 0 h w ch /workspace/parkingclassifier-caffe/lenet32x40_2.prototxt 
# solver
_xxx_solver.prototxt

# training with log 
0. in the caffe root
1. ./build/tools/caffe train -solver /workspace/parkingclassifier-caffe/xxx_solover.prototxt 2>&1 | tee your_name.log

1-1. Saving the information to display the curve after training as follows:
	GLOG_logtostderr=1 ./build/examples/train_net.bin solver.prototxt 2> caffe.log 
1-2. This command does not display the process status during training, instead it saves the log information into the given log file (caffe.log)
1-3. To plot the log, use parse_log.py and plot the graph with plot_caffe.py I made.


# convert to binary for the company
1. ./build/tools/ive_tool_caffe 1 h w ch (channel: 3 for color) /workspace/parkingclassifier-caffe/lenet32x40_2.prototxt \
	/workspace/parkingclassifier-caffe/lenet32x40_2.caffemodel /workspace/parkingclassifier-caffe/lenet32x40_2.bin
	
# Note:
Using docker 
	1. in windows, Lower/Capital character file name is not effective
	2. in ubuntu, it is sensitive