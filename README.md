# parkingclassifier-caffe
Produced files (components) from caffe-related training data
and it can be used inside the Caffe

#### Update 
- 20191221: train with lenet32x40_3 for 20191221_data which has been augmented with options of constras, sat, hue, and rnd vertical flip only in training phase.
- 20191228: train with lenet32x40_3 for 20191228_data changing brightness 0.2 to 0.4 of 20191221 data

# Procedure 
0. develop a pytorch model and convert the model into caffe's files using pytorch2caffe project for easy architecture development
1. I assume that caffe-related files including .prototxt(s) (_solver.prototxt} and _model.prototxt) and resides in the source root (ex. /workspace/)
2. edit xxx.prototxt for a target database and xxx_solver.prototxt for the xxx.prototxt location 
3. first edit and run list_files.sh in the source directory (ex. /workspace/parkingclassifier-caffe/20190812_data/)
4. edit create_pk_class.sh and run it to generate database files in the caffe 
5. train the data after editting a solver file.

# Model files and their descriptions
- lenet32x40_1 : 20190820 12phase lenet32x40_1
- lenet32x40_2 : 20191125 6 phase 10000 samples training 14->28: acc: 99.5
- lenet32x40_3 : 20191126 6 phase 10000 samples training 50->28: acc: 99.7
- lenet32x40_3_1:LeNet32x40_2 Test under lenet32x40_3 
- lenet32x40_3 : 20191221 6 phase all data(46000 each class) including augmented data (contrast: 0.2, sat: 0.2, hue: 0.2 with data_split): acc: 99.5
- lenet32x40_3 : 20191228 6 phase all data(46000 each class) including augmented data (brightness: 0.4, sat: 0.2, hue: 0.2 with data_split): acc: 99.75

# Model file confirmation for the given system
1. ./build/tools/ive_tool_caffe 0 h w ch /workspace/parkingclassifier-caffe/lenet32x40_2.prototxt 
# Solver
_xxx_solver.prototxt

# Training with log 
0. in the caffe root
1. ./build/tools/caffe train -solver /workspace/parkingclassifier-caffe/xxx_solover.prototxt 2>&1 | tee your_name.log

* 1-1. Saving the information to display the curve after training as follows:
	GLOG_logtostderr=1 ./build/examples/train_net.bin solver.prototxt 2> caffe.log 
* 1-2. This command does not display the process status during training, instead it saves the log information into the given log file (caffe.log)
* 1-3. To plot the log, use parse_log.py and plot the graph with plot_caffe.py I made.


# Converting to binary for the company
1. ./build/tools/ive_tool_caffe 1 h w ch (channel: 3 for color) /workspace/parkingclassifier-caffe/lenet32x40_2.prototxt \
	/workspace/parkingclassifier-caffe/lenet32x40_2.caffemodel /workspace/parkingclassifier-caffe/lenet32x40_2.bin
2. ** important ** To convert ive-caffe in a success into a bin file, the prototxt 	should include TEST only in accuracy layer at the last part. 
- 2.1 However, to make log and draw the accuracy graphs for TRAIN/TEST phases, the last accuracy layer should include both.

# Note:
Using docker 
	1. in windows, Lower/Capital character file name is not effective
	2. in ubuntu, it is sensitive
	3. sudo docker run -it --ipc=host -p 9428:22 -p 9488:8888 -p 9482:6006 -v ~/workspace:/workspace sangkny/caffe:caffe-ive-bin /bin/bash
	4. under the docker, cd /opt/source/caffe (here is caffe root)
	
# Draw the Accuracy/Loss Graph
0. We assumed that we got log file during the Train phase as the above <traing with log>
1. In caffe, ./tools/extra/parse_log.py /path/from/logfile/xxx.log /path/to/output => makes xx.train and xx.test  
2. use plot-caffe.py I made 
3. Please include Train/Test phases in the Accuracy Layer. IVE does not allow the train phase included in accurracy Layer in the prototxt file. 
- 3.1 However, for the purpose of drawing the Accuracy/Loss graph, it does not matter.
