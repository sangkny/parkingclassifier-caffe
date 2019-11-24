# parkingclassifier-caffe

Produced files (components) from caffe-related training data
and it can be used inside the Caffe

0. develop a pytorch model and convert the model into caffe's files using pytorch2caffe project for easy architecture development
0. I assume that caffe-related files including .prototxt(s) (_solver.prototxt} and _model.prototxt) and resides in the source root (ex. /workspace/) 
1. first edit and run list_files.sh in the source directory (ex. /workspace/20190812_data/)
2. edit create_pk_class.sh and run it to generate database files
3. train the data after editting a solver file.

