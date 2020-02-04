# USAGE
# This script describes how to use caffe or darknet model to detecte objects.
# This script is for development purposes only.
#
# python svg_object_detect.py --conf --input img.jpg

# inform the user about framerates and speeds
print("[INFO] NOTE: It only takes images for caffe and darknet models. " \
	"In near future, a video will be accepted as well")

# import the necessary packages

from datetime import datetime
from threading import Thread
import numpy as np
import argparse
import time
import cv2
import os

def upload_file(tempFile, client, imageID):
	# upload the image to Dropbox and cleanup the tempory image
	print("[INFO] uploading {}...".format(imageID))
	path = "/{}.jpg".format(imageID)
	client.files_upload(open(tempFile.path, "rb").read(), path)
	tempFile.cleanup()

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(frame, classes, classId, conf, left, top, right, bottom, color=(0,255,0)):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    if args["showImgDetailText"]:
        cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                (0, 255, 255), cv2.FILLED)
        # cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="Path to the input video file")
ap.add_argument("-mt", "--modeltype", type=str, default='caffe', help='model type : caffe or darknet')

# debug options
ap.add_argument('--showImgDetail', type = int, default=1, help ='show image in detail')
ap.add_argument('--showText', type=int, default=1, help='show text in the ouput.')
ap.add_argument('--showRoiImgDetail', type = int, default=1, help ='show Roi image in detail')
ap.add_argument('--showImgDetailText', type = int, default=1, help ='flag to show texts in ROI image')
ap.add_argument('--debugTextDetail', type=int, default=1, help='flag for displaying texts in Detail')

args = vars(ap.parse_args())
# -c config/config.json -i sample_data/20180911_113611_cam_0.avi

args["input"] = "D:/sangkny/pyTest/MLDL/NexQuadDataSets/6phase/incorrect_20191010/NOBJ3376.jpg"
args["modeltype"] = 'caffe'

mtype = 1 if args["modeltype"] == "caffe" else 0


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "Cars"]
# Load names of classes
modelBaseDir = "C:/Users/mmc/workspace/yolo"
#modelBaseDir = "C:/Users/SangkeunLee/workspace/yolo"
classesFile = modelBaseDir + "/data/itms/itms-classes.names"
clASSES = None
with open(classesFile, 'rt') as f:
    cLASSES = f.read().rstrip('\n').split('\n')


# load our serialized model from disk
print("[INFO] loading model...")
if mtype == 1:
	prototxt_path = os.getcwd()+"/LeNet32x40_3_ive_deploy.prototxt"
	model_path = os.getcwd()+"/Nexquad-delivery/lenet32x40_3_ive_iter_9500.caffemodel"
	net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
else:
	modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l.cfg"
	modelWeights = modelBaseDir + "/config/itms-dark-yolov3-tiny_3l_150000.weights" # /config/itms-dark-yolov3-tiny_100000.weights
	# modelConfiguration = modelBaseDir + "/config/itms-dark-yolov3.cfg"
	# modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final_20200117_416.weights"
	#modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final_20200113.weights"
	#modelWeights = modelBaseDir + "/config/itms-dark-yolov3_final.weights"
	net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

#net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(args["input"])
time.sleep(2.0)


# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
H = None
W = None

# Initialize the parameters
confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416 # 32*10 #32*10  # 608     #Width of network's input image # 320(32*10)
inpHeight = 416 #32*9 #32*9 # 608     #Height of network's input image # 288(32*9) best
scaleFactor = 1

model_channel = 1
model_width = 40
model_height = 32 # for caffe

track_object = 1

trackers = []
trackableObjects = {}

# keep the count of total number of frames
totalFrames = 0

# initialize the log file
logFile = None


# loop over the frames of the stream or images
while True:
	# grab the next frame from the stream, store the current
	# timestamp, and store the new date
	ret, frame  = vs.read()
	ts = datetime.now()
	newDate = ts.strftime("%m-%d-%y")

	# check if the frame is None, if so, break out of the loop
	if frame is None:
		break


	# resize the frame
	if(model_channel == 1 and frame.shape[2]>1):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(frame, (model_width, model_height))


	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]


	# initialize our list of bounding box rectangles returned by
	# either (1) our object detector or (2) the correlation trackers
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % track_object == 0:
		# initialize our new set of object trackers
		trackers = []

		if mtype == 1: # use caffe
			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			# blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
			# 	ddepth=cv2.CV_8U)
			# net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
			# 	127.5, 127.5])
			blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(model_width, model_height), mean=[0,0,0], swapRB=True,
										 crop=False)
			net.setInput(blob)
			# Runs the forward pass to get output of the output layers
			detections = net.forward()
			# compute performance time / milisecs
			et, _ = net.getPerfProfile()
			tlabel = et * 1000.0 / cv2.getTickFrequency()  # milisecs
			print(tlabel)

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence`
				# is greater than the minimum confidence
				if confidence > conf["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a car, ignore it
					if CLASSES[idx] != "car":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					rect = (startX, startY, endX, endY)



		else: # use darknet Yolov3
			# loop for multi-block roi ----------------------- see analysis_roi_~ for detail
			classIds = []
			confidences = []
			boxes = []
			etimes = []  # elapse time for net.forward
			debugFrame = frame.copy()
			# bboxes = bboxes[100:]
			# bboxes = [(0, 0, W, H)]
			wr, hr = W/1920, H/1080
			bboxes = [(round(340*wr), round(32*hr), round(1024*wr), round(992*hr)), (round(772*wr), round(0*hr), round(320*wr), round(288*hr))]
			for bidx, bb in enumerate(bboxes):
				[bx, by, bwidth, bheight] = bb
				subFrame = frame[by:by + bheight, bx:bx + bwidth]
				subFrame = cv2.resize(subFrame, (inpWidth, inpHeight))
				# sub frame information
				subclassIds = []
				subconfidences = []
				subboxes = []
				# Create a 4D blob from a frame.
				blob = cv2.dnn.blobFromImage(subFrame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
				# Sets the input to the network
				net.setInput(blob)
				# Runs the forward pass to get output of the output layers
				outs = net.forward(getOutputsNames(net))
				if args["debugTextDetail"]:
					print("subROI: {}, blob: {}".format(bidx, blob.shape))
					print(getOutputsNames(net))

				# compute performance time / milisecs
				et, _ = net.getPerfProfile()
				tlabel = et * 1000.0 / cv2.getTickFrequency()  # milisecs
				etimes.append(tlabel)
				# let's correct coordinates as
				# corrent only center positions   [x,y, width, height] is  [detection[0], detection[1], detection[2], detection[3]]
				[rcx, rcy, rwidth, rheight] = bboxes[bidx]  # this is bb
				cnt = 0
				# save information
				for out in outs:
					if args["debugTextDetail"]:
						print("out.shape : ", out.shape)
					for detection in out:
						# if detection[4]>0.001:
						scores = detection[5:]
						classId = np.argmax(scores)
						# if scores[classId]>confThreshold:
						confidence = scores[classId]
						# Remove the bounding boxes with low confidence
						if detection[4] > confThreshold:
							if args["debugTextDetail"]:
								print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
						# print(detection)
						if confidence > confThreshold:
							center_x = rcx + int(detection[0] * rwidth)
							center_y = rcy + int(detection[1] * rheight)
							width = int(detection[2] * rwidth)
							height = int(detection[3] * rheight)
							left = int(center_x - width / 2)
							top = int(center_y - height / 2)
							classIds.append(classId)
							confidences.append(float(confidence))
							boxes.append([left, top, width, height])
							# sub frame
							subclassIds.append(classId)
							subconfidences.append(float(confidence))
							subboxes.append([left, top, width, height])
							cnt = cnt + 1
				if args["debugTextDetail"]:
					print('# of candidates for {}-th roi: {}'.format(bidx, cnt))

				if args["showImgDetail"]:
					roi_ious = []
					# draw each sub frame information
					subindices = cv2.dnn.NMSBoxes(subboxes, subconfidences, confThreshold, nmsThreshold)
					if args["showImgDetail"]:
						# debugFrame.fill(0)
						debugFrame = frame.copy()
						cv2.rectangle(debugFrame, (rcx, rcy), (rcx + rwidth, rcy + rheight), (255, 0, 255), 2)
						# if args.showText:
						textLabel = 'Roi (x,y,width,height, # objs):({}, {}, {}, {}, #{}) in {} msec'.format(rcx, rcy,
																											 rwidth,
																											 rheight,
																											 len(
																												 subindices),
																											 str(
																												 tlabel))
						cv2.putText(debugFrame, textLabel, (rcx, rcy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
					for i in subindices:
						i = i[0]
						box = subboxes[i]
						left = box[0]
						top = box[1]
						width = box[2]
						height = box[3]
						# search GT and compute IOU to find out the corresponding objects
						boxB = [left, top, left + width, top + height]
						# # roi_ious=[idx if bb_intersection_over_union(boxB, cvtYolo2XY([frameWidth, frameHeight], gtbox)) > 0.5 else '' for idx, gtbox in enumerate(GTBoxes)]
						# for idx, gtbox in enumerate(GTBoxes):
						# 	if bb_intersection_over_union(boxB, cvtYolo2XY([frameWidth, frameHeight], gtbox)) > 0.5:
						# 		roi_ious.append(idx)

						if args["showImgDetail"]:
							drawPred(debugFrame, cLASSES, subclassIds[i], subconfidences[i], left, top, left + width,
									 top + height, (0, 255, 0))

					if args["showImgDetail"]:
						# cv.imshow("subROI:"+str(sfidx), tmpFrame)
						cv2.imshow("subROI", debugFrame)
						cv2.waitKey(1)

					# # put the roi_iou_information
					# if len(roi_ious) > 0:
					# 	# outInfoFile.write("%.6f %.6f %.6f %.6f\n" % (xcen, ycen, w, h))
					# 	lineText = "%d %d %d %d" % (bx, by, bwidth, bheight)  # roi box (x,y, width, height)
					# 	for idx in range(0, len(roi_ious)):
					# 		lineText = lineText + ' ' + str(roi_ious[idx])
					# 	lineText = lineText + '\n'
					# 	outInfoFile.write(lineText)

			# Perform non maximum suppression to eliminate redundant overlapping boxes with
			# lower confidences.
			indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

			if args["debugTextDetail"]:
				print("# of Roi:{}, # of Cands:{}, # of object:{}".format(len(bboxes), len(boxes), len(indices)))

			for i in indices:
				i = i[0]
				box = boxes[i]
				left = box[0]
				top = box[1]
				width = box[2]
				height = box[3]
				(startX, startY, endX, endY) = (int(left), int(top), int(left+width), int(top+height))
				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

				drawPred(frame, cLASSES, classIds[i], confidences[i], left, top, left + width, top + height)

			# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
			# t, _ = net.getPerfProfile()
			tot = 0
			for etime in etimes:
				tot = tot + etime
			label = 'Inference time: %.2f ms' % (tot)
			if args["debugTextDetail"]:
				# label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
				print(label)
			if args["showImgDetailText"]:
				cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing
	# throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))


	# tracking starts
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, if there is a trackable object and its speed has
		# not yet been estimated then estimate it
		elif not to.estimated:
			# check if the direction of the object has been set, if
			# not, calculate it, and set it
			if to.direction is None:
				y = [c[0] for c in to.centroids]
				direction = centroid[0] - np.mean(y)
				to.direction = direction

			# if the direction is positive (indicating the object
			# is moving from left to right)
			if to.direction > 0:
				# check to see if timestamp has been noted for
				# point A
				if to.timestamp["A"] == 0 :
					# if the centroid's x-coordinate is greater than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] > conf["speed_estimation_zone"]["A"]:
						to.timestamp["A"] = ts
						to.position["A"] = centroid[0]

				# check to see if timestamp has been noted for
				# point B
				elif to.timestamp["B"] == 0:
					# if the centroid's x-coordinate is greater than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] > conf["speed_estimation_zone"]["B"]:
						to.timestamp["B"] = ts
						to.position["B"] = centroid[0]

				# check to see if timestamp has been noted for
				# point C
				elif to.timestamp["C"] == 0:
					# if the centroid's x-coordinate is greater than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] > conf["speed_estimation_zone"]["C"]:
						to.timestamp["C"] = ts
						to.position["C"] = centroid[0]

				# check to see if timestamp has been noted for
				# point D
				elif to.timestamp["D"] == 0:
					# if the centroid's x-coordinate is greater than
					# the corresponding point then set the timestamp
					# as current timestamp, set the position as the
					# centroid's x-coordinate, and set the last point
					# flag as True
					if centroid[0] > conf["speed_estimation_zone"]["D"]:
						to.timestamp["D"] = ts
						to.position["D"] = centroid[0]
						to.lastPoint = True

			# if the direction is negative (indicating the object
			# is moving from right to left)
			elif to.direction < 0:
				# check to see if timestamp has been noted for
				# point D
				if to.timestamp["D"] == 0 :
					# if the centroid's x-coordinate is lesser than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] < conf["speed_estimation_zone"]["D"]:
						to.timestamp["D"] = ts
						to.position["D"] = centroid[0]

				# check to see if timestamp has been noted for
				# point C
				elif to.timestamp["C"] == 0:
					# if the centroid's x-coordinate is lesser than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] < conf["speed_estimation_zone"]["C"]:
						to.timestamp["C"] = ts
						to.position["C"] = centroid[0]

				# check to see if timestamp has been noted for
				# point B
				elif to.timestamp["B"] == 0:
					# if the centroid's x-coordinate is lesser than
					# the corresponding point then set the timestamp
					# as current timestamp and set the position as the
					# centroid's x-coordinate
					if centroid[0] < conf["speed_estimation_zone"]["B"]:
						to.timestamp["B"] = ts
						to.position["B"] = centroid[0]

				# check to see if timestamp has been noted for
				# point A
				elif to.timestamp["A"] == 0:
					# if the centroid's x-coordinate is lesser than
					# the corresponding point then set the timestamp
					# as current timestamp, set the position as the
					# centroid's x-coordinate, and set the last point
					# flag as True
					if centroid[0] < conf["speed_estimation_zone"]["A"]:
						to.timestamp["A"] = ts
						to.position["A"] = centroid[0]
						to.lastPoint = True

			# check to see if the vehicle is past the last point and
			# the vehicle's speed has not yet been estimated, if yes,
			# then calculate the vehicle speed and log it if it's
			# over the limit
			if to.lastPoint and not to.estimated:
				# initialize the list of estimated speeds
				estimatedSpeeds = []

				# loop over all the pairs of points and estimate the
				# vehicle speed
				for (i, j) in points:
					# calculate the distance in pixels
					d = to.position[j] - to.position[i]
					distanceInPixels = abs(d)

					# check if the distance in pixels is zero, if so,
					# skip this iteration
					if distanceInPixels == 0:
						continue

					# calculate the time in hours
					t = to.timestamp[j] - to.timestamp[i]
					timeInSeconds = abs(t.total_seconds())
					timeInHours = timeInSeconds / (60 * 60)

					# calculate distance in kilometers and append the
					# calculated speed to the list
					distanceInMeters = distanceInPixels * meterPerPixel
					distanceInKM = distanceInMeters / 1000
					estimatedSpeeds.append(distanceInKM / timeInHours)

				# calculate the average speed
				to.calculate_speed(estimatedSpeeds)

				# set the object as estimated
				to.estimated = True
				print("[INFO] Speed of the vehicle that just passed"\
					" is: {:.2f} MPH".format(to.speedMPH))

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
			, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4,
			(0, 255, 0), -1)

		# check if the object has not been logged
		if not to.logged:
			# check if the object's speed has been estimated and it
			# is higher than the speed limit
			if to.estimated and to.speedMPH > conf["speed_limit"]:
				# set the current year, month, day, and time
				year = ts.strftime("%Y")
				month = ts.strftime("%m")
				day = ts.strftime("%d")
				time = ts.strftime("%H:%M:%S")

				# check if dropbox is to be used to store the vehicle
				# image
				if conf["use_dropbox"]:
					# initialize the image id, and the temporary file
					imageID = ts.strftime("%H%M%S%f")
					tempFile = TempFile()
					cv2.imwrite(tempFile.path, frame)

					# create a thread to upload the file to dropbox
					# and start it
					t = Thread(target=upload_file, args=(tempFile,
						client, imageID,))
					t.start()

					# log the event in the log file
					info = "{},{},{},{},{},{}\n".format(year, month,
						day, time, to.speedMPH, imageID)
					logFile.write(info)

				# otherwise, we are not uploading vehicle images to
				# dropbox
				else:
					# log the event in the log file
					info = "{},{},{},{},{}\n".format(year, month,
						day, time, to.speedMPH)
					logFile.write(info)

				# set the object has logged
				to.logged = True

	# if the *display* flag is set, then display the current frame
	# to the screen and record if a user presses a key
	if conf["display"]:
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check if the log file object exists, if it does, then close it
if logFile is not None:
	logFile.close()

# close any open windows
cv2.destroyAllWindows()

# clean up
print("[INFO] cleaning up...")
vs.release()