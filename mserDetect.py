import cv2
import numpy as np
import sys
import PIL
import argparse
import math
import os
from PIL import Image

# Display image
def displayImg(name, img):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	img = cv2.resize(img, (600, 600))
	cv2.imshow(name, img)
	cv2.moveWindow(name, 50, 0)
	cv2.waitKey(0)
	cv2.destroyWindow(name)

# def getImage(img):
# 	img = cv2.imread(img)
# 	return img

"""
takes system arg to
find folder with
pictures
"""
def inputFolder():
	parser = argparse.ArgumentParser()
	parser.add_argument("folder_name")
	args = parser.parse_args()
	folderName = str(args.folder_name)
	return folderName

"""
Stores all the file names and 
paths from given directory in
an array
"""
def listFolder(dirName):
	folder = dirName
	fNames = []
	fPaths = []
	for file in os.listdir(folder):
		if file != '.DS_Store':
			fNames.append(file)
			# print file
			filepath = os.path.join(folder, file)
			# f = open(filepath, 'r')
			fPaths.append(filepath)
			# print f.read()
			# f.close
	return fNames, fPaths

# Use MSER to detect features
def getMSER(img):
	size = (150, 150)
	vis = img.copy()
	mser = cv2.MSER_create(_delta = 5, _min_area = 100, _max_area=14400, _max_variation = 0.15, _min_diversity = .4, _max_evolution = 200,  _area_threshold = 1.01, _min_margin = 0.003, _edge_blur_size = 5)
	gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

	gray_blur = cv2.GaussianBlur(gray, (7,7), 0)

	regions = mser.detectRegions(gray_blur, None)
	hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]
	target_hulls = []
	
	for hull in hulls:
		is_duplicate(target_hulls,hull)
	
	just_hulls = []
	for hull in hulls:
	    just_hulls.append(hull)
    	
	#cv2.polylines(vis, just_hulls, 1, (0, 255, 0))
	#displayImg("just_hulls", vis)
	path = 'newPics'

	for index,hull_tup in enumerate(target_hulls):
		hull,_ = hull_tup
		x,y,width,height =  cv2.boundingRect(hull)
		#crops roi (should add some  leeway)
		inc_w = int(width/7)
		inc_h =int(height/7)

		roi = vis[y-inc_h if y-inc_h >0 else y :y+height+inc_h, x-inc_w if x-inc_w >0 else x :x+width+inc_w]
		#resize to 150 by 150
		im = Image.fromarray(roi)
		roiA = im.resize(size, Image.ANTIALIAS)
		roiA = np.array(roiA)
		cv2.imwrite(os.path.join(path, "roi%d.jpeg" %index),roiA)
		displayImg("roi", roiA)

	return vis

def centroid_distance(c1,c2):
	x1,y1 = c1
	x2,y2 = c2
	dist = int(math.sqrt(math.pow((x1-x2),2) + math.pow((y1-y2),2)))
	return dist

def get_centroid(hull):
	M = cv2.moments(hull)
	hcentroid_x = int(M['m10']/M['m00'])
	hcentroid_y = int(M['m10']/M['m00'])
	return (hcentroid_x, hcentroid_y)

def is_duplicate(reference_hulls,hull):
	#center_threshold
	center_threshold = 0
	hcentroid = get_centroid(hull)
	for index, ref in enumerate(reference_hulls):
		ref_hull, ref_centroid = ref

		if centroid_distance(hcentroid,ref_centroid) <= center_threshold:
			if cv2.contourArea(hull) >= cv2.contourArea(ref_hull):
				reference_hulls[index] = (hull, hcentroid)
				return True
			else:
				return True
	reference_hulls.append((hull,hcentroid))
	return False

# def inputImg():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("image")
# 	args = parser.parse_args()
# 	image_file = str(args.image)
# 	img = cv2.imread(image_file)
# 	return img, image_file

def main():
	# img, name = inputImg()	
	# displayImg(name,img)

	# mserPic = getMSER(img)
	fileNames = []
	filePaths = []
	fname = inputFolder()
	fileNames, filePaths = listFolder(fname)
	count = 0
	while (count < len(filePaths)):
		img = cv2.imread(filePaths[count])
		getMSER(img)
		#displayImg(fileNames[count], img)
		# print('filename = ' + fileNames[count])
		count = count + 1

main()
	