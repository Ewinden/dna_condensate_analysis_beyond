import warnings
warnings.simplefilter("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
from Tkinter import Tk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
from sys import exit
from gaussfitter import gaussfit
from math import sqrt
# ~ import pandas
# ~ from skimage import morphology
# ~ from scipy.misc import toimage
# ~ from skimage.measure import regionprops
# ~ from skimage.morphology import label
# ~ from skimage import feature
# ~ from xlrd import open_workbook
# ~ import pdb
# ~ import imutils
# ~ import time

def gaussian(x, a, b, c, d): 
	return a*np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2))) + d

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	pill = abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
	return pill

def detect_name(file_name):
	
	'''from wScan checks if a given folder is a good fit'''
	
	
	if file_name == ():
		print("nothing is chosen,Exit!")
		exit()
	print("The path you chose is " + str(file_name)	)
	
	
def my_convert_16U_2_8U(image):
	
	'''from wScan converts a 16 bit image to an 8 bit one using a flat scalar from the lowest to the highest value'''
	
	min_ = np.amin(image)
	max_ = np.amax(image)
	a = 255/float(max_-min_)
	b = -a*min_
	#print min_, max_ , a, b 
	img8U = np.zeros(image.shape,np.uint8)
	cv2.convertScaleAbs(image,img8U,a,b)
	return img8U


def displayPoint(image, point):
	
	'''displays given point in red in an 8bit image'''
	
	img = cv2.imread(image,1)
	#imblur = cv2.GaussianBlur(img,(5,5),0)
	img8 = my_convert_16U_2_8U(img)
	cv2.line(img8, (point[0],point[1]), (point[0],point[1]),(0,0,255),1)
	cv2.imshow('Window',img8)
	cv2.waitKey(0)
	
	
def drawGaussCircle(img, center, width):
	
	'''pull up an image and draws a circle to show the radius determined'''
	
	h, w = img.shape[:2]
	print(h,w)
	print(center)
	print(width)
	img8 = my_convert_16U_2_8U(img)
	img4x = cv2.resize(img8, (w, h), cv2.INTER_NEAREST)
	center4x = (int(center[0]), int(center[1]))
	cv2.circle(img4x, (center4x[0], center4x[1]), int(width), (0,255,0),1)
	cv2.rectangle(img4x, (center4x[0], center4x[1]),(center4x[0], center4x[1]),(0,0,255),1)
	cv2.imshow('Window',img4x)
	cv2.waitKey(0)
	
	
def getNeighbors(locs):
	
	'''Given a list of points, returns a shape around the points of all neighbors'''
	
	trub = list(locs)
	for pix in locs:
		for x in [pix[0]-1,pix[0],pix[0]+1]:
			for y in [pix[1]-1,pix[1],pix[1]+1]:
				if (x,y) not in trub:
					trub.append((x,y))
	return trub


def checkOnePeak(img):
	
	'''Checks if only one peak is in image by distance transform and then counting contours'''
	
	imb = cv2.GaussianBlur(my_convert_16U_2_8U(img), (5,5), 0)
	ret1, thr1 = cv2.threshold(imb, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	dt = cv2.distanceTransform(thr1, cv2.DIST_L2, 3)
	ret2, thr2 = cv2.threshold(dt.astype('uint8'), 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	contours = cv2.findContours(thr2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours[1]) == 1:
		return(True)
	else:
		return(False)


def getBrightBox(img, point, outLength):
	
	'''Given a point, returns a square around the point size 2n+1'''
	shap = img.shape
	startx = int(max(0, point[1] - outLength))
	starty = int(max(0, point[0] - outLength))
	endx = int(min(shap[1], point[1] + outLength+1))
	endy = int(min(shap[0], point[0] + outLength+1))
	brightBox = img[starty:endy,startx:endx]
	avg = brightBox.mean()

	# ~ img8 = my_convert_16U_2_8U(img)
	# ~ cv2.rectangle(img8,(startx,starty),(endx,endy),(255),thickness=1)
	# ~ cv2.line(img8, (point[1],point[0]), (point[1],point[0]),(255,255,255),thickness=1)
	# ~ cv2.imshow('image',img8[starty-50:endy+50,startx-50:endx+50])
	# ~ cv2.waitKey(0)
	# ~ cv2.destroyAllWindows()
	
	'''gaussfitter requires float objects in the array passed as data for the fitting'''
	return((brightBox.astype(float)), startx, starty, avg)


def gaussBoxFitPoint(img, point, outLength):
	
	'''Given a point in an image and outLength to test, returns a 2d gaussian fit to the object around the point and its errors'''
	
	#print(point)
	bbout = getBrightBox(img, point, outLength)
	pixSet = bbout[0]
	# ~ sideLength = (outLength*2)+1
	if checkOnePeak(pixSet) == True:
		# ~ xes = np.arange((outLength*2)+1)
		bits = gaussfit((pixSet), return_error='True', circle='True')
		# ~ drawGaussCircle(img, (bbout[2]+bits[0][2], bbout[1]+bits[0][3]), bits[0][4])
		return([bits[0],bits[1]])
	else:
		return([np.zeros(5),np.zeros(5)])

def findBrightObjects(img):
	
	'''Finds bright objects using a threshold and findContours function returning a list of contours with summaries'''
	
	conts=[]
	shap=img.shape
	img8=my_convert_16U_2_8U(img)
	#img4x = cv2.resize(img8, (shap[1]/2,shap[0]/2), cv2.INTER_NEAREST)
	imgb = cv2.GaussianBlur(img8, (5,5), 0)
	ret1, thr1 = cv2.threshold(imgb, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	# ~ cv2.imshow("img", thr1)
	# ~ cv2.waitKey(0)
	# ~ cv2.destroyAllWindows()
	
	contours = cv2.findContours(thr1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cont in contours[1]:
		area = cv2.contourArea(cont)
		per = cv2.arcLength(cont, True)
		(x,y), rad = cv2.minEnclosingCircle(cont)
		mask = np.zeros(img.shape,np.uint8)
		cv2.drawContours(mask, [cont], -1, 255,1)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask)
		mean = cv2.mean(img, mask)[0]
		conts.append([cont,area,per,max_loc[0],max_loc[1],rad,mean])
	return(conts)

def filterContours(conts, contlimits = [[0,100],[0,100],[0,2560],[0,2160],[0,100],[0,1000]]):
	
	'''Filter contours list by area, perimeter, xposition, y position, radius'''
	
	newConts=[]
	areac = contlimits[0]
	perc = contlimits[1]
	radc = contlimits[2]
	xc = contlimits[3]
	yc = contlimits[4]
	meanc = contlimits[5]
	print('Contour Filter Start: '+str(len(conts)))
	for obj in conts:
		area = obj[1]
		per = obj[2]
		x = obj[3]
		y = obj[4]
		rad = obj[5]
		mean = obj[6]
		if area >= areac[0] and area <= areac[1] and per >= perc[0] and per <= perc[1] and x >= xc[0] and x <= xc[1] and y >= yc[0] and y <= yc[1] and rad >= radc[0] and rad <= radc[1] and mean >= meanc[0] and mean <= meanc[1]:
			newConts.append(obj)
	print('Contour Filter End: '+str(len(newConts)))
	return(newConts)
	
def filterGaussians(bgauses, glimits=[[0,1000],[0,1000000],[0,30],[0,30],[0,10],[0,1],[0,1],[0,1],[0,1],[0,1]]):
	
	'''Filter gaussians list by gaussian returns'''
	
	newGauss=[]
	print('Gaussian Filter Start: '+str(len(bgauses)))
	for obj in bgauses:
		gooutok = True
		for i in range(0,9):
			if obj[6+i] <= glimits[i][0] or obj[6+i] > glimits[i][1]:
				# ~ print(str(obj[5][0][i]) + ', ' + str(glimits[i]))
				gooutok = False
				break
		if gooutok == True:
			newGauss.append(obj)
	print('Gaussian Filter End: '+str(len(newGauss)))
	return(newGauss)


def findGaussSummary(gauses):
	
	'''Find a summary from a list of gaussians returned by gaussBoxFitPoint'''
	
	''' Get lists of values '''
	bgs = []
	peaks = []
	xs = []
	ys = []
	widths = []
	bges = []
	peakes = []
	xes = []
	yes = []
	widthes = []
	for obj in gauses:
		if float(any(obj)) <= 0 or float(any(obj)) <= 0:
			continue
		else:
			bgs.append(obj[0])
			peaks.append(obj[1])
			xs.append(obj[2])
			ys.append(obj[3])
			widths.append(obj[4])
			bges.append(obj[5])
			peakes.append(obj[6])
			xes.append(obj[7])
			yes.append(obj[8])
			widthes.append(obj[9])
	
	''' Get medians '''
	bgmed = np.median(bgs)
	peakmed = np.median(peaks)
	xmed = np.median(xs)
	ymed = np.median(ys)
	widthmed = np.median(widths)
	bgemed = np.median(bges)
	peakemed = np.median(peakes)
	xemed = np.median(xes)
	yemed = np.median(yes)
	widthemed = np.median(widthes)
	
	''' Get standard deviations '''
	bgsd = np.std(bgs)
	peaksd = np.std(peaks)
	xsd = np.std(xs)
	ysd = np.std(ys)
	widthsd = np.std(widths)
	bgesd = np.std(bges)
	peakesd = np.std(peakes)
	xesd = np.std(xes)
	yesd = np.std(yes)
	widthesd = np.std(widthes)
	
	outtro = [[bgmed,peakmed,xmed,ymed,widthmed,bgemed,peakemed,xemed,yemed,widthemed],[bgsd,peaksd,xsd,ysd,widthsd,bgesd,peakesd,xesd,yesd,widthesd]]
	return(outtro)

def showDaPoints(img, objects):
	cimg = np.stack((img,)*3, axis=-1)
	shap=cimg.shape
	cimg8=my_convert_16U_2_8U(cimg)
	cimg8r = cv2.resize(cimg8, (shap[1]/2,shap[0]/2), cv2.INTER_NEAREST)
	for obj in objects:
		cv2.circle(cimg, (int(obj[2]),int(obj[3])), int(obj[4]*4), (65335,65335,65335),1)
	cv2.imshow("img", cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

'''
-----------------------------------------------------------START OF MAIN-----------------------------------------------------------------
'''

outL = 15

'''Define limits'''
areac = [0,100]
perc = [1,100]
radc = [0.5,100]
xc = [200,2360]
yc = [200,1960]
meanc = [0,65536]
contLimits = [areac, perc, radc, xc, yc, meanc]
bgl = [0,1000]
peakl = [0,65536]
xl = [0,30]
yl = [0,30]
widthl = [0,10]
bgel = [0,1]
peakel = [0,1]
xel = [0,1]
yel = [0,1]
widthel = [0,1]
gausLimits = [bgl,peakl,xl, yl, widthl,bgel,peakel,xel,yel,widthel]

'''Get directory to analyze images in'''
dirtouse = askdirectory(title="Select Directory of images")
biganalysis = os.path.join(dirtouse, 'complete_analysis.txt')
intoc = open(biganalysis, 'w')
intoc.write('image, area, perimeter, x, y, radius, meanfi, background, peak, xin, yin, width, background.error, peak.error, xin.error, yin.error, width.error, wnm, pzd\n')

'''Move through all files in directory, analysing every .tif file'''
for filename1 in os.listdir(dirtouse):
	if filename1.endswith('.tif'):
		print(filename1)
		'''Define files'''
		pictouse = os.path.join(dirtouse,filename1)
		lstshort = pictouse[:(pictouse.rfind('.'))]
		newfac = lstshort+'_analysis.txt'
		imgname = filename1[:(filename1.rfind('.'))]
		
		'''Open files'''
		img = cv2.imread(pictouse,-1)
		shap = img.shape
		
		'''Find contours to predict beads with'''
		conts = findBrightObjects(img)
		limitedConts = filterContours(conts, contLimits)
		
		'''Building table and gaussian analysis for found obects from findBrightObjects'''
		completepairs=[]
		for line in limitedConts:
			out = ', '.join(str(o) for o in line)+'\n'
			area = line[1]
			per = line[2]
			x = line[3]
			y = line[4]
			pnt = [y,x]
			rad = line[5]
			mean = line[6]
			pair = gaussBoxFitPoint(img, pnt, outL)
			wnm = pair[0][4]*114
			p2dff = -436.07197+58.30619*sqrt(wnm)
			completepairs.append([area, per, x, y, rad, mean, pair[0][0], pair[0][1], pair[0][2], pair[0][3], pair[0][4], pair[1][0], pair[1][1], pair[1][2], pair[1][3], pair[1][4], wnm, p2dff])
		
		
		'''Write the filtered file'''
		intob = open(newfac, 'w')
		intob.write('area, perimeter, x, y, radius, meanfi, background, peak, xin, yin, width, background.error, peak.error, xin.error, yin.error, width.error, wnm, pzd\n')
		
		'''Summarize and filter gaussian curves'''
		# gaussianSummary = findGaussSummary(completepairs)
		filteredGaussians = filterGaussians(completepairs, gausLimits)
		# showDaPoints(img, filteredGaussians)
		for to in filteredGaussians:
			out=''
			for li in to:
				out += str(li) + ', '
			out=out[:-2]+'\n'
			intob.write(out)
			intoc.write(str(imgname)+', '+out)
		intob.close()
intoc.close()
