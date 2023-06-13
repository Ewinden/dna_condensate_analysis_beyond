'''
Finds all objects in a set of images in a folder and fits gaussian curves to each, adding them to a data file.

Created on Thu Dec 13 2018

@author: ewinden
'''

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
	# ~ print(h,w)
	# ~ print(center)
	# ~ print(width)
	img8 = my_convert_16U_2_8U(img)
	img4x = cv2.resize(img8, (w, h), cv2.INTER_NEAREST)
	center4x = (int(center[0]), int(center[1]))
	cv2.circle(img4x, (center4x[0], center4x[1]), int(width), (0,255,0),1)
	cv2.rectangle(img4x, (center4x[0], center4x[1]),(center4x[0], center4x[1]),(0,0,255),1)
	cv2.imshow('Window',img4x)
	cv2.waitKey(0)
	
def drawPoints(img, centerl, name="Image"):
	
	'''displays an image with many points displayed'''
	
	h, w = img.shape[:2]
	img8 = my_convert_16U_2_8U(img)
	img4x = cv2.resize(img8, (w, h), cv2.INTER_NEAREST)
	shap4 = img4x.shape
	blank = np.zeros(shap4)
	for center in centerl:
		center4x = (int(center[0]), int(center[1]))
		# ~ cv2.circle(img4x, (center4x[0], center4x[1]), int(center[2]), (255,255,255),1)
		cv2.circle(blank, (center4x[0], center4x[1]), int(center[2]), (255,255,255),1)
		# ~ cv2.rectangle(img4x, (center4x[0], center4x[1]),(center4x[0], center4x[1]),(255,255,255),1)
		cv2.rectangle(blank, (center4x[0], center4x[1]),(center4x[0], center4x[1]),(255,255,255),1)
		
	# ~ cv2.imshow('Window',img4x)
	cv2.imshow(name,blank)
	# ~ cv2.waitKey(0)
	# ~ cv2.destroyAllWindows()
	
	
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
	# ~ print(point)
	# ~ print(brightBox)
	# ~ print(brightBox[outL][outL])
	# ~ print(img[point[0]][point[1]])

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
	
	bbout = getBrightBox(img, point, outLength)
	pixSet = bbout[0]
	if checkOnePeak(pixSet) == True:
		# ~ xes = np.arange((outLength*2)+1)
		bits = gaussfit((pixSet), return_error='True', circle='True')
		# ~ drawGaussCircle(img, (bbout[2]+bits[0][2], bbout[1]+bits[0][3]), bits[0][4])
		return([bits[0],bits[1]])
	else:
		return([np.zeros(5),np.zeros(5)])

def findBrightObjects(img, adans=[51,-15]):
	
	'''Finds bright objects using a threshold and findContours function returning a list of contours with summaries'''
	
	conts=[]
	shap=img.shape
	img8=my_convert_16U_2_8U(img).copy()
	#img4x = cv2.resize(img8, (shap[1]/2,shap[0]/2), cv2.INTER_NEAREST)
	#imb = cv2.GaussianBlur(img8, (5,5), 0)
	# ~ ret1, thr1 = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
	thr2 = cv2.adaptiveThreshold(img8,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,adans[0],adans[1])
	# ~ cv2.imshow("img1", thr1)
	if showimgs == True:
		cv2.imshow("threshold", thr2)
	contours = cv2.findContours(thr2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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

def filterContours(conts, contlimits = [[0,100],[0,100],[0,2560],[0,2160],[0,100],[0,1000],[-10,10],[-10,10]]):
	
	'''Filter contours list by area, perimeter, xposition, y position, radius'''
	
	newConts=[]
	points=[]
	closepoints=[]
	areac = contlimits[0]
	perc = contlimits[1]
	radc = contlimits[4]
	xc = contlimits[2]
	yc = contlimits[3]
	meanc = contlimits[5]
	xdisc = contlimits[6]
	ydisc = contlimits[7]
	for i,l in enumerate(conts):
		points.append([i,l[3],l[4]])
	print('Contour Filter Start: '+str(len(conts)))
	for ico,obj in enumerate(conts):
		gooutok=True
		extracted = points[ico]
		# ~ print('extract: ' + str(extracted))
		exrange = range(extracted[1]+xdisc[0],extracted[1]+xdisc[1])
		eyrange = range(extracted[2]+ydisc[0],extracted[2]+ydisc[1])
		for lic, litem in enumerate(points):
			# ~ print('item: ' + str(litem))
			if lic == ico:
				continue
			if (litem[1] in exrange) and (litem[2] in eyrange):
				closepoints.append([litem[1],litem[2],2])
				gooutok=False
		for i in range(0,6):
			if contlimits[i][0] >= obj[i+1] or contlimits[i][1] <= obj[i+1]:
				gooutok = False
		if gooutok == True:
			newConts.append(obj)
	print('Contour Filter End: '+str(len(newConts)))
	if showimgs == True:
		drawPoints(img,closepoints,'ClosePoints')
	return(newConts)
	
def filterGaussians(bgauses, glimits=[[0,1000],[0,1000000],[0,30],[0,30],[0,10],[0,1],[0,1],[0,1],[0,1],[0,1]]):
	
	'''Filter gaussians list by gaussian returns'''
	
	newGauss=[]
	print('Gaussian Filter Start: '+str(len(bgauses)))
	for obj in bgauses:
		gooutok = True
		for i in range(0,9):
			if obj[6+i] <= glimits[i][0] or obj[6+i] >= glimits[i][1]:
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

# ~ print "\n please choose picture!"
# ~ pictouse = askopenfilename(initialdir = "/",title = "Select file")
# ~ directory = (pictouse[:(pictouse.rfind('/')+1)])

pictouse = '/home/edwin/Downloads/sus49.tif'
directory = '/home/edwin/Downloads/'

outL = 5
showimgs = False

'''
Define limits, as [lowerbound,upperbound]
contour limits:
	areac: area returned in findcontours, can be 0
	perc: perimeter returned in findcontours
	perc: redius returned in findcontours
	xc: x location of maximum value in contour
	yc: y location of maximum value in contour
	meanc: mean intensity of object
'''
areac = [0,100]
perc = [1,100]
radc = [0,100]
xc = [0,2360]
yc = [0,1960]
meanc = [0,65536]
xdis = [-5,5]
ydis = [-5,5]
contLimits = [areac, perc, xc, yc, radc, meanc, xdis, ydis]
'''
Define limits, as [lowerbound,upperbound]
gaussian limits:
	bgl: background estimated in gaussian
	peakl: peak estimated in gaussian
	xl: x location within small area, should be 15
	yl: y location within small area, should be 15
	widthl: width (FWHM) estimated in gaussian
	bgel: background error estimated in gaussian
	peakel: peak error estimated in gaussian
	xel: x location error within small area
	yel: y location error within small area
	widthel: width error estimated in gaussian
'''
bgl = [0,100000]
peakl = [0,65536]
xl = [0,30]
yl = [0,30]
widthl = [-1,100]
bgel = [-1,1]
peakel = [-1,1]
xel = [-1,1]
yel = [-1,1]
widthel = [-1,1]
gausLimits = [bgl,peakl,xl, yl, widthl,bgel,peakel,xel,yel,widthel]
'''
Define inputs for adaptive thresholding
	adaarea: region to search to make threshold
	adacon: constant to be subtracted from each value
'''
adaarea = 51
adacon = -15
adas = [adaarea, adacon]

'''Define files'''
lstshort = pictouse[:(pictouse.rfind('.'))]
newfac = lstshort+'_analysis.txt'

'''Opens files'''
img = cv2.imread(pictouse,-1)
shap = img.shape
if showimgs == True:
	cv2.imshow("original", my_convert_16U_2_8U(img))

'''Finds contours to predict beads with'''
conts = findBrightObjects(img)
if showimgs == True:
	grump = []
	for line in conts:
		grump.append([line[3],line[4],line[5]])
	drawPoints(img, grump, 'All Contours')
limitedConts = filterContours(conts, contLimits)
if showimgs == True:
	grump = []
	for line in limitedConts:
		grump.append([line[3],line[4],line[5]])
	drawPoints(img, grump, 'Filtered Contours')

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
	xg = x-outL+pair[0][2]
	yg = y-outL+pair[0][3]
	# ~ print(str(int(xg))+", " + str(int(yg)))
	# ~ print(img[int(yg)][int(xg)])
	# ~ print("XY: "+str(x)+", "+str(y)+"; GXY: "+str(xg)+", "+str(yg)+"; G+: "+str(pair[0][2])+", "+str(pair[0][3]))
	completepairs.append([area, per, x, y, rad, mean, pair[0][0], pair[0][1], pair[0][2], pair[0][3], pair[0][4], pair[1][0], pair[1][1], pair[1][2], pair[1][3], pair[1][4], wnm, p2dff])

if showimgs == True:
	grump = []
	for line in completepairs:
		grump.append([line[2],line[3],line[10]])
	drawPoints(img, grump, 'All Gaussians')

'''Trying to summarize and filter gaussian curves'''
# ~ gaussianSummary = findGaussSummary(completepairs)
filteredGaussians = filterGaussians(completepairs, gausLimits)
if showimgs == True:
	grump = []
	for line in filteredGaussians:
		grump.append([line[2],line[3],line[10]])
	drawPoints(img, grump, 'Filtered Gaussians')
	cv2.waitKey(0)
	cv2.destroyAllWindows()

'''Write the filtered file'''
intob = open(newfac, 'w')
intob.write('area, perimeter, x, y, radius, meanfi, background, peak, xin, yin, width, background.error, peak.error, xin.error, yin.error, width.error, wnm, pzd\n')
for to in filteredGaussians:
	out=''
	for li in to:
		out += str(li) + ', '
	out=out[:-2]+'\n'
	intob.write(out)
intob.close()
	
		



'''

# ~ print "\n please choose coordinate list from imagej!"
# ~ lsttouse = askopenfilename(title = "Select file")

# ~ lsttouse = '/home/eamon/Downloads/yuminbeads/mount 1/6.txt'
# ~ pictouse = '/home/eamon/Downloads/yuminbeads/mount 1/6.tif'



newfh = lstshort+'_handanalysis.txt'
newft = lstshort+'_comparedanalysis.txt'
newfa = lstshort+'_analysis_nofilter.txt'


intoa = open(newfa, 'w')
intoa.write('area, perimeter, radius, x, y, mean fi, background, peak, xin, yin, width, background error, peak error, x error, y error, width error, wnm, pzd\n')


	# pairstr = ''
	# for d in pair:
		# for l in d:
			# pairstr+=', '
			# pairstr+=str(l)
	# out = str(area) + ', ' + str(per) + ', ' + str(rad) + ', ' + str(x) + ', ' + str(y) + pairstr + ', ' + str(mean) + ', ' + str(wnm) + ', ' + str(p2dff)+'\n'
	# intoa.write(out)
# intoa.close()


#intoh = open(newfh, 'w')
#intoh.write('name, slice, id, xpick, ypick, background, peak, xin, yin, width, background error, peak error, x error, y error, width error, average, wnm, pzd\n')

#Building table and gaussian analysis for hand selected objects
# ~ for line in list(locsfile)[1:]:
	# ~ linls = line.split()
	# ~ pointv = linls[0]
	# ~ xpos = int(linls[1])
	# ~ ypos = int(linls[2])
	# ~ slicen = int(linls[3])
	# ~ idn = int(linls[5])
	# ~ pnt = [ypos,xpos]
	# ~ pair = gaussBoxFitPoint(img, pnt, outL)
	# ~ wnm = pair[0][4]*114
	# ~ p2dff = -436.07197+58.30619*sqrt(wnm)
	# ~ pairstr = ''
	# ~ for d in pair:
		# ~ for l in d:
			# ~ pairstr+=', '
			# ~ pairstr+=str(l)
	# ~ out = str(pointv) + ', ' + str(slicen) + ', ' + str(idn) + ', ' + str(xpos) + ', ' + str(ypos) + pairstr + ', ' + str(wnm) + ', ' + str(p2dff)+'\n'
	# ~ intoh.write(out)
# ~ intoh.close()

#Comparing hand selected and found objects
 

# ~ intoh = open(newfh, 'r')
# ~ intoa = open(newfac, 'r')
# ~ intot = open(newft, 'w')
# ~ intohls = list(intoh)[1:]
# ~ intoals = list(intoa)[1:]
# ~ print(intohls[0])
# ~ print(intoals[0])
# ~ intot.write('name, slice, id, hx, hy, hbackground, hpeak, hxin, hyin, hwidth, hbackground_error, hpeak_error, hx_error, hy_error, hwidth_error, haverage, hwnm, hpzd, area, perimeter, ax, ay, radius, meanfi, abackground, apeak, axin, ayin, awidth, abackground_error, apeak_error, ax_error, ay_error, awidth_error, awnm, apzd\n')
# ~ for ih, lineh in enumerate(intohls):
	# ~ linehls = lineh.split(',')
	# ~ if float(linehls[9])==0 or float(linehls[14])==0:
		# ~ continue
	# ~ for ia, linea in enumerate(intoals):
		# ~ lineals = linea.split(', ')
		# ~ if float(lineals[9])==0 or float(lineals[14])==0:
			# ~ continue
		# ~ if (abs(float(linehls[3]) - float(lineals[2])) <= float(10) and (abs(float(linehls[4]) - float(lineals[3])) <= float(10))):
			# print(lineh)
			# print(linea)
			# for i, l in enumerate(linehls):
				# print(str(i)+': '+str(l))
			# for i,l in enumerate(lineals):
				# print(str(i)+': '+str(l))
			# ~ intot.write(lineh[:-2] + ', ' + linea)
'''
