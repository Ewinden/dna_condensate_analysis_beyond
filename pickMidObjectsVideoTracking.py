# %%
import warnings
warnings.simplefilter("ignore")
from sys import exit
import os.path
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from scipy import ndimage
from skimage import measure, io, exposure, morphology
from pathlib import Path
from skimage.feature import peak_local_max
import csv
import math

# %%
'''
outL: box size when identifying molecules
splitter: amount of frames between check-ins to get new molecules
frameRate = frame rate of camera to determine time
'''
outL = int(99)
splitter = 50
frameRate = 10

'''
showimgs: shows all images for testing, set as false for just running
saveimgs: allows user to save images
saveForTest: saves additional files full of images
saveForPeaks: saves additional files for precise peak reading
'''
#showimgs = False
saveimgs = True
#saveForTest = False
#saveForPeaks = False

enter = 13
right = 83
left = 81
esc = 27

wantGain = False
'''
Define pixel size of current camera in nm/pix-el: MD: 114, Andor: 128, Sam: 101.4
'''
pixelSize = 128
refPt = []
threshold_value= 0.95
footprint=np.zeros([3,3])
titles = ['point_number','image_name','x','y','area','perimeter','mean','max_x','max_y','max_val','equivalent_diameter']#,'ellipse_axes_lengths']

# %%
def my_convert_16U_2_8U(image):
    min_ = np.amin(image)
    max_ = np.amax(image)
    a = 255/float(max_-min_)
    b = -a*min_ 
    img8U = np.zeros(image.shape,np.uint8)
    cv2.convertScaleAbs(image,img8U,a,b)
    return img8U

# %%
def getBrightBox(img, point, outLength):
    shap = img.shape
    startx = int(max(0, point[1] - outLength))
    starty = int(max(0, point[0] - outLength))
    endx = int(min(shap[1], point[1] + outLength+1))
    endy = int(min(shap[0], point[0] + outLength+1))
    ml = int(min(0, point[1] - outLength))
    mr = int(max(0, point[1] + outLength + 1 - shap[1]))
    mt = int(min(0, point[0] - outLength))
    mb = int(max(0, point[0] + outLength + 1 - shap[0]))
    miss = [int(mt*-1), int(mb), int(ml*-1), int(mr)]
    brightBox = img[starty:endy,startx:endx]
    avg = brightBox.mean()
    return(brightBox, startx, starty, avg, miss)

# %%
def findBackground(img):
    avgz_fit = np.zeros(img.shape, np.float64)
    len_x = img.shape[0]
    len_y = img.shape[1]
    z = img.flatten()
    z = z.reshape((len(z),1))
    y = np.array([ i for i in range(len_y) ]*len_x)
    x = np.array([ [i]*len_y for i in range(len_x)  ])
    x = x.flatten()
    xx = np.power(x, 2)
    yy = np.power(y, 2)
    xy = x*y
    ones = np.ones(len(z))
    H = np.array([ xx, yy, xy, x, y, ones ])
    H = H.T
    c = np.dot(H.T, H)
    c = np.linalg.inv(c) 
    c = np.dot(c, H.T)
    c = np.dot(c, z)
    z_fit = np.dot(H, c)
    z_fit = z_fit.reshape((len_x, len_y)).astype(np.uint16)
    return(z_fit)

# %%
def describeContour(img,thr,point):
    contours = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    lowdis = -999
    for icont in contours[0]:
        bdis = cv2.pointPolygonTest(icont,point, True)
        if bdis > lowdis:
            lowdis = bdis
            cont = icont
    if lowdis == -999:
        return(False)
    area = cv2.contourArea(cont)
    per = cv2.arcLength(cont, True)
    M = cv2.moments(cont)
    x = (int(M['m10']/M['m00']) if M['m00'] > 0 else 0)
    y = (int(M['m01']/M['m00']) if M['m00'] > 0 else 0)
    mask = np.zeros(img.shape,np.uint8)
    cv2.drawContours(mask, [cont], -1, 255,-1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask)
    mean = cv2.mean(img, mask)[0]
    eqdiam = 2*(math.sqrt(area/math.pi))
    #ellipse = cv2.fitEllipse(cont)
    return([cont,x,y,area,per,mean,max_loc[1],max_loc[0],max_val,eqdiam,mask])

# %%
def clickPoint(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [y,x]

# %%
def onFrame(img):
    global points
    global refPt
    #global stackedPoints
    while True:
        refPt = []
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", clickPoint)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if (k==esc or k==left or k==right or k==enter):
            if k==enter and refPt != []:
                points.append(refPt)
                cv2.circle(img, [refPt[1],refPt[0]], 7, (255,255,255), -1)
                continue
            break
    return(k)

# %%
def setGain(img):
    ratios = [1,1.5,2,4,8,12,16,20,24,100]
    n = 0
    while True:
        r = ratios[n]
        maxm = int(65535/r)
        dimg = img.copy()
        dimg[dimg>maxm]=65535
        if np.amin(dimg) >= maxm:
            break
        eimg = my_convert_16U_2_8U(dimg*r)
        cv2.imshow("image", eimg)
        cv2.setMouseCallback("image", clickPoint)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k==enter or n == len(ratios)-1:
            n+=1
        elif k==esc:
            break
    return(ratios[n],eimg)

# %%
def findObjects(img, thrval):
    mean = np.mean(img)
    saliency = np.power(img - mean,2)
    dx = ndimage.sobel(saliency, 0)  # horizontal derivative
    dy = ndimage.sobel(saliency, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 1 / np.max(mag)  # normalize
    #plow, phigh = np.percentile(mag, (12, 98))
    #remag = exposure.rescale_intensity(mag, in_range=(plow, phigh))
    mag = exposure.equalize_hist(mag)
    binary_mask = mag >= thrval
    clean_mask = ndimage.morphology.binary_fill_holes(binary_mask).astype(np.uint8)
    return(clean_mask,mag.astype(np.uint8),binary_mask.astype(np.uint8))

# %%
def grabObject(img,point,thrval):
    bgd = findBackground(img).astype(np.uint16)
    bgdImg = cv2.subtract(img, bgd)
    gbgdImg = ndimage.filters.gaussian_filter(bgdImg,1)
    brightBox=getBrightBox(img,point,outL)
    gbgdbBox=getBrightBox(gbgdImg,point,outL)
    foundobjsn,edges,foundobjs = findObjects(gbgdbBox[0],thrval)
    cont = describeContour(brightBox[0],foundobjsn,(outL,outL))
    p = [int(cont[2])+brightBox[2],int(cont[1])+brightBox[1]]
    return(cont,p,brightBox[0])

# %%
#wantGain=True

'''Ask for picture location'''
print("\n please choose image!")
dirtouse = askdirectory(title = "Select image")
#dirtouse = 'E:/Downloads/Test/Test'
print(dirtouse)

allpanalysis = pd.DataFrame()
#pass through each "positions folder which contains a set of images with times"
for root, dirs, files in os.walk(dirtouse):
    #Move through all files in directory, saving file names and images if they're tifs
    tifs = []
    imgs = []
    for filename in sorted(files):
        if filename.endswith('.tif'):
            tifs.append(filename)
            pictouse = os.path.join(root,filename)
            img = io.imread(pictouse,-1)
            imgs.append(img)
    length = len(imgs) -1
    if length < 1:
        continue

    #set up analysis 
    allpanalysis = pd.DataFrame(columns=titles)
    stats = pd.DataFrame()
    
    #Pick points from first image in imgs
    img = imgs[0]
    pointcounter = 0
    points = []
    if wantGain == True:
        plow, phigh = np.percentile(img, (12, 98))
        eimg = exposure.rescale_intensity(img, in_range=(plow, phigh))
    else:
        eimg = my_convert_16U_2_8U(img)
    onFrame(eimg)

    #Track and measure points
    for ir, point in enumerate(points):
        panalysis = pd.DataFrame(columns=titles)
        cont, p, brightBox = grabObject(img,point,threshold_value)
        if saveimgs == True:
            cv2.imwrite(os.path.join(root,str(ir)+'im0.png'),brightBox)
            cv2.imwrite(os.path.join(root,str(ir)+'mask0.png'),cont[-1])
        for i in range(1,length+1):
            img = imgs[i]
            imgname = tifs[i][:-4]
            cont, p, brightBox = grabObject(img,p,threshold_value)
            panalysis.loc[len(panalysis)] = [ir, imgname] + [cont[1]] + p + cont[4:-1]
            if saveimgs == True:
                cv2.imwrite(os.path.join(root,str(ir)+'im'+str(i)+'.png'),brightBox)
                cv2.imwrite(os.path.join(root,str(ir)+'mask'+str(i)+'.png'),cont[-1])
        cstats = {}
        cstats['Point']=[ir]
        cstats['X']=[point[0]]
        cstats['Y']=[point[1]]
        cstats['Average Size'] = [panalysis["area"].mean()]
        cstats['Minimum Size'] = [panalysis["area"].min()]
        cstats['Maximum Size'] = [panalysis["area"].max()]
        allpanalysis = pd.concat([allpanalysis,panalysis],ignore_index=True)
        stats = pd.concat([stats,pd.DataFrame.from_dict(cstats)], ignore_index=True)
    stats.to_csv(os.path.join(root,'stats.csv'),index=False)
allpanalysis.to_csv(os.path.join(dirtouse,'analysis'+'.csv'),index=False)
       


# %%



