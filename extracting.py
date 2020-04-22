#Loading all required libraries 
get_ipython().run_line_magic('pylab', 'inline')
import cv2
import numpy as np 
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt
import statistics



#Setting matplot figure size


plt.rcParams['figure.figsize'] = [15, 8]

def img_(img):
#    img = cv2.imread(img,0)
    # return img

    # img = img_('data/f.jpg')

    # showing image
    # imgplot = plt.imshow(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # for adding border to an image
    img1= cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255])
    img123 = img1.copy()

    # Thresholding the image
    (thresh, th3) = cv2.threshold(img1, 11, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 

    # to flip image pixel values
    th3 = 255-th3


    # imgplot = plt.imshow(cv2.resize(th3, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # initialize kernels for table boundaries detections
    if(th3.shape[0]<1000):
        ver = np.array([[1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1]])
        hor = np.array([[1,1,1,1,1,1]])

    else:
        ver = np.array([[1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1],
                   [1]])
        hor = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])


    # to detect vertical lines of table borders
    img_temp1 = cv2.erode(th3, ver, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, ver, iterations=3)


    # imgplot = plt.imshow(cv2.resize(verticle_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # to detect horizontal lines of table borders
    img_hor = cv2.erode(th3, hor, iterations=3)
    hor_lines_img = cv2.dilate(img_hor, hor, iterations=4)



    # imgplot = plt.imshow(cv2.resize(hor_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # adding horizontal and vertical lines
    hor_ver = cv2.add(hor_lines_img,verticle_lines_img)



    # imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 



    hor_ver = 255-hor_ver



    # imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 




    # subtracting table borders from image
    temp = cv2.subtract(th3,hor_ver)


    # imgplot = plt.imshow(cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 



    temp = 255-temp



    # imgplot = plt.imshow(cv2.resize(temp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 

    #Doing xor operation for erasing table boundaries
    tt = cv2.bitwise_xor(img1,temp)



    # imgplot = plt.imshow(cv2.resize(tt, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    iii = cv2.bitwise_not(tt)

    # imgplot = plt.imshow(cv2.resize(iii, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 

    tt1=iii.copy()


    # imgplot = plt.imshow(cv2.resize(tt1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    #kernel initialization
    ver1 = np.array([[1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1],
                   [1,1]])

    hor1 = np.array([[1,1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1]])


    # In[68]:


    #morphological operation
    temp1 = cv2.erode(tt1, ver1, iterations=1)
    verticle_lines_img1 = cv2.dilate(temp1, ver1, iterations=1)


    # In[69]:


    # imgplot = plt.imshow(cv2.resize(verticle_lines_img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # In[70]:


    temp12 = cv2.erode(tt1, hor1, iterations=1)
    hor_lines_img2 = cv2.dilate(temp12, hor1, iterations=1)


    # In[71]:


    # imgplot = plt.imshow(cv2.resize(hor_lines_img2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # In[72]:


    # doing or operation for detecting only text part and removing rest all
    hor_ver = cv2.add(hor_lines_img2,verticle_lines_img1)


    # In[73]:


    # imgplot = plt.imshow(cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 



    dim1 = (hor_ver.shape[1],hor_ver.shape[0])


    # In[75]:


    dim = (hor_ver.shape[1]*2,hor_ver.shape[0]*2)


    # In[76]:


    # resizing image to its double size to increase the text size
    resized = cv2.resize(hor_ver, dim, interpolation = cv2.INTER_AREA)


    # In[77]:


    #bitwise not operation for fliping the pixel values so as to apply morphological operation such as dilation and erode
    want = cv2.bitwise_not(resized)


    # In[78]:


    # imgplot = plt.imshow(cv2.resize(want, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # In[79]:


    if(want.shape[0]<1000):
        kernel1 = np.array([[1,1,1]])
        kernel2 = np.array([[1,1],
                            [1,1]])
        kernel3 = np.array([[1,0,1],[0,1,0],
                           [1,0,1]])
    else:
        kernel1 = np.array([[1,1,1,1,1,1]])
        kernel2 = np.array([[1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1],
                            [1,1,1,1,1]])


    # In[80]:


    tt1 = cv2.dilate(want,kernel1,iterations=14)


    # In[81]:


    # imgplot = plt.imshow(cv2.resize(tt1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # In[82]:


    # getting image back to its original size
    resized1 = cv2.resize(tt1, dim1, interpolation = cv2.INTER_AREA)


    # In[83]:


    # Find contours for image, which will detect all the boxes
#    im21, contours1, hierarchy1 = cv2.findContours(resized1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours1, hierarchy1 = cv2.findContours(resized1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # In[84]:


    #function to sort contours by its x-axis (top to bottom)
    def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
    	reverse = False
    	i = 0
    
    	# handle if we need to sort in reverse
    	if method == "right-to-left" or method == "bottom-to-top":
    		reverse = True
    
    	# handle if we are sorting against the y-coordinate rather than
    	# the x-coordinate of the bounding box
    	if method == "top-to-bottom" or method == "bottom-to-top":
    		i = 1
    
    	# construct the list of bounding boxes and sort them from top to
    	# bottom
    	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    		key=lambda b:b[1][i], reverse=reverse))
    
    	# return the list of sorted contours and bounding boxes
    	return (cnts, boundingBoxes)


    #sorting contours by calling fuction
    (cnts, boundingBoxes) = sort_contours(contours1, method="top-to-bottom")


    # In[86]:


    #storing value of all bouding box height
    heightlist=[]
    for i in range(len(boundingBoxes)):
        heightlist.append(boundingBoxes[i][3])


    # In[87]:


    #sorting height values
    heightlist.sort()


    # In[88]:


    sportion = int(.5*len(heightlist))


    # In[89]:


    eportion = int(0.05*len(heightlist))


    # In[90]:


    #taking 50% to 95% values of heights and calculate their mean 
    #this will neglect small bounding box which are basically noise 
    try:
        medianheight = statistics.mean(heightlist[-sportion:-eportion])
    except:
        medianheight = statistics.mean(heightlist[-sportion:-2])


    # In[91]:


    #keeping bounding box which are having height more then 70% of the mean height and deleting all those value where 
    # ratio of width to height is less then 0.9
    box =[]
    imag = iii.copy()
    for i in range(len(cnts)):    
        cnt = cnts[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if(h>=.7*medianheight and w/h > 0.9):
            image = cv2.rectangle(imag,(x+4,y-2),(x+w-5,y+h),(0,255,0),1)
            box.append([x,y,w,h])
        # to show image


    # In[92]:


    # imgplot = plt.imshow(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),cmap='gray')
    # 


    # In[93]:


    # cv2.imwrite('imagegen.jpg',image)


    # In[94]:


    #rearranging all the bounding boxes horizontal wise where every box fall on same horizontal line 
    main=[]
    j=0
    l=[]
    for i in range(len(box)):    
        if(i==0):
            l.append(box[i])
            last=box[i]
        else:
            if(box[i][1]<=last[1]+medianheight/2):
                l.append(box[i])
                last=box[i]
                if(i==len(box)-1):
                    main.append(l)
            else:
    #             (l)            
                main.append(l)
                l=[]
                last = box[i]
                l.append(box[i])


    # In[95]:


    #calculating maximum number of box in a particular row
    maxsize=0
    for i in range(len(main)):
        l=len(main[i])
        if(maxsize<=l):
            maxsize=l   


    # In[96]:


    ylist=[]
    for i in range(len(boundingBoxes)):
        ylist.append(boundingBoxes[i][0])


    # In[97]:


    ymax = max(ylist)
    ymin = min(ylist)


    # In[98]:


    ymaxwidth=0
    for i in range(len(boundingBoxes)):
        if(boundingBoxes[i][0]==ymax):
            ymaxwidth=boundingBoxes[i][2]


    # In[99]:


    TotWidth = ymax+ymaxwidth-ymin


    # In[100]:


    width = []
    widthsum=0
    for i in range(len(main)):
        for j in range(len(main[i])):
            widthsum = main[i][j][2]+widthsum

    #     (" Row ",i,"total width",widthsum)
        width.append(widthsum)
        widthsum=0



    # In[101]:


    #removing all the lines which are not the part of the table
    main1=[]
    flag=0
    for i in range(len(main)):
        if(i==0):
            if(width[i]>=(.8*TotWidth) and len(main[i])==1 or width[i]>=(.8*TotWidth) and width[i+1]>=(.8*TotWidth) or len(main[i])==1):
                flag = 1
        else:
            if(len(main[i])==1 and width[i-1]>=.8*TotWidth):
                flag=1

            elif(width[i]>=(.8*TotWidth) and len(main[i])==1):
                 flag=1

            elif(len(main[i-1])==1 and len(main[i])==1 and (width[i]>=(.7*TotWidth) or width[i-1]>=(.8*TotWidth))):
                flag=1


        if(flag==1):
            pass
        else:
            main1.append(main[i])

        flag=0


    # In[102]:


    maxsize1=0
    for i in range(len(main1)):
        l=len(main1[i])
        if(maxsize1<=l):
            maxsize1=l  


    # In[103]:


    #calculating the values of the mid points of the columns 
    midpoint=[]
    for i in range(len(main1)):
        if(len(main1[i])==maxsize1):
    #         (main1[i])
            for j in range(maxsize1):
                midpoint.append(int(main1[i][j][0]+main1[i][j][2]/2))
            break


    # In[104]:


    midpoint=np.array(midpoint)
    midpoint.sort()


    # In[105]:


    final = [[]*maxsize1]*len(main1)


    # In[106]:


    #sorting the boxes left to right
    for i in range(len(main1)):
        for j in range(len(main1[i])):
            min_idx = j        
            for k in range(j+1,len(main1[i])):
                if(main1[i][min_idx][0]>main1[i][k][0]):
                    min_idx = k

            main1[i][j], main1[i][min_idx] = main1[i][min_idx],main1[i][j]


    # In[107]:


    #storing the boxes in their respective columns based upon their distances from mid points  
    finallist = []
    for i in range(len(main1)):
        lis=[ [] for k in range(maxsize1)]
        for j in range(len(main1[i])):
    #         diff=np.zeros[maxsize]
            diff = abs(midpoint-(main1[i][j][0]+main1[i][j][2]/4))
            minvalue = min(diff)
            ind = list(diff).index(minvalue)
    #         (minvalue)
            lis[ind].append(main1[i][j])
    #     ('----------------------------------------------')
        finallist.append(lis)





    # In[109]:


    #extration of the text from the box using pytesseract and storing the values in their respective row and column
    todump=[]
    for i in range(len(finallist)):
        for j in range(len(finallist[i])):
            to_out=''
            if(len(finallist[i][j])==0):
                ('-')
                todump.append(' ')

            else:
                for k in range(len(finallist[i][j])):                
                    y,x,w,h = finallist[i][j][k][0],finallist[i][j][k][1],finallist[i][j][k][2],finallist[i][j][k][3]

                    roi = iii[x:x+h, y+2:y+w]
                    roi1= cv2.copyMakeBorder(roi,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255])
                    img = cv2.resize(roi1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((2, 1), np.uint8)
                    img = cv2.dilate(img, kernel, iterations=1)
                    img = cv2.erode(img, kernel, iterations=2)
                    img = cv2.dilate(img, kernel, iterations=1)



                    out = pytesseract.image_to_string(img)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(img)

                    to_out = to_out +" "+out

                # (to_out)

                todump.append(to_out)
    #             cv2.imshow('image',img)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()

    #creating numpy array
    npdump = np.array(todump)


    #creating dataframe of the array 
    dataframe = pd.DataFrame(npdump.reshape(len(main1),maxsize1))
    # (dataframe)
    return dataframe

