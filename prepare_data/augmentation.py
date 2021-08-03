import numpy as np
import random as rand
import cv2
import os
import time
from utils import read_txtline_to_list,list_dir,cv_imread,cv_imwrite,save_patches
import math

def get_boundingbox(point,boxsize,imgWidth,imgHeight):
    x_start=0
    y_start=0
    box_width=boxsize
    box_height=boxsize

    if( point[0] +box_width/2>imgWidth):
        x_start=imgWidth-box_width
    elif( point[0]-box_width/2<0):
        x_start=0
    else:
        x_start=point[0]-box_width/2

    if( point[1] +box_height/2>imgHeight):
        y_start=imgHeight-box_height
    elif(point[1]-box_height/2<0):
        y_start=0
    else:
        y_start=point[1]-box_height/2
    
    return int(x_start),int(y_start),int(x_start+box_width),int(y_start+box_height)

def vis_patches(img,rects):
    for rect in rects:
        cv2.rectangle(img, (rect[0],rect[1]),(rect[2],rect[3]),(0, 255, 0),3)
        cv2.circle(img, (int((rect[0]+rect[2])/2), int((rect[1]+rect[3])/2)), 4, (255, 255, 255), 3)

def box2center(points):
    left_top_r = round(points[0][0],2)  # y
    left_top_c = round(points[0][1],2)  # x
    right_bottom_r = round(points[1][0],2)
    right_bottom_c = round(points[1][1],2)
    cX=round((left_top_r+right_bottom_r)/2)
    cY=round((left_top_c+right_bottom_c)/2)
    return cX,cY

def get_neighbor_patches(cX,cY,img_width,img_height,patch_size=512,arate=2,b_random=False):
    rects=[]
    offset=0
    if b_random==False:
        offset=int(patch_size/8)
    else:
        rand.seed(time.time())
        offset=int(rand.random()*patch_size/5)
    if arate==1:
        rects.append(get_boundingbox((cX,cY),patch_size,img_width,img_height))
    elif arate==3:
        rects.append(get_boundingbox((cX-offset,cY-offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX+offset,cY+offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX,cY),patch_size,img_width,img_height))
    elif arate==5:
        rects.append(get_boundingbox((cX-offset,cY-offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX-offset,cY+offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX+offset,cY-offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX+offset,cY+offset),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX,cY),patch_size,img_width,img_height))
    elif arate==15:
        for i in range(14):
            theta=round(360/14)*(1+i)
            x=cX+round(offset*math.cos(math.radians(theta)))
            y=cY+round(offset*math.sin(math.radians(theta)))
            print(i,theta,round(offset*math.cos(math.radians(theta))),round(offset*math.sin(math.radians(theta))))
            rects.append(get_boundingbox((x,y),patch_size,img_width,img_height))
        rects.append(get_boundingbox((cX,cY),patch_size,img_width,img_height))
    else:
        raise AssertionError("the arugment rate is not supported")
    tmp=set(rects)
    return list(tmp)

def get_rotate_patches(cX,cY,img_width,img_height,angle,patch_size=512,arate=2):
    affine_point=rotate_point((cX,cY), (int(img_width/2), int(img_height/2)), -angle, 1.0)
    print(affine_point)
    return get_neighbor_patches(affine_point[0],affine_point[1],img_width,img_height,patch_size,arate)

def create_negative_patches(dataset_basepath,sclass,inside_filename, outside_filename,outpath,patchsize,b_augment=False):
    outimg_size=(1024,1024)
    #outimg_size=(int(patchsize/2),int(patchsize/2))
    inside_files=read_txtline_to_list(os.path.join(dataset_basepath,inside_filename))
    outside_files=read_txtline_to_list(os.path.join(dataset_basepath,outside_filename))
    print("the number of set files:", len(inside_files))
    num_patches=0
    num_fileout=0
    num_filein=0
    files=list_dir(os.path.join(dataset_basepath,sclass))
    print("the number of all images:",len(files))
    for j,eachfile in enumerate(files):
        if eachfile not in inside_files:
            if eachfile in outside_files:
                print(eachfile + " is in other set!" )
            else:
                print("ERROR",eachfile, "is not in the set" )
            num_fileout+=1
            continue

        num_filein+=1
        filepre,fileext=os.path.splitext(eachfile)
        imgfile=os.path.join(dataset_basepath,sclass,eachfile)
        img=cv_imread(imgfile)
        if int(img.shape[0]*100/img.shape[1])==133:
            img=cv2.resize(img,(3000,4000))
        elif int(img.shape[0]*100/img.shape[1])==75:
            img=cv2.resize(img,(4000,3000))
        elif img.shape[0]>img.shape[1]:
            rate=4000/img.shape[0]
            img=cv2.resize(img,(int(img.shape[1]*rate),int(img.shape[0]*rate)))
        elif img.shape[0]<img.shape[1]:
            rate=4000/img.shape[1]
            img=cv2.resize(img,(int(img.shape[1]*rate),int(img.shape[0]*rate)))
        offset=30
        if img.any()==None:
            print("read image %s file failed",imgfile)
        height, width=img.shape[0],img.shape[1]
        cX=int(img.shape[1]/2)
        cY=int(img.shape[0]/2)
        if b_augment==False:
            x1,y1,x2,y2=get_boundingbox((cX,cY),patchsize,img.shape[1],img.shape[0])
            #cv2.rectangle(img, (x1,y1),(x2,y2),(0, 255, 0),2)
            print(cX,cY, x1,y1,x2,y2)
            roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域
            #cv2.imwrite(eachfile,img)
            roiImg = cv2.resize(roiImg,outimg_size)
            cv_imwrite(os.path.join(outpath,filepre+".jpg"),roiImg)
            continue

        for i in range(1):
            rand.seed(time.time())
            rand_x=rand.randint(cX-int(patchsize/5),cX+int(patchsize/5))  
            rand_y=rand.randint(cY-int(patchsize/5),cY+int(patchsize/5)) 
            p=(rand_x,rand_y) 
            x1,y1,x2,y2=get_boundingbox(p,patchsize,img.shape[1],img.shape[0])
            #cv2.rectangle(img, (x1,y1),(x2,y2),(0, 255, 0),2)
            print(cX,cY,p)
            roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域
            #cv2.imwrite(eachfile,img)
            roiImg = cv2.resize(roiImg,outimg_size)
            cv_imwrite(os.path.join(outpath,filepre+"_"+str(i)+".jpg"),roiImg)
            num_patches+=1


        ret=rotate_bound_samesize(img,15,(cX,cY))
        x1,y1,x2,y2=get_boundingbox(p,patchsize,img.shape[1],img.shape[0])
        print(x1,y1,x2,y2)
        roiImg = ret[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域
        roiImg = cv2.resize(roiImg,outimg_size)
        cv_imwrite(os.path.join(outpath,filepre+"_"+str(i+1)+".jpg"),roiImg)
        num_patches+=1
        ret=rotate_bound_samesize(img,-15,(cX,cY))
        x1,y1,x2,y2=get_boundingbox(p,patchsize,img.shape[1],img.shape[0])
        print(x1,y1,x2,y2)
        roiImg = ret[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域
        roiImg = cv2.resize(roiImg,outimg_size)
        cv_imwrite(os.path.join(outpath,filepre+"_"+str(i+2)+".jpg"),roiImg)
        print(j+1,'finshed write:',filepre+"_X.jpg")
        num_patches+=1

    print("the number in set files %d, out set files %d, create files %d" %(num_filein,num_fileout,num_patches))
    return num_patches


def rotate_point(point1, point2, angle, height):
    """
    点point1绕点point2(base point)旋转angle(正：表示逆时针，负：表示顺时针)后的点
    ======================================
    在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
    x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    ======================================
    将图像坐标(x,y)转换到平面坐标(x`,y`)：
    """
    x1, y1 = point1
    x2, y2 = point2
    # transform the image coordinate to the plane coordinate
    y1 = height - y1
    y2 = height - y2
    x = (x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2
    y = (x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2
    # transform the plane coordinate to the image coordinate
    y = height - y
    return (x,y)


def rotate_bound_samesize(image,angle,point):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    #affine_point=rotate_point(point, (int(cX), int(cY)), -angle, 1.0)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
 
    # perform the actual rotation and return the image
    ret=cv2.warpAffine(image, M, (w, h),cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,)
    #cv2.circle(ret, (int(affine_point[0]),int(affine_point[1])), 4, (0, 0, 255), 3)
    return ret

def rotate_bound_largersize(image, angle,point):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    affine_point=rotate_point(point, (int(cX), int(cY)), -angle, h)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    ret=cv2.warpAffine(image, M, (nW, nH),cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,)
    #cv2.circle(ret, (int(affine_point[0]),int(affine_point[1])), 4, (0, 0, 255), 3)
    return ret

if __name__ == "__main__":

    img="sample1.jpg"
    img=cv2.imread(img)
    img=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    point=(int(img.shape[1]/2),int(img.shape[0]/2))
    cv2.circle(img, point, 4, (255, 255, 255), 3)
    cv2.imshow("img",img)
    ret=rotate_bound_samesize(img,-15,point)
    cv2.imshow("ret_samesize",ret)
    ret=rotate_bound_largersize(img,-15,point)
    cv2.imshow("ret_largesize",ret)
    print(img.shape[1],img.shape[0],ret.shape[1],ret.shape[0])
    cv2.waitKey(0)