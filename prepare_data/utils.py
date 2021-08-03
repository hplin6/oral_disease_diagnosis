import os
import cv2
import numpy as np

def wline(base_path,sub_path):
    full_path=os.path.join(base_path,sub_path)
    path_files=os.listdir(full_path)
    for eachfile in path_files:
        print(eachfile)
        img=cv2.imread(os.path.join(full_path,eachfile),cv2.IMREAD_GRAYSCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        img = cv2.dilate(img, kernel, iterations=1)
        img = 255-img
        print(img)
        cv2.imwrite(test_path+"label2/"+eachfile,img)

def resize(base_path,sub_path):
    full_path=os.path.join(base_path,sub_path)
    path_files=os.listdir(full_path)
    for eachfile in path_files:
        #print(eachfile)
        img=cv2.imread(os.path.join(full_path,eachfile),cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(320,320));
        cv2.imwrite(os.path.join(full_path,eachfile),img)   
      
def get_allfiles(dir):
    if not os.path.exists(dir):
        return []
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_allfiles(sub_path)
    return result       

def list_dir(filepath):
    files=[]
    pathDir=os.listdir(filepath)
    if os.path.isfile(os.path.join(filepath,pathDir[0])):
        return pathDir

    pathDir.sort(key=lambda x: x[:-4].lower(),reverse=True)
    for eachfile in pathDir:
        #file=os.path.join('%s\%s' %(filepath,eachfile))
        #print(eachfile)
        files.append(eachfile)
    return files


def read_pathfiles(curpath):
    paths=list_dir(curpath)
    retfiles=[]
    for path in paths:
        childpath=os.path.join(curpath,path)
        files=list_dir(childpath)
        path_dir={"dir":childpath,"filenames":[]}
        for file in files:
            path_dir["filenames"].append(file)
        retfiles.append(path_dir)
    return retfiles

def remove_files(in_files,extname):
    ret_files=[]
    for file in in_files:
        if extname[1:] not in file:
            ret_files.append(file)
    return ret_files

def read_images(patchpath,filter=None,b_color=True,dim=2):
    path_files=os.listdir(patchpath)
    path_files=remove_files(path_files,"*.json")
    path_files.sort()
    imgs=[]
    files=[]
    print("start read %d images" %(len(path_files)))
    for eachfile in path_files:
        #print(eachfile)
        if filter is not None:
           if filter not in eachfile:
               continue
        filename=os.path.join(patchpath,eachfile)
        if b_color==True:
            mode=cv2.IMREAD_COLOR
        else:
            mode=cv2.IMREAD_GRAYSCALE
        img=cv2.imread(filename,mode)
        if b_color==False and dim==3:
            img=np.expand_dims(img,axis=2)
        imgs.append(img)
        files.append(eachfile)
    return imgs,files


def get_roiimg(src_img,width,height,b_debug):
    assert( width <=src_img.shape[0] and height <=src_img.shape[1])
    center_point=(int(src_img.shape[0]/2-1),int(src_img.shape[1]/2-1))
    half_width=int(width/2)
    half_height=int(height/2)
    rect=(center_point[0]-half_width,center_point[0]+half_width,center_point[1]-half_height,center_point[1]+half_height)
    roi_img=src_img[rect[0]:rect[1],rect[2]:rect[3],:]
    if b_debug:
        cv2.rectangle(src_img, (rect[2],rect[0]), (rect[3], rect[1]), (255, 0, 255), 2)
        roi_img=src_img
    return roi_img

def autofind_wh(img_shape,crop_rate=0.1):
    minsize=min(img_shape[0],img_shape[1])
    width=int(minsize*(1-crop_rate))
    height=width
    return width,height

def read_txtline_to_list(file):
    files=open(file,'r',encoding='UTF-8')
    linecontents=[]        
    for eachline in files:
        linecontents.append(eachline[:-1])
    return linecontents

def cv_imread(file_path,shape=None):
    #stream = open(file_path, "rb")
    #bytes = bytearray(stream.read())
    #numpyarray = np.asarray(bytes, dtype=np.uint8)
    #cv_img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

    if shape!=None:
        npdata=np.fromfile(file_path,dtype=np.uint8)#.reshape((shape[0], shape[1],3))
    else:
        npdata=np.fromfile(file_path,dtype=np.uint8)
    cv_img=cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    return cv_img


def cv_imwrite(filePath,img):
    cv2.imencode('.jpg', img)[1].tofile(filePath) #正确的解决办法

def save_patches(path,filename,img,rects,save_imgsize):
    for i,rect in enumerate(rects):
        roiImg = img[rect[1]:rect[3],rect[0]:rect[2]] #利用numpy中的数组切片设置ROI区域
        file,ext=os.path.splitext(filename)
        roiImg = cv2.resize(roiImg,save_imgsize)
        cv_imwrite(os.path.join(path,file+"_"+str(i)+".jpg"),roiImg)