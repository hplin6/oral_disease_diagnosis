from utils import list_dir,cv_imread,cv_imwrite,save_patches
import os
import cv2
import augmentation as ag
import random

def create_center_position_patches(input_dir,output_dir,patch_size,save_image_size,neighboring_num,b_rotate):

    files=list_dir(input_dir)
    print("the number of files in %s: %d" %(input_dir, len(files)))
    num_patches=0
    for j,eachfile in enumerate(files):
        filepre,fileext=os.path.splitext(eachfile)
        imgfile=os.path.join(input_dir,eachfile)
        img=cv_imread(imgfile)
        if img.any()==None:
            print("read image %s file failed",imgfile)
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

        height, width=img.shape[0],img.shape[1]
        cX=int(img.shape[1]/2)
        cY=int(img.shape[0]/2)

        rects=ag.get_neighbor_patches(cX,cY,width,height,patch_size=patch_size,arate=neighboring_num)
        #img=ag.rotate_bound_samesize(img,30,shapes["points"])
        save_patches(output_dir,filepre+"_1.jpg",img,rects,save_image_size)
        num_patches+=len(rects)
        if b_rotate:
            ret_img=ag.rotate_bound_largersize(img,-15,(cX,cY))
            height_inc, width_inc=int((ret_img.shape[0]-img.shape[0])/2),int((ret_img.shape[1]-img.shape[1])/2)
            nrects=[(a[0]+width_inc,a[1]+height_inc, a[2]+width_inc, a[3]+height_inc) for a in rects]
            #print(rects,nrects,height_inc, width_inc)
            save_patches(output_dir,filepre+"_2.jpg",ret_img,nrects,save_image_size)
            num_patches+=len(rects)
            ret_img=ag.rotate_bound_largersize(img,15,(cX,cY))
            height_inc, width_inc=int((ret_img.shape[0]-img.shape[0])/2),int((ret_img.shape[1]-img.shape[1])/2)
            nrects=[(a[0]+width_inc,a[1]+height_inc, a[2]+width_inc, a[3]+height_inc) for a in rects]
            save_patches(output_dir,filepre+"_3.jpg",ret_img,nrects,save_image_size)
            num_patches+=len(rects)
        print(num_patches)
    return num_patches

def create_random_position_patches(input_dir,output_dir,save_image_size):

    files=list_dir(input_dir)
    print("the number of files in %s: %d" %(input_dir, len(files)))
    num_patches=0
    for j,eachfile in enumerate(files):

        num_patches+=1
        filepre,fileext=os.path.splitext(eachfile)
        imgfile=os.path.join(input_dir,eachfile)
        img=cv_imread(imgfile)
        if img.any()==None:
            print("read image %s file failed",imgfile)
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

        height, width=img.shape[0],img.shape[1]
        cX=int(img.shape[1]/2)
        cY=int(img.shape[0]/2)
        patch_size=int((width if height>width  else height)*0.8)
        cX=int(patch_size/2)+random.random()*(width-patch_size)
        cY=int(patch_size/2)+random.random()*(height-patch_size)
        rects=ag.get_neighbor_patches(cX,cY,width,height,patch_size=patch_size,arate=1)
        #img=ag.rotate_bound_samesize(img,30,shapes["points"])
        print(len(rects))
        save_patches(output_dir,filepre+"_1.jpg",img,rects,save_image_size)
        num_patches+=len(rects)

    return num_patches

def create_dataset_random_position():
    #create training set using resampling method
    b_train=True

    #create test set using center crop method
    #b_train=False

    #crop_patch_size=1024+256+256
    crop_patch_size=1024+512+512
    save_image_size=(1024,1024)

    if b_train==False:
        b_rotate=False
        neighboring_nums={"normal":1,"ulcer":1,"lowrisk":1,"highrisk":1,"cancer":1}
        dataset_basepath="E:\\dataset_oral\\our_oral\\v4\\test"
        rebuild_basepath="E:\\dataset_oral\\our_oral\\random-position\\test"
    else:
        b_rotate=False
        neighboring_nums={"normal":1,"ulcer":1,"lowrisk":1,"highrisk":1,"cancer":1}
        dataset_basepath="E:\\dataset_oral\\our_oral\\v4\\train"
        rebuild_basepath="E:\\dataset_oral\\our_oral\\random-position\\train"

    sclasses=["normal","cancer","ulcer","highrisk","lowrisk"]
    for sclass in sclasses:
        input_dir=os.path.join(dataset_basepath,sclass)
        output_dir=os.path.join(rebuild_basepath,sclass)
        create_random_position_patches(input_dir,output_dir,save_image_size)
          
def create_dataset_center_position():

    #create training set using resampling method
    b_train=True

    #create test set using center crop method
    #b_train=False

    #crop_patch_size=1024+256+256
    crop_patch_size=1024+512+512
    save_image_size=(1024,1024)

    if b_train==False:
        b_rotate=False
        neighboring_nums={"normal":1,"ulcer":1,"lowrisk":1,"highrisk":1,"cancer":1}
        dataset_basepath="E:\\dataset_oral\\our_oral\\v4\\test"
        rebuild_basepath="E:\\dataset_oral\\our_oral\\center-position\\test"
    else:
        b_rotate=True
        neighboring_nums={"normal":1,"ulcer":3,"lowrisk":3,"highrisk":5,"cancer":15}
        dataset_basepath="E:\\dataset_oral\\our_oral\\v4\\train"
        rebuild_basepath="E:\\dataset_oral\\our_oral\\center-position\\train"

    sclasses=["normal","cancer","ulcer","highrisk","lowrisk"]
    for sclass in sclasses:
        input_dir=os.path.join(dataset_basepath,sclass)
        output_dir=os.path.join(rebuild_basepath,sclass)
        create_center_position_patches(input_dir,output_dir,crop_patch_size,save_image_size,neighboring_nums[sclass],b_rotate)


if __name__ == "__main__":
    create_dataset_center_position()
    #create_dataset_random_position()