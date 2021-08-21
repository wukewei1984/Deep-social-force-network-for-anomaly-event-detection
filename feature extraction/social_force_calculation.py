# -- coding: UTF-8 --

import numpy as np
import cv2
import video_analysis
import os
import skvideo.io
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from numpy import *
import scipy
import matplotlib.pyplot as pyplot

'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

social_force_calculation.py includes functionality of  calculation flow and social force and creating videos with overlayed flow and force

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''


'''
GET_EFFECTIVE_VELOCITY calculates effective velocity of pixel using bilinear interpolation from nearest neighbors

Arguments:
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    
Returns:
    O - flow matrix with effective velocities
'''
def get_effective_velocity(flow):
    sp=flow.shape
    O = np.empty((sp[0], sp[1], sp[2]))
    for r in range(sp[0]):
        for c in range(sp[1]):
            Vx,Vy=bilinear_interpolation(r,c,flow)
            O[r,c,0]=Vx
            O[r,c,1]=Vy
    return O


'''
BILINEAR_INTERPOLATION is doing interpolation from nearest four points

Arguments:
    x - x coordinate of pixel
    y - y coordinage of pixel
    flow - flow matrix

Returns:
    Vx - interpolated x coordinate value of pixel
    Vy - interpolated y coordinate value of pixel
'''
def bilinear_interpolation(x, y, flow):
    #Interpolate (x,y) from values associated with four points.
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    im=x if x-1<0 else x-1
    ip=x if x+1>=flow.shape[0] else x+1
    jm=y if y-1<0 else y-1
    jp=y if y+1>=flow.shape[1] else y+1

    point1x=flow[im,jm,0]
    point1y=flow[im,jm,1]

    point2x = flow[im, jp, 0]
    point2y = flow[im, jp, 1]

    point3x = flow[ip, jm, 0]
    point3y = flow[ip, jm, 1]

    point4x = flow[ip, jp, 0]
    point4y = flow[ip, jp, 1]

    Vx=(point1x*(x-im)*(y-jm)+point2x*(x-im)*(jp-y)+point3x*(ip-x)*(y-jm)+point4x*(ip-x)*(jp-y))/((ip-im)*(jp-jm))
    Vy = (point1y * (x - im) * (y - jm) + point2y * (x - im) * (jp - y) + point3y * (ip - x) * (y - jm) + point4y * (
    ip - x) * (jp - y)) / ((ip - im) * (jp - jm))
    return Vx, Vy

'''
FORCE_CALCULATION calculates optical flow and social force

Arguments:
    videoUrl - url of video
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize - image resize factor

Returns:
    result - force flow matrix with size frame_count x video_image_size_1-2 x video_image_size_2-2x3(rgb)
    forceresult- force matrix with size  frame_count x video_image_size_1-2 x video_image_size_2-2

According to Reference 1 for calculation simplicity 25% of initial size of video is taken so because of that here we include resize parameter to resize image
to get flow we used opencv calcOpticalFlowFarneback method that calculates flow from two adjacent frames using Farneback algorithm
Desired velocity and social force are calculated using Reference 1 page 3 (4) (5)
To created colormap from force first the force value is calculated from (Fx, Fy) components and then this force normalized from range 0-255
to get full range of colors in image(normalization was done frame by frame)
Force picture sizes are smaller by 2 pixels from video image sizes (video_image_size_1-2)  this is done to avoid badly interpolated corner pixels
'''

def force_calculation(videoUrl,tau,Pi,resize):
    cam = cv2.VideoCapture(videoUrl)
    ret, prev = cam.read()
    if not ret:
        print ('Cannot read '+videoUrl)
        cam.release()
        return False, np.array([]),np.array([])
    prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevflow=np.zeros((prev.shape[0],prev.shape[1],2))
    result=np.empty((0,prev.shape[0]-2,prev.shape[1]-2,3))
    forceresult = np.empty((0, prev.shape[0] - 2, prev.shape[1] - 2))
    #fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)  # frames per second
    fps = cam.get(cv2.CAP_PROP_FPS)
    while (cam.isOpened()):
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        Vef=get_effective_velocity(flow)
        Vq=(1-Pi)*Vef+Pi*flow       #desired velosity Vea
        F=tau*(Vq-flow)-(flow-prevflow)/1/fps
        F1=get_Force_from_components_and_normalize_for_image(F)
        imC = cv2.applyColorMap(F1, cv2.COLORMAP_JET)
        result=np.append(result,np.array([imC]),axis=0)
        forceresult = np.append(forceresult, np.array([F1]), axis=0)
        prevflow=flow
        prevgray = gray
    cam.release()
    return True, result,forceresult

'''
GET_FORCE_FROM_COMPONENTS_AND_NORMALIZE_FOR_IMAGE is calculating force from (Fx, Fy) components and normalizing in range 0-255

Arguments:
    Force-force matrix

Returns:
    force matrix
'''
def get_Force_from_components_and_normalize_for_image(Force):
    F=np.sqrt(Force[:,:,0]**2+Force[:,:,1]**2)
    F=F[1:F.shape[0]-1,1:F.shape[1]-1]  #removing corner pixels  these have not good bilinear interpolation velocities
    F*=255.0/np.max(F)#求序列的最值
    return np.round(F).astype(np.uint8)
'''
GET_FORCE_FLOW is calculating force from (Fx, Fy) components, normalizing in range 0-255 and opencv colormaping

Arguments:
    Force-force matrix

Returns:
    imC-opencv colormaped image matrix
'''
def get_Force_Flow(F):
    F1 = get_Force_from_components_and_normalize_for_image(F)
    imC = cv2.applyColorMap(F1, cv2.COLORMAP_JET)#伪色彩函数
    return imC






'''
GET_VIDEO_AND_CREATE_VIDEO_WITH_FORCE_AND_FLOW creates videos with flows and overlayed forces
for every video 2 video is outputing one with optical flow(yellow) and force (red) lines and
the second is colormaped force flow overlayed to actual video

Arguments:
    directory - videos directory
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize - image resize factor

Returns:
    creates videos in the same directory
'''
#获取视频并且创建带有暴力流图的视频

def get_video_and_create_video_with_force_and_flow(directory,tau,Pi,resize):
    # 返回指定的文件夹中包含的文件或文件夹的名字的列表，这里返回的是包含UCF_Crimes中测试视频的名字列表
    # 按照字母的先后顺序排列
    videos = os.listdir(directory)
    videos.sort()
    # 对列表中的每一个元素进行如下操作
    for file in videos:
        print (file)
        fn=file.split(".")[0]#分割一次后取序列为0的项
        fn_ext = file.split(".")[-1]#分割一次后取序列为-1的项
        #获取当前路径下的文件
        cam = cv2.VideoCapture(directory + '/' + file)
        #　read()是按帧读取数据 ret表示是否读取到视频，prev表示截取到的每一帧图片
        ret_1, prev_1 = cam.read()
        ret_2, prev_2 = cam.read() #读取初始两帧计算这两帧的光流
        #当没有读取到视频时释放
        if not ret_1 and not ret_2:
            print ('Cannot read '+file+' continuing to next')
            cam.release()
            continue #跳过当前循环的剩余语句，并继续进行下一次循环
        # 统一帧的大小
        prev_1 = cv2.resize(prev_1, (0, 0), fx=resize, fy=resize)
        prev_2 = cv2.resize(prev_2, (0, 0), fx=resize, fy=resize)

        # 转为灰度图像，并计算最开始两帧的光流
        prevgray_1 = cv2.cvtColor(prev_1, cv2.COLOR_BGR2GRAY)
        prevgray_2 = cv2.cvtColor(prev_2, cv2.COLOR_BGR2GRAY)
        prevflow = cv2.calcOpticalFlowFarneback(prevgray_1, prevgray_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


        fps = cam.get(cv2.CAP_PROP_FPS)#获取当前视频文件的帧率
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        #print(prev_2.shape)

        #writer_flow = cv2.VideoWriter(directory+'/Flow/'+fn + '_flow.avi', fourcc, fps, (prev_2.shape[1], prev_2.shape[0]))
        writer_overlay = cv2.VideoWriter('/media/image/GSP1RMCULXF/00_devin_file/picture/' + fn + '.avi', fourcc,fps, (prev_2.shape[1] - 2, prev_2.shape[0] - 2))
        #writer_overlay = cv2.VideoWriter('/media/image/DevinWZM/0000_dataset_collection/UCF_H3/'+fn + '.avi', fourcc, fps, (prev_2.shape[1]-2, prev_2.shape[0]-2))


        prevgray=prevgray_2
        m=0
        while (cam.isOpened()):
            # if m % 100 == 0:
            #     print(m)
            # m+=1

            # ret表示是否读取到视频，image表示截取到的每一帧图片
            ret, img = cam.read()
            if not ret:
                break#结束while循环

            img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)#计算全局稠密光流


            #三种启发式规则，其中启发式规则１是２、３的基础
            #启发式规则１
            acceleration=flow-prevflow  #启发式规则１，公式５

            # #启发式规则２
            # F_ibc = acceleration
            # for i in range(300):
            #     temp = F_ibc[0][i]
            #     F_ibc[0][i]=0
            #     for j in range(300):
            #         if (i-j>150) or (j-i>150):
            #             g_ij=0
            #         else:g_ij=1
            #         F_ibc[0][i]+=temp*g_ij

            #启发式规则3，主要用于行人类的异常行为判别
            F_ibc=acceleration
            for i in range(300):
                temp_i = F_ibc[0][i]
                F_ibc[0][i] = 0
                for j in range(300):
                    temp_j=F_ibc[0][j]
                    cos_ij_1 = temp_i*temp_j
                    cos_ij_2 = sqrt(pow(temp_i[0],2)+pow(temp_i[1],2))*sqrt(pow(temp_i[0],2)+pow(temp_i[1],2))
                    if cos_ij_2 == 0:
                        cos_ij_2=1
                    w_ij =(1-cos_ij_1/cos_ij_2)/2
                    if (i - j > 150) or (j - i > 150):
                        f_ij = 0
                    else:
                        f_ij = 1
                    F_ibc[0][i] +=w_ij*temp_i * f_ij

            # 计算光流的差值求得”加速度“后，弱化了正常的运动引起的光流变化，相当于起到了一次滤波的作用
            # 对由剧烈变化引起的异常行为起到了较好的作用，如车祸

            #writer_flow.write(video_analysis.draw_flow_with_force(gray, acceleration,F_ibc))

           # writer_flow.write(video_analysis.draw_flow(gray, flow))
            """以下两行只是会生成人眼能看到的东西，具体有啥用不好说"""
            imC = get_Force_Flow(F_ibc)  # 生成伪彩色内容
            writer_overlay.write(video_analysis.overlay_image(gray[1:gray.shape[0]-1, 1:gray.shape[1]-1], imC))

            prevflow = flow
            prevgray = gray
        cam.release()
        #writer_flow.release()
        writer_overlay.release()

'''
GET_VIDEO_AND_CREATE_COLORMAP_VIDEO creates videos force color maps and for
calculation simplicity the size of image is taken of 25% of initial image size Reference 1 page 5

Arguments:
    directory - videos directory
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize=0.25 - image resize factor 25 % of initial image size

Returns:
    creates videos in the same directory
'''
#获取视频并且创建颜色图视频
def get_video_and_create_colormap_video(directory,tau,Pi,resize=0.25):
    videos = os.listdir(directory)
    for file in videos:
        print (file)
        fn=file.split(".")[0]
        fn_ext=file.split(".")[-1]
        cam = cv2.VideoCapture(directory+'/'+file)
        ret, prev = cam.read()
        if not ret:
            cam.release()
            print ('Cant read '+file+' continuing to next')
            continue
        prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prevflow = np.zeros((prev.shape[0], prev.shape[1], 2))
        #fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        fps = cam.get(cv2.CAP_PROP_FPS)
        #fourcc=cv2.cv.CV_FOURCC('I','Y','U','V')
        fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
        writer_flow = cv2.VideoWriter(directory+'/Colormap/'+fn + '_colormap.avi', fourcc, fps, (prev.shape[1]-2, prev.shape[0]-2,))

        i = 0
        while (cam.isOpened()):
            ret, img = cam.read()
            if not ret:
                break
            img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None,0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            Vef = get_effective_velocity(flow)
            Vq = (1 - Pi) * Vef + Pi * flow
            F = tau * (Vq - flow) - (flow - prevflow) * fps
            imC = get_Force_Flow(F)
            writer_flow.write(imC)


          #  pyplot.imshow(imC)  # 对图像进行处理，但是无法显示，需和下一条语句配合使用
           # pyplot.show()  # 显示处理过后的图像

            #pyplot.imshow(imC)  # 对图像进行处理，但是无法显示，需和下一条语句配合使用
            #pyplot.savefig('/home/ty/UCF_Crimes/force_flow2/force_%d' % i)
            if i%100==0:
                print(i)
            i += 1
            prevflow = flow
            prevgray = gray
        cam.release()
        writer_flow.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys

    #make it true if you want to create videos
    createVideos=False
    createOnlyTestVideo=True
    if createVideos:
        #normal crowd videos
        get_video_and_create_video_with_force_and_flow('Normal Crowds', 0.5, 0, 1)
        get_video_and_create_colormap_video('Normal Crowds', 0.5, 0)

        #abnormal crowd videos
        get_video_and_create_video_with_force_and_flow('Abnormal Crowds', 0.5, 0, 1)
        get_video_and_create_colormap_video('Abnormal Crowds', 0.5, 0)

        #crowd dataset extra
        get_video_and_create_video_with_force_and_flow('Crowd Dataset - extra', 0.5, 0, 1)
        get_video_and_create_colormap_video('Crowd Dataset - extra', 0.5, 0)

    if createOnlyTestVideo:
        # test dataset
        #get_video_and_create_video_with_force_and_flow('Test Dataset Crowd', 0.5, 0, 1)
        #get_video_and_create_colormap_video('Test Dataset Crowd', 0.5, 0)

        #获取视频并且创建带有暴力流图的视频
        #get_video_and_create_video_with_force_and_flow('/media/image/DevinWZM/0000_dataset_collection/UCF_Crimes/Videos', 0.5, 0, 1)

        video_folder_path='/media/image/DevinWZM/0000_dataset_collection/Videos/'
        for folder in os.listdir(video_folder_path):
            # for folder in '/media/image/DevinWZM/UCF_Crimes/Videos' :
            print(folder)
            get_video_and_create_video_with_force_and_flow(video_folder_path + folder, 0.5, 0, 1)
            print(folder + ' had been all processed over!!! continue to next folder...')
        print(' everything is done eventually ! please do the next thing...')
        #获取视频并且创建颜色图视频
        # get_video_and_create_colormap_video('/home/ty/UCF_Crimes/Videos/Test', 0.5, 0)


