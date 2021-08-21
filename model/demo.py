#coding:utf-8
'''

import os
import shutil
import numpy as np
import pandas as pd
import time

利用“Temporal_Anomaly_Annotation.txt”进行异常和正常行为的分类
异常来源于文档中有标记的部分，正常来源于文档中没有标记的异常视频中的部分
'''
def video_to_two_types():
    anomaly_Annotation_txt_dir = '/media/image/abnormal_behaviour/0000_dataset_collection/UCF_Crimes/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt'

    anomaly_Annotation = []
    with open(anomaly_Annotation_txt_dir,"r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str1 in data:
                sub_str = str1.split()
            if sub_str:
                anomaly_Annotation.append(sub_str)

    anomaly_Annotation = np.array(anomaly_Annotation)
    '''
    anomaly_Annotation每一行各个值的含义: 视频编号　视频类别　第一段异常起始帧　第一段异常结束帧　第二段异常起始帧　第二段异常结束帧
    '''
    numOfVideos = len(anomaly_Annotation)
    srcAbnormalVideos = '/media/image/abnormal_behaviour/0000_dataset_collection/UCF_Crimes_Changes/UCF_Crimes/Videos'
    desAbnormal = '/media/image/abnormal_behaviour/0000_dataset_collection/UCF_Crimes_binary/abnormal' #存放异常视频的异常片段
    desNormal = '/media/image/abnormal_behaviour/0000_dataset_collection/UCF_Crimes_binary/normal'     #存放异常视频的正常片段

    for i in range(numOfVideos):
        if(anomaly_Annotation[i][1]!='Normal'): #只对异常视频进行处理

            start_time = time.clock()

            test = anomaly_Annotation[i][0].split('.')
            test = np.array(test)

            src = srcAbnormalVideos + '/' + anomaly_Annotation[i][1] + '/' + test[0]
            pic_list = os.listdir(src)
            pic_list.sort()
            print(anomaly_Annotation[i][0] + " has " + str(len(pic_list)) + " frames")

            #创建存储异常片段的文件夹
            des_Abnormal_dir = desAbnormal + '/' + test[0]
            isAbnormalExists = os.path.exists(des_Abnormal_dir)
            if not isAbnormalExists:
                os.mkdir(des_Abnormal_dir)

            #创建存储正常片段的文件夹
            des_Normal_dir = desNormal + '/' + test[0]
            isNormalExists = os.path.exists(des_Normal_dir)
            if not isNormalExists:
                os.mkdir(des_Normal_dir)



            '''
            将有异常时间标记的视频的片段放入对应的文件夹中
            异常帧放到abnormal，正常帧放到normal
            这样可以保证分类器训练出来的网络对特定的异常敏感
            '''
            if(anomaly_Annotation[i][2]!=-1): #第一段异常存在
                for j in range(1,int(anomaly_Annotation[i][2])): #这部分是正常视频
                    shutil.copy(src + '/' + pic_list[j-1], des_Normal_dir)
                for j in range(int(anomaly_Annotation[i][2]), int(anomaly_Annotation[i][3])+1): #这部分是异常视频
                    shutil.copy(src + '/' + pic_list[j-1], des_Abnormal_dir)

                if(anomaly_Annotation[i][4] != -1): #第二段异常存在
                    for j in range(int(anomaly_Annotation[i][3])+1, int(anomaly_Annotation[i][4])):  # 这部分是正常视频
                        shutil.copy(src + '/' + pic_list[j-1], des_Normal_dir)
                    for j in range(int(anomaly_Annotation[i][4]), int(anomaly_Annotation[i][5])+1):  # 这部分是异常视频
                        shutil.copy(src + '/' + pic_list[j-1], des_Abnormal_dir)
                else:#第二段异常不存在
                    for j in range(int(anomaly_Annotation[i][3])+1, int(anomaly_Annotation[i][5])+1):  # 这部分是正常视频
                        shutil.copy(src + '/' + pic_list[j-1], des_Normal_dir)

            print(anomaly_Annotation[i][0] + " is done")
            print("Running time: %s Seconds" %(time.clock()-start_time))
            print(" ")
    print("all are done, please do next things")

'''
将一个完整的视频帧按连续的16帧一组进行打包
1、打包之前应创建对应的文件夹，命名为“视频名_序号”，如Abuse028_x264_1
2、因为是要检测异常得分，因此打包结束后，所有的包都视为异常，训练时异常为0，正常为1，因此所有的视频都标记为1
'''
def video_to_16_frams_bags():
    directory = '/media/image/abnormal_behaviour/00c00_dataset_collection/ShanghaiTechDataset/Testing/frames_part2'
    des_dir = '/media/image/abnormal_behaviour/0000_dataset_collection/ShanghaiTechTest'
    videos = os.listdir(directory)
    videos.sort()

    for video_name in videos:
        print(video_name + " is begining...")
        src_dir = directory + '/' +video_name
        des_dir_videos = des_dir + '/' + video_name

        #判断目录是否存在，如果没有则创建一个
        isExists = os.path.exists(des_dir_videos)
        if not isExists:
            os.mkdir(des_dir_videos)

        frame_list = os.listdir(src_dir)
        frame_list.sort()
        frame_nums = len(frame_list)
        frame_list = np.array(frame_list)
        for i in range(1,frame_nums-15):
            #创建文件夹
            test_bags = des_dir_videos + '/' + str(i)
            os.mkdir(test_bags)

            #放入对应的视频帧，窗口的步数为１
            target = test_bags
            for j in range(16):
                source = src_dir + '/'+ frame_list[i+j]
                shutil.copy(source,target)

        print(video_name + " is done")
    print("all are done, please do next things")



video_to_16_frams_bags()