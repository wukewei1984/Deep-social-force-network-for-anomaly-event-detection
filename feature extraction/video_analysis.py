# -*- coding: utf-8 -*-
#09
import numpy as np
import cv2
import argparse
import os
import math
import matplotlib.pyplot as pyplot
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="0"
Pi=3.14

'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

video_analysis.py includes functionality of calculation and drawing the social force and flow on video 

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''

'''
WRITE_TEXT_ON_IMAGE added text to image that then will be formed to video this is for add "Normal" or "Abnormal" to image

Arguments:
    img - opencv image matrix
    text - required text
'''
parser = argparse.ArgumentParser()
parser.add_argument("--scale", type = float, default = 1.5, help="adjust this according to video to see the results properly")
parser.add_argument("--video_path", type = str, default = "/media/image/325A897F5A894119/devin/Test/RoadAccidents002_x264.avi", help = "the path for video input")
args = parser.parse_args()

def write_text_on_image(img,text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (40, 40), font, 0.1, (255, 255, 255), 2)



'''
DRAW_FLOW this methods draws flow on image

Arguments:
    img - opencv image matrix
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    step - step size of points with flow lines for visualization

Returns:
    vis - image with flow
'''
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype('int')
    scale = args.scale             # default = 1.5
    fx, fy = flow[y, x].T*scale

    accerlation=list(flow)
    #按垂直方向（行顺序）堆叠数组构成一个新的数组
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5) #转化为整数
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #绘制光流路线
    #cv2.polylines(vis, lines, 0, (0, 255, 0),1)
    '''
    绘制粒子网络，但这里即使没有这个绘制网络，依然可以得到一个带网络（不清晰）的视频
    这里的(x1,y1)表示的即是粒子网络中点的坐标
    从循环的次数可以知道这个网络中点的个数
    通过这些点或许可以利用到启发式规则２
    '''
    point_num=0 #记录网格中的总点数
    #用于求中心点坐标
    x=0
    y=0
    net_coordinate=[] #存放网格中点的坐标
    net_acceleration=[] #存放网格点对应的加速度
    for (x1, y1), (x2, y2) in lines:
        #cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        net_acceleration.append(accerlation[y1][x1]) #注意这里
        net_coordinate.append([y1,x1])
        x += x1
        y += y1
        point_num += 1
    '''
        1.每次循环得到网络中一个点的坐标，将这些点的坐标存入一个坐标数组当中　
        2.循环结束后对于每一个数组中的坐标运用启发式规则２得到一个新的值，即 body interaction force
        3.将得到的结果再以流图的形式与原始视频结合进行输出
    '''
    #网格中心点的横纵坐标，用于求解高斯函数的带宽
    center_x=int(x/point_num)
    center_y=int(y/point_num)

    bandwidth_total=0
    g_total=0
    ag_total=0
    #求解带宽，这部分的合理性有待商榷
    for i in range(0,point_num):
        bandwidth_total+=pow((net_coordinate[i][0]-center_x),2)+pow((net_coordinate[i][1]-center_y),2) #两点间距离公式
    bandwidth_square=bandwidth_total/point_num
    #bandwidth_square=bandwidth_square/100
    #bandwidth_square=100
    F_ibc=[]
    #计算启发式规则２
    for i in range(0,point_num):
        for j in range(0,point_num):
            distance_square=pow((net_coordinate[i][0]-net_coordinate[j][0]),2)+pow((net_coordinate[i][1]-net_coordinate[j][1]),2) #两点间距离公式
            #g_ij=(math.exp(-distance_square/bandwidth_square))/(Pi*bandwidth_square)
            g_ij=1
            ag_total+=net_acceleration[j]*g_ij
            g_total+=g_ij
            #cv2.polylines(vis, ag_total/g_total, 0, (0, 255, 0), 1)
        F_ibc.append(ag_total/g_total)

    F_ibc= np.vstack(F_ibc).T.reshape(-1, 2,2)
    #F_ibc=list(F_ibc)
    F_ibc = np.int32(F_ibc)  # 转化为整数
    #im = Image.fromarray(F_ibc)
    cv2.polylines(vis, F_ibc, 0, (0, 255, 0), 1)
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(img, (x1, y1), 1, (0, 0, 255), -1)
    return vis
'''
DRAW_FLOW_WITH_FORCE this methods draws flow and force on image

Arguments:
    img - opencv image matrix
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    force - force matrix with Forcex(Fx, Fy) for every pixel
    step - step size of points with flow and force  lines for visualization

Returns:
    vis - image with flow and force
'''

def draw_flow_with_force(img, flow, force, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    y=y.astype(np.int32)
    x = x.astype(np.int32)
    scale=args.scale            # default = 1.5
    fx, fy = flow[y, x].T*scale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    scale1=args.scale
    fx1, fy1 = force[y, x].T*scale1

    lines1 = np.vstack([x, y, x + fx1, y + fy1]).T.reshape(-1, 2, 2)
    lines1 = np.int32(lines1 + 0.5)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #cv2.polylines(vis, lines, 0, (0, 255, 255),1)
    cv2.polylines(vis, lines1, 0, (0, 0, 255),1)

    # 绘制蓝色的粒子网络
    # for (x1, y1), (x2, y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1 ,(0, 0, 255), -1)

    return vis

'''
overlay_image this methods overlays one image on another using alpha channel

Arguments:
    source - image matrix
    overlay - image matrix that needs to be overlayed
Returns:
    result - overlayed image matrix
'''
def overlay_image(source, overlay):
    h, w, depth = overlay.shape
    result = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            color1 = source[i, j]
            color2 = overlay[i, j]
            alpha = 1 #不同的值使得加在原图像上的内容的显示程度不一样
            new_color = [(1 - alpha) * color1 + alpha * color2[0],
                         (1 - alpha) * color1 + alpha * color2[1],
                         (1 - alpha) * color1 + alpha * color2[2]]
            result[i, j] = new_color
    return result


if __name__ == '__main__':
    import sys

# here and example of video with optical flow
    resize=2
    cam = cv2.VideoCapture(args.video_path)
    ret, prev = cam.read()
    if not ret:
        print ('Cant read file')
    prev = cv2.resize(prev, None, fx=resize, fy=resize)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        ret, img = cam.read()
        try:
            img = cv2.resize(img, None, fx=resize, fy=resize)
        except:
            break
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray,None, 0.5, 5, 15, 3, 5, 1.1, 1)  #cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        cv2.imshow('flow', cv2.cvtColor(draw_flow(gray, flow), cv2.COLOR_BGR2RGB))
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()