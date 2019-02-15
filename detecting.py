import numpy as np
import cv2 as cv
import os


def knnUpdate(train,train_labels,newData=None,newDataLabel=None):
    newData = newData.reshape(-1, 400).astype(np.float32)
    train = np.vstack((train, newData))
    train_labels = np.hstack((train_labels, newDataLabel))
    # train_labels.append(newDataLabel)
    np.savez('knn_data_1.npz', train=train, train_labels=train_labels)
    print(train.shape, train_labels.shape)

def detect0(knn):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    test = cv.imread('6.jpg',0)
    ret,test_bin = cv.threshold(test,0,255,cv.THRESH_BINARY_INV|cv.THRESH_TRIANGLE)
    th = cv.resize(test_bin,(20,20))
    cv.imshow("th",th)
    th = th.reshape(-1,400).astype(np.float32)
    ret_n,result,neighbour,dist = knn.findNearest(th,k=5)
    print(int(result[0][0]))

def detect2(knn,train,train_labels):
    rois = []
    img = cv.imread("test_num_2.jpg")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,img_bin = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_TRIANGLE)
    cv.imshow("bin",img_bin)
    contours,hierachy = cv.findContours(img_bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h = cv.boundingRect(con)
        if w > 10 and h > 10 and h>w:
            add_h = int(0.2*h)
            edge_w = np.zeros((int((add_h+h-w)/2),h),dtype=np.uint8)

            num = img_bin[y:y+h,x:x+w].copy()
            num = np.insert(num,-1,edge_w,axis=1)
            num = np.insert(num, 0, edge_w, axis=1)
            dh,dw = num.shape
            edge_h = np.ones((add_h,dw),dtype=np.uint8)
            num = np.insert(num, 0, edge_h, axis=0)
            num = np.insert(num, -1, edge_h, axis=0)
            num = cv.resize(num,(20,20))
            rois.append(num)
            th = num.reshape(-1,400).astype(np.float32)
            ret_n,result,neighbour,distance = knn.findNearest(th,k=5)

            cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv.putText(img,str(int(result[0][0])),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,00),1)
    cv.imshow("test",img)
    c = cv.waitKey()
    if c == ord('u'):
        print("--- update knn ---")
        for r in rois:
            cv.imshow('num',r)
            c2 = cv.waitKey()
            if c2 >= 48 and c2<=57:
                data = r.reshape(-1,400).astype(np.float32)
                knnUpdate(train, train_labels, data, c2-48)
            cv.destroyWindow('num')



def detect3():
    with np.load('knn_data_1.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    rois = []
    capture = cv.VideoCapture(0)
    while True:
        c = cv.waitKey(20)
        if c == 27:
            break

        ret1, img = capture.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray2 = cv.dilate(gray,kernel)
        gray2 = cv.erode(gray2,kernel)
        edges = cv.absdiff(gray,gray2)
        # 运用Sobels算子去噪点
        x = cv.Sobel(edges, cv.CV_16S, 1, 0)
        y = cv.Sobel(edges, cv.CV_16S, 0, 1)
        # convertScaleAbs()函数将其转回原来的uint8形式，否则将无法显示图像
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        # Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        ret, img_bin = cv.threshold(dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        cv.imshow("edges", edges)
        cv.imshow("bin",img_bin)
        contours, hierachy = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for con in contours:
            x, y, w, h = cv.boundingRect(con)
            if w > 5 and h > 5 and h > w:
                num = img_bin[y:y + h, x:x + w].copy()
                num = cv.resize(num, (20, 20))
                rois.append(num)
                th = num.reshape(-1, 400).astype(np.float32)
                ret_n, result, neighbour, distance = knn.findNearest(th, k=5)

                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv.putText(img, str(int(result[0][0])), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 00), 1)

        cv.imshow("test", img)


class NumberInfo:
    def __init__(self,x,y,w,h,detect,level=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.detect = detect
        self.level = level



number_infos = []

img_test = cv.imread("test_num_8.png")
def detect4():
    with np.load('knn_data_1.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    rois = []

    gray = cv.cvtColor(img_test,cv.COLOR_BGR2GRAY)
    ret,img_bin = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_TRIANGLE)
    cv.imshow("bin",img_bin)
    contours,hierachy = cv.findContours(img_bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for con in contours:
        x,y,w,h = cv.boundingRect(con)
        if (w > 15 or h > 15) and (2*h > w):
            num = img_bin[y:y+h,x:x+w].copy()
            num = cv.resize(num,(20,20))
            rois.append(num)
            th = num.reshape(-1,400).astype(np.float32)
            ret_n,result,neighbour,distance = knn.findNearest(th,k=5)

            # cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            # cv.putText(img,str(int(result[0][0])),(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,00),1)
            number_infos.append(NumberInfo(x,y,w,h,int(result[0][0])))
    #cv.imshow("test",img)
    c = cv.waitKey()
    if c == ord('u'):
        print("--- update knn ---")
        for r in rois:
            cv.imshow('num',r)
            c2 = cv.waitKey()
            if c2 >= 48 and c2<=57:
                data = r.reshape(-1,400).astype(np.float32)
                knnUpdate(train, train_labels, data, c2-48)
            cv.destroyWindow('num')

def sortNumbers():
    detect4()
    h_ava = 0       # average of h
    w_ava = 0
    ys = []
    lines = []      # 每一行的数字
    y_min = number_infos[0].y
    for nb in number_infos:
        ys.append(nb.y)
        h_ava += nb.h
        w_ava += nb.w
        if y_min >= nb.y:
            y_min = nb.y
    h_ava = h_ava/len(number_infos)
    w_ava = w_ava / len(number_infos)
    line_num = 1
    line_y = ys[0]
    next_line = []
    next_line.append(ys[0])
    for y in ys:
        if np.abs(y-line_y) > int(0.8*h_ava):
            line_y = y
            line_num += 1
            next_line.append(y)
    for i in range(line_num):
        line = []
        for nb in number_infos:
            if np.abs(nb.y-next_line[i]) > int(0.8*h_ava):    #放入一行
                line.append(nb)
        for i in range(len(line)-1):        #调整前后顺序
            for j in range(i,len(line)):
                if line[i].x > line[j].x:
                    tmp = line[j]
                    line[j] = line[i]
                    line[i] = tmp
        lines.append(line)
        print(len(line),line[0].detect)
    new_lines = []
    print(len(lines))
    print(lines)
    for line in lines:
        new_line = []
        if len(line) > 1:       # 不是除法符号
            cur = 0     #当前位置
            for n in range(0,len(line)-1):
                if n < cur:
                    continue
                for m in range(n+1,len(line)):
                    print("n=",n)
                    if line[n].x + line[n].w + int(1.3*w_ava) >= line[m].x:  # 认为是一个多位数
                        line[n].detect = 10 * line[n].detect + line[m].detect
                        line[n].w +=  (line[m].x - line[n].x - line[n].w) + line[m].w
                        if line[m].h > line[n].h:
                            line[n].h = line[m].h
                        if line[m].y < line[n].y:
                            line[n].y = line[m].y
                    else:
                        cur = m
                        print("break, cur=",cur)
                        break
                    if m == len(line)-1:
                        cur = m
                print("detect=",line[n].detect)
                new_line.append(line[n])
                cv.rectangle(img_test, (line[n].x, line[n].y), (line[n].x + line[n].w, line[n].y + line[n].h), (0, 0, 255), 1)
                cv.putText(img_test, str(line[n].detect), (line[n].x, line[n].y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 00), 1)
        new_lines.append(new_line)
    print("num of lines: ",line_num)
    cv.imshow("img",img_test)




def knnDataset1():
    numbers = cv.imread("digits.png", 0)
    cells = [np.hsplit(row, 100) for row in np.vsplit(numbers, 50)]
    cv.imshow("t1",cells[3][10])
    #cv.imshow("test",cells[10][10])
    opt = []
    for cell in cells:
        opt1 = []
        for c in cell:
            contours,hie = cv.findContours(c,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv.boundingRect(contours[0])
            num = c[y:y+h,x:x+w].copy()
            num = cv.resize(num,(20,20))
            opt1.append(num)
        opt.append(opt1)
    train = np.array(opt).reshape(-1, 400).astype(np.float32)
    k = np.arange(10)
    train_labels = np.repeat(k, 500)
    np.savez('knn_data_1.npz', train=train, train_labels=train_labels)

if __name__ == '__main__':
    train=None
    train_labels=None
    if os.path.exists('knn_data.npz') == True:
        with np.load('knn_data.npz') as data:
            train = data['train']
            train_labels = data['train_labels']
    else:
        numbers = cv.imread("digits.png", 0)
        cells = [np.hsplit(row, 100) for row in np.vsplit(numbers, 50)]
        train = np.array(cells).reshape(-1, 400).astype(np.float32)
        k = np.arange(10)
        train_labels = np.repeat(k, 500)
        np.savez('knn_data.npz', train=train, train_labels=train_labels)

    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    #detect2(knn,train,train_labels)
    #detect3()
    #knnDataset1()
    #detect4()
    sortNumbers()
    cv.waitKey()