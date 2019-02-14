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


def detect4():
    with np.load('knn_data_1.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    rois = []
    img = cv.imread("test_num_6.png")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,img_bin = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV|cv.THRESH_TRIANGLE)
    cv.imshow("bin",img_bin)
    contours,hierachy = cv.findContours(img_bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h = cv.boundingRect(con)
        if w > 5 and h > 5 and h>w:
            num = img_bin[y:y+h,x:x+w].copy()
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
    # knn, train, train_labels = knnUpdate(knn, train, train_labels)
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    print(train_labels)
    print(type(train_labels))
    #detect2(knn,train,train_labels)
    detect3()
    #knnDataset1()
    #detect4()
    cv.waitKey()