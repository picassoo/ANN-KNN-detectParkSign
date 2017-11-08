from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from sklearn import neighbors
import statistics as mode

def readDirectory(directory):
    imageFiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return imageFiles

def detectShape(img,height):

    area1 = False
    area2 = False
    area3 = False

    middle = int(height/2)+2

    for i in range(0,5):
        for j in range(0,5):
            if(img[j,i,1]<180):
                area1 = True
        for j in range(0, 5):
            if (img[height-j-1, i, 1] < 180):
                area2 = True
        for j in range(0, 5):
            if (img[middle-j, i, 1] < 180):
                area3 = True
    if(area2 and area1):
        return "rectangle"
    elif(area2):
        return "triangle"
    elif(area1):
        return "reTriangle"
    elif(area3):
        return "circle"
    else:
        return "other"

def detectProporties(image):

    shapes = {"rectangle":0,"triangle":1,"reTriangle":2,"circle":3,"other":4}
    properties = np.array([0,0])
    height,width = image.shape[:2]

    #en çok egemen iki renk var.(beyaz dışında)
    #bunlardan biri mavi biride kırmızı
    #parketme-durma işaretlerinde egemen renk diğerlerinde genellikle kırmızı oluyor.
    red = 0
    blue = 0
    for i in range(0,height):
        for j in range(0,width):
            if(image[i,j,1]<200):
                if(image[i,j,0]>image[i,j,2]):
                    blue+=1
                elif(image[i,j,0]<image[i,j,2]):
                    red+=1

    if(red >blue):
        properties[0]=1
    else:
        properties[0]=2

    shape=detectShape(image,height)

    properties[1]=shapes[shape]
    return properties

def readImage(imageFiles,directory,mode):
    labels = {"trafikisaretleri/egitim/parketme-durma": 0,
              "trafikisaretleri/egitim/tanzim": 1,
              "trafikisaretleri/egitim/tehlike-uyari": 2}
    labelTestDict = {"trafikisaretleri/test/parketme-durma": 0,
                     "trafikisaretleri/test/tanzim": 1,
                     "trafikisaretleri/test/tehlike-uyari": 2}

    properties = list()
    label = list()
    for i in imageFiles:
        path = directory +"/"+i
        img = cv2.imread(path)
        property = detectProporties(img)
        properties.extend(property)
        if(mode==0):
            label.extend([labels[directory]])
        else:
            label.extend([labelTestDict[directory]])


    return label,properties
def Train(label,train,directories,mode):
    for directory in directories:
        imgFiles = readDirectory(directory)
        labels, trains= readImage(imgFiles,directory,mode)
        label.extend(labels)
        train.extend(trains)

    train = np.array(train)
    train = np.reshape(train,(-1,2))
    label = np.array(label)

    return label,train
def Test(afterTest):
    allK = []
    for i in test:
        dist = np.argsort(np.sum((i - train) ** 2, axis=1))
        nnc = label[dist[:k]]
        afterTest.extend([mode.mode(nnc)])

        nnc1 = mode.mode(label[dist[:1]])
        nnc2 = mode.mode(label[dist[:3]])
        nnc3 = mode.mode(label[dist[:5]])
        allK.extend([mode.mode([nnc1,nnc2,nnc3])])
        #print(nnc1," ",nnc2," ",nnc3,mode.mode([nnc1,nnc2,nnc3]))

    afterTest=np.array(afterTest)
    return afterTest,allK
def Confussion(labelTest,afterTest,list):
    confussion = np.zeros([3, 3])
    confussion = np.reshape(confussion, (3, 3))

    for i in range(0, len(afterTest)):
        confussion[afterTest[i], labelTest[i]] += 1

    target = 0
    all = 0
    for i in range(0,len(list)):
        a=""
        for j in range(0,len(list)):
            a +=str(confussion[list[i],list[j]])+" "
            all +=confussion[list[i],list[j]]
        print(a)
        target += confussion[list[i],list[i]]
    print(k)
    print("Accuracy ", target /all)

if __name__ == "__main__":
    label = list()
    train = list()
    test = list()
    labelTest = list()
    afterTest = list()

    directories = ["trafikisaretleri/egitim/parketme-durma",
                   "trafikisaretleri/egitim/tanzim",
                   "trafikisaretleri/egitim/tehlike-uyari"]

    directoriesTest = ["trafikisaretleri/test/parketme-durma",
                   "trafikisaretleri/test/tanzim",
                   "trafikisaretleri/test/tehlike-uyari"]


    print("0\tTehlike-Uyari ve Trafik Tanzim Isaretlerie")
    print("1\tTehlike-Uyari ve Durma-Parketme Isaretleri")
    print("2\tTum Gruplar")

    select = int(input("Secenek:"))

    if(select==0):
        listed=[1,2]
    elif(select==1):
        listed = [0, 2]
    elif(select==2):
        listed = [0,1, 2]
    else:
        exit()
    label,train = Train(label,train,directories,0)

    labelTest, test = Train(labelTest, test, directoriesTest,1)


    k = int(input("K:"))

    afterTest,allKvalue = Test(afterTest)
    Confussion(labelTest,afterTest,listed)
    print("----1-3-5-----")
    Confussion(labelTest,allKvalue,listed)


    if(k==1):
        k1 = neighbors.KNeighborsClassifier(k)
        k1.fit(train, label)
        predict = k1.predict(test)
        print("--sklearn---")
        Confussion(labelTest, predict, listed)
    elif (k == 3):
        k3 = neighbors.KNeighborsClassifier(k)
        k3.fit(train, label)
        predict = k3.predict(test)
        print("--sklearn---")
        Confussion(labelTest, predict, listed)
    elif (k == 5):
        k1 = neighbors.KNeighborsClassifier(k)
        k1.fit(train, label)
        predict = k1.predict(test)
        print("--sklearn---")
        Confussion(labelTest, predict, listed)