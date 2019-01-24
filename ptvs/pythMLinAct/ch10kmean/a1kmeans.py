#!/usr/bin/env python3
# a1kmeans.py
""" K-均值聚类算法
利用K-均值聚类算法对未标注数据分组
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


''' -------------- K-均值聚类算法及函数 --------------
'''
# 读取文件生成列表
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#初始化聚类中心
def randCent(dataSet, k):
    n = shape(dataSet)[1] #获取列数，数据样本的维度
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) #得到该列数据的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ) #得到该列数据的范围(最大值-最小值)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) #范围内获取k个随机值
    return centroids

def demo01():
    daMat = mat(loadDataSet(r"./ch10kmean/data/testSet.txt"))
    centr = randCent(daMat, 5) #取5个随机点
    dist  = distEclud(daMat[0], daMat[1])
    print("dist=", dist)
    x = array(daMat[:,0]); y = array(daMat[:,1])
    x1= array(centr[:,0]); y1= array(centr[:,1])
    plt.scatter(x, y, c='g',marker='x')
    plt.scatter(x1,y1,c='r',marker='o')
    #plt.plot(daMat, 'gx')
    #plt.plot(centr, 'ro')
    plt.show()

# K-均值算法：创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
#@dataSet:聚类数据集
#@k:用户指定的k个类
#@distMeas:距离计算方法，默认欧氏距离distEclud()
#@createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged: #反复迭代，直到所有数据点的簇分配结果不再改变为止。
        clusterChanged = False
        #遍历数据集每一个样本向量
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1 #初始化最小距离最正无穷；最小距离对应索引为-1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])#计算数据点到质心的欧氏距离
                if distJI < minDist: #如果是最小距离，记录距离和索引
                    minDist = distJI; minIndex = j
            #当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #记录当前变化样本的聚类结果和平方误差
        #print(centroids)
        #遍历每一个质心
        for cent in range(k):#recalculate centroids
            #将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            #计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

def demo02():
    datMat = mat(loadDataSet(r"./ch10kmean/data/testSet.txt"))
    myCentroids, clustAssing = kMeans(datMat,5) #k由4换成5试试
    x = array(datMat[:,0]); y = array(datMat[:,1])
    x1= array(myCentroids[:,0]); y1= array(myCentroids[:,1])
    plt.scatter(x, y, c='g',marker='x')
    plt.scatter(x1,y1,c='r',marker='o')
    plt.show()


''' -------------- 二分K-均值算法 --------------
为克服K-均值算法收敛于局部最小值的问题,二分K-均值的聚类效果要好于K-均值算法。
'''
#二分K-均值聚类算法
#@dataSet:待聚类数据集
#@k：用户指定的聚类个数
#@distMeas:用户指定的距离计算方法，默认为欧式距离计算
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss) )
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

def demo03():
    datMat = mat(loadDataSet(r"./ch10kmean/data/testSet2.txt"))
    centList, myNewAssments=biKmeans(datMat, 3)
    x = array(datMat[:,0]); y = array(datMat[:,1])
    x1= array(centList[:,0]); y1= array(centList[:,1])
    plt.scatter(x, y, c='g',marker='x')
    plt.scatter(x1,y1,c='r',marker='o')
    plt.show()


''' -------------- 示例：对地图上的点进行聚类 --------------
'''
#球面距离计算及鑛绘图函数
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

#对地理坐标进行聚类 
def clusterClubs(numClust=5):
    datList = []
    for line in open(r'./ch10kmean/data/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread(r'./ch10kmean/data/Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    #a = a1trgui()
    #demo01()
    # k-means计算
    demo02()
    #二分K-均值聚类
    demo03()
    #示例
    clusterClubs(5)
