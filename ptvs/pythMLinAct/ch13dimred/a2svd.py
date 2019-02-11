#!/usr/bin/env python3
# a2svd.py
""" 降维技术-SVD
https://blog.csdn.net/qq_36523839/article/details/82347332
"""
from numpy import *
from numpy import linalg as la

class a2svd(object):
    """description of class"""
    def __init__(self):pass


'''
使用sklearn中SVD处理方法
'''
def demo01():
    U,Sigma,VT=linalg.svd([[3, 1],[7, 2]])
    print("U", U)
    print("Sigma", Sigma)
    print("VT:", VT)



'''------------- 利用Python实现SVD -------------
'''
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def demo02():
    Data = loadExData()
    U,Sigma,VT=linalg.svd(Data)
    print("U", U)
    print("Sigma", Sigma)
    print("VT:", VT)
    #重构一个原始矩阵的近似矩阵
    Sig3=mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    arr=U[:,:3] * Sig3 * VT[:3,:]
    print("arr:", arr)


'''------------- 基于协同过滤的推荐引擎 -------------
'''
#欧几里德距离, 欧氏距离转换为2范数计算
def ecludSim(inA,inB): #0最大相似，1最小相似? 反了
    return 1.0/(1.0 + la.norm(inA - inB))

# 皮尔逊相关系数 
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    # 使用0.5+0.5*x 将-1，1 转为 0，1
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

# 余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def demo03():
    myMat=mat(loadExData())
    #相似度计算
    print(ecludSim(myMat[:,0],myMat[:,4]))
    print(ecludSim(myMat[:,0],myMat[:,0]))
    print(cosSim(myMat[:,0],myMat[:,4]))
    print(cosSim(myMat[:,0],myMat[:,0]))
    print(pearsSim(myMat[:,0],myMat[:,4]))
    print(pearsSim(myMat[:,0],myMat[:,0]))


'''------------- 示例：餐馆菜肴推荐引擎 -------------
'''
#对物品评分  (数据集 用户行号 计算误差函数 推荐商品列号)
#未评级物品的评分预测函数
#@dataMat：数据矩阵
#@user：目标用户编号(矩阵索引，从0开始)
#@simMeans：相似度计算方法
#@item：未评分物品编号(索引，从0开始)
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0  # 两个计算估计评分值变量初始化
    for j in range(n):
        userRating = dataMat[user,j] #获得此人对该物品的评分
        if userRating == 0: continue #若此人未评价过该商品则不做下面处理
        #获得相比较的两列同时都不为0的数据行号
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        # 求两列的相似度
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item, j, similarity) )
        simTotal += similarity # 计算总的相似度
        ratSimTotal += similarity * userRating # 不仅仅使用相似度，而是将评分权值*相似度 = 贡献度
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal # 归一化评分 使其处于0-5（评级）之间

# 替代上面的standEst(功能) 该函数用SVD降维后的矩阵来计算评分
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity) )
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

# 给出推荐商品评分， 默认余弦距离
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items 
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def demo04():
    myMat=mat(loadExData())
    myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4 #将数据某些值替换，增加效果
    myMat[3,3] = 2
    result1 = recommend(myMat, 2)  #余弦相似度
    print(result1)
    result2 = recommend(myMat,2,simMeas=ecludSim) #欧氏距离
    print(result2)
    result3 = recommend(myMat,2,simMeas=pearsSim) #皮尔逊相关度
    print(result3)
    # SVD测试
    myMat = mat(loadExData2())
    result1 = recommend(myMat,1,estMethod=svdEst)   # 需要传参改变默认函数
    print(result1)
    result2 = recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
    print(result2)


'''------------- 基于SVD的图像压缩 -------------
关于如何将SVD应用于图像压缩的例子.
一张手写的数字图像，使用SVD来对数据降维，从而实现图像的压缩。
'''

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end='')
            else: print(0, end='')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    fr = open(r'./ch13dimred/data/0_5.txt')
    for line in fr.readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    #SVD计算
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)    

def demo05():
    imgCompress(2)


if __name__ == '__main__':
	#
    #demo01()
	#
    #demo02()
	#相似度计算比较
    #demo03()
	#
    demo04()
    #
    demo05()
