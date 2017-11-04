import requests
import time
import json
import numpy as np
import random
import matplotlib.pyplot as plt


standards = {"太急": 5.2, "熟练度": 3.4, "侥幸": 1.5, "粗心": 1.5, "违规": 3.5}
baseURL = "http://oldcar-ssl.smartgslb.com/"
#baseURL = 'https://nekoweb.000webhostapp.com/oldcar/'
type5 = ['停车场', '驾驶证', '非法', '号牌', "违规"]
type4 = ['确认', '非机动车', '违反', '车门',"粗心"]
type3 = ['信号灯', '欺骗', "故意", '妨碍', "醉酒", "侥幸"]
type2 = ['缓慢', '时速未达', '缓慢行驶', '悬挂', '黄标', "熟练度"]
type1 = ['超过规定时速', '以上', '超车', "太急"]

type = [type1, type2, type3, type4, type5]


# getUser(Uuid) -> {id, carInfo[{ven, en, license}, moblie]
# serUser(Uuid, {tags:{words:weight}, records :[], sentiments{}}])


def generateJSON(detail):
    ans = []
    for i in range(80):
        randomIndex = random.randint(0, len(detail) - 1)
        code = list(detail.keys())[randomIndex]
        value = detail[code]['chinese']
        a = {
    "code": "200",
    "data": [
        {
            "apiDesc": "违章查询",
            "apiId": 1000,
            "code": "200",
            "message": "success",
            "result": [
                {
                    "province": "HB",
                    "city": "HB_HD",
                    "hphm": "冀DHL327",
                    "hpzl": "02",
                    "lists": [
                        {
                            "date": "2013-12-29 11:57:29",
                            "area": "316省道53KM+200M",
                            "act": code +": " + value,
                            "code": "",
                            "fen": "6",
                            "money": "100",
                            "handled": "0"
                        }
                    ]
                }
            ]
        }
    ],
    "message": "success"
        }
        ans.append(a)
    return ans


def getAllUsers():
    return requests.get(baseURL + "getAllUsers/?timestamp=234567890").json()['data']


def getUser(uid):
    return requests.get(baseURL + "getUser/?id=" + uid).json()['data']


def setUser(uid, detail):
    requests.post(baseURL + "setUser/?id=" + uid,  data={"data": detail})


def getLocation():
    # TODO
    location = requests.get("").json()
    return location


def getInformation(words):
    requireURL = baseURL + "keywords/?src="+words
    return requests.get(requireURL).json()


def loadCode():
    with open('modifyData.txt', 'r') as f:
        t = f.readlines()
        dic = {}
        rec = {}
        for i in t:
            i = i.split('\t')
            dic[i[0]] = {'times': 0, 'location': [], "chinese": i[1]}
            rec[i[0]] = {'times': 0, 'location': []}
    return dic, rec


def fetchInfo(carInfo):
    vin = carInfo['vin']
    engine = carInfo['engine']
    license = carInfo['license']
    headers = {'Accept': 'application/json',
               'Authorization': 'Bearer 5da98295-9bbb-4525-950e-d11a73bb3099'}
    checkUrl = 'https://openapi.saicmotor.com/services/unicdata/' + 'violation/v1.0.0/vehvio?appkey' \
                                                                    '=700031983A55923A57A3F3F5328A12A0&pid=142&cityid=348&' \
                                                                    'licensePlate=' + license + '&vincode' \
                                                                                                '=' + vin + '&engineNumber=' + engine
    #print(checkUrl)
    details = requests.get(checkUrl, headers=headers).json()
    return details




def checkIn(type, str):
    for i in type:
        if i in str:
            return True
        else:
            return False


users = getAllUsers()

for uid in users:
    dic, rec = loadCode()
    user = getUser(uid)
    details = []
    tags = {}
    sentiments = {"太急": -5, "熟练度": -2.5, "侥幸": 1.3, "粗心": .2, "违规": .3}
    for car in user['carinfo']:
        detail = generateJSON(dic)
        print("detail:", detail)
        for d in detail:
            d = d['data'][0]['result'][0]['lists'][0]
           # print(d)
            code = d['act'].split(" ")[0].rstrip(':')
            dic[code]['times'] += 1
            rec[code]['times'] += 1
            dic[code]['location'].append(d['area'])
    keyList = list(dic.keys())
    for key in keyList:
        if dic[key]['times'] < 1:
            del dic[key]
            del rec[key]
    records = rec
    #print(len(records), records)
    for key in dic.keys():
        sentence = dic[key]['chinese']
        for i in type:
            if checkIn(i, sentence):
                #print(sentence)
                sentiments[i[-1]] += 1
        info = getInformation(sentence)
        if info['status']:
            print(info)
            for keyList in info['keywords']:
                if keyList != None:
                    keyword = keyList['keyword']
                    if keyword in tags.keys():
                        tags[keyword] += 1
                    else:
                        tags[keyword] = 1
    tot = 0
    for key in sentiments.keys():

        sentiments[key] /= standards[key]
        tot += sentiments[key]
    tot -= 4
    for key in sentiments.keys():
        sentiments[key] /= tot
        if sentiments[key] < 0:
            sentiments[key] = 0
    data = json.dumps({
        'sentiments': sentiments,
        'tags': tags,
        'records': records
    })
    print(data)
    setUser(uid, data)


# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))


# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

                ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()
