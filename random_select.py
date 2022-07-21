import json

import subprocess
import random
from collections import defaultdict

#a = subprocess.run(["ls", "saved_json_test"]).stdout
a = subprocess.Popen(["ls", "saved_json2"], stdout=subprocess.PIPE)
#output, error = a.communicate()
a_list = a.stdout.readlines()
a_list = [str(l.strip(),'utf-8') for l in a_list]
#print(a_list)
print("length of list is", len(a_list))
#print(output.decode("utf-8"))

fileCount = 15000
curr = fileCount
arrSize = len(a_list)
queryRange = arrSize
for i in range(fileCount):

    randomNum = random.randint(0, queryRange-1)

    a_list[randomNum], a_list[queryRange-1] = a_list[queryRange-1], a_list[randomNum]
    queryRange -= 1

def customRank(x):

    x = x.replace(".json", "")

    xint = int(x)

    return (xint, )

pickedList = a_list[arrSize-fileCount:arrSize]
#pickedList = sorted(pickedList)
pickedList = sorted(pickedList, key = lambda x: customRank(x))

#print(pickedList)
#print(len(pickedList))

myPickedArticles = {}

file = "./myPickedArticles.json"
myPickedArticles["pickedArticles"] = pickedList
json_object = json.dumps(myPickedArticles, indent=4)
with open(file,"w",encoding='utf-8') as currfile:
    currfile.write(json_object)


