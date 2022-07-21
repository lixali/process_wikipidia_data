import subprocess
import json
# JSON file
f = open ('myPickedArticles.json', "r")
  
# Reading from file
data = json.loads(f.read())
  
# Iterating through the json
# list
copyTargetfolder = "./saved_json_randomPicked/"
subprocess.run(["mkdir", "-p", copyTargetfolder ])
for articleFiles in data["pickedArticles"]:
    #print(txtfile)
    currfile = articleFiles
    subprocess.run(["cp", "saved_json2/" + articleFiles , copyTargetfolder ])
    #subprocess.run(["cp",  txtfile, "../saved_json_top50/"])
  
# Closing file
f.close()
