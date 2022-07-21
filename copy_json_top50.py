import subprocess
import json
# JSON file
f = open ('containfile.json', "r")
  
# Reading from file
data = json.loads(f.read())
  
# Iterating through the json
# list
for word in data:
    for txtfile in data[word]:
        #print(txtfile)
        currfile = txtfile
        subprocess.run(["cp", "saved_json/" + txtfile , "saved_json_top50/" ])
        #subprocess.run(["cp",  txtfile, "../saved_json_top50/"])
  
# Closing file
f.close()
