from flask import Flask
from flask import request,render_template,send_file
import json
from predict import *
import dataset
import train
import os
from dataset import *
from train import *
import subprocess

#App URL=https://pacific-bayou-35154.herokuapp.com

Root="Predict"
SampleRoot="Samples"
predict_filename="predfile.txt"
app=Flask(__name__)
@app.route('/predict', methods = ['POST'])
def Predict():
    print (request.is_json)
    readings=request.get_json()['Data']
    f=open(Root+os.sep+predict_filename,"w")
    f.write(readings)
    f.close()
    process=subprocess.Popen(['python','predict.py','-f=predfile.txt'],stdout=subprocess.PIPE)
    predicted_word,err=process.communicate()
    print(predicted_word)
    return predicted_word
@app.route('/train', methods = ['POST'])
def GatherData():
    print (request.is_json)
    readings=request.get_json()['Data']
    reading_fileName=request.get_json()['FileName']
    f=open(SampleRoot+os.sep+reading_fileName+".txt","w")
    f.write(readings)
    f.close()
    return "RECEIVED "+reading_fileName

@app.route('/trainModel', methods = ['POST'])
def TrainModel():
    print (request.is_json)
    choice=request.get_json()['TrainModel']
    if(choice=="YES"):
    	createDataset()
    	os.system("python train.py > modelaccuracy.txt")
    	f=open("modelaccuracy.txt","r")
    	for line in f.readlines():
    		if("SCORE" in line):
    			f.close()
    			return line
    return "ERROR"

@app.route('/clearDataset', methods = ['POST'])
def ClearDataset():
    print (request.is_json)
    choice=request.get_json()['Clear']
    if(choice=="YES"):
    	print("Deleting Dataset")
    	ctr=0
    	for path,subdir,files in os.walk(SampleRoot):
    		for name in files:
    			print(os.path.join(path,name))
    			os.remove(os.path.join(path,name))
    			ctr+=1
    	return "Deleted files="+str(ctr)
    return "ERROR"
@app.route('/helloApp',methods=['GET'])
def helloApp():
    return "Hello App"

'''
@app.route('/GetGestureNames',methods=['GET'])
def getGestureNames():
	filenames=os.listdir(".")
	#GestureNames=dict()
	for filename in filenames: 
			gesturename=filename.split("_")[0]
			if(GestureNames.haskey(gesturename)):
				GestureNames[gesturename]+=1
			else:
				GestureNames[gesturename]=1
	return ""+filenames

'''
@app.route('/getModelDetails',methods=['GET'])
def sendModelDetails():
	return "BLAH"

@app.route('/testSite',methods=['GET'])
def testSite():
	return render_template('index.html')
#app.run(host='0.0.0.0',port='8090')
