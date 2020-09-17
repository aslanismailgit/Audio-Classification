from app import app
from flask import render_template
from flask import request, redirect, jsonify, make_response, session
import os
import numpy as np
import json
import shutil

embed = 0
app.config["SECRET_KEY"] = "OCML3BRawWEUeaxcuKHLpw"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/intro")
def intro():
    return render_template("intro.html")


@app.route('/makeclassfolder', methods=['GET','POST'])
def makeclassfolder():
    parent_dir ="./app/static/audio/records/"
    if request.method == "POST":
        classname =request.data.decode("utf-8")
        session['classname'] = classname
        path = os.path.join(parent_dir, classname) 
        try:
            os.mkdir(path)
        except:
            pass
        return redirect(request.url) 
    return render_template('makerecord.html')

@app.route('/makerecord', methods=['GET','POST'])
def makerecord():
    print("sayfa açılırken",request.url)
    filename ="audio.wav" 
    parent_dir ="./app/static/audio/records/"

    if (request.method == "POST"):
        classname = session.get('classname', None)
        print("clasname=====>", classname)
        savetopath =  os.path.join(parent_dir, classname,filename)
        print(savetopath)       
        with open(savetopath, 'wb') as f:
            f.write(request.data)
        res = make_response(jsonify({"message": "kayit tamam"}))
        
        return res #redirect(request.url)
    else: 
        pass
    return render_template('makerecord.html')


@app.route("/preparedata", methods=["GET", "POST"])
def preparedata():
    #print("sayfa açılırken",request.url.split("/")[-1])
    #if request.url.split("/")[-1] == "preparedata":
    userslist = createuserlist()
    print(userslist)
    if (request.method == "POST"):
        print("select butondan--->")
        print(request.form)
        selectedusers=[]
        value = request.form['submit_button']

        if value == 'select':
            print("selected selected")
            for i,m in enumerate(userslist):
                r = request.form.getlist(m)
                if r:
                    selectedusers.append(userslist[i])
            print(selectedusers)
            session["selectedusers"]=selectedusers
            print(selectedusers)
            makeuserdirectory(selectedusers)
            if embed == 1:
                from app.audio_embed import save_embeddings
                save_embeddings()

            else:
                from app.audio_prep import save_mfcc
                save_mfcc() 
            deleteuserdirectorytr(selectedusers)
            dataready = True
            return render_template("preparedata.html",userslist=userslist,dataready=dataready)
        elif value == 'delete':
            print("delete selected")
            for i,m in enumerate(userslist):
                r = request.form.getlist(m)
                if r:
                    selectedusers.append(userslist[i])
            print(selectedusers)
            session["selectedusers"]=selectedusers      
            deleteuserdirectoryrecord(selectedusers)
            userslist = createuserlist()
            return render_template("preparedata.html",userslist=userslist) 

    return render_template("preparedata.html",userslist=userslist)


def createuserlist():
    userslist = os.listdir("./app/static/audio/records")
    userslist = [ x for x in userslist if "." not in x]
    return userslist
def createmodellist():
    modellist = os.listdir("./app/static/saved_model")
    modellist = [ x for x in modellist if "." not in x]
    return modellist

def makeuserdirectory(selectedusers):
    source = "./app/static/audio/records"
    target = "./app/static/audio/train"
    for file in selectedusers:
        #print(file)
        tempsource = os.path.join(source, file)
        temptarget = os.path.join(target, file)
        shutil.copytree(tempsource, temptarget)
        filelist = os.listdir(temptarget)
        filelist = [ x for x in filelist if ".wav" in x]
        tempfolderlength = len(filelist)
        dublicate = False
        #print(filelist)
        #print(tempfolderlength)
        if (tempfolderlength==1) and not bool(embed):
            for filename in filelist: 
                temps = temptarget + "/"+ filename
                tempt = temptarget + "/copy_of_"+ filename
                #print("\n",temps,"\n",tempt)
                if ".wav" in temps: 
                    shutil.copy2(temps, tempt)
        

    return
def deleteuserdirectorytr(selectedusers):
    #modellist = os.listdir("./app/static/saved_model")
    #modellist = [ x for x in modellist if "." not in x]
    print("train files deleted")
    target = "./app/static/audio/train"
    print(selectedusers)
    for file in selectedusers:
        temptarget = os.path.join(target, file)
        shutil.rmtree( temptarget)
        print(temptarget, "-------- deleted")

    return
def deleteuserdirectoryrecord(selectedusers):
    #modellist = os.listdir("./app/static/saved_model")
    #modellist = [ x for x in modellist if "." not in x]
    print("train files deleted")
    target = "./app/static/audio/records"
    print(selectedusers)
    for file in selectedusers:
        temptarget = os.path.join(target, file)
        shutil.rmtree( temptarget)
        print(temptarget, "-------- deleted")

    return
def deletemodeldirectoryrecord(selectedmodels):
    #modellist = os.listdir("./app/static/saved_model")
    #modellist = [ x for x in modellist if "." not in x]
    print("train files deleted")
    target = "./app/static/saved_model"
    print(selectedmodels)
    for file in selectedmodels:
        temptarget = os.path.join(target, file)
        shutil.rmtree( temptarget)
        print(temptarget, "-------- deleted")
    return


@app.route("/train", methods=["GET", "POST"])
def train():
    if (request.method == "POST"):
        modelname = request.form['modelname']

        from app.train_model import train_model
        print("------ train_model imported ------------")
        labels, history = train_model(modelname,embed)
        print("------ MODEL IS READY ------------")

        session['labels'] = labels
        session['modelname'] = modelname
        session['history'] = history

        res = {
            "trained":True,
            "modelname":modelname,
        #    "labels":labels
        }
        print("------------model trained--------", modelname)
        return render_template("train.html", res=res)
    print("no name entered")
    return render_template("train.html")


@app.route("/plot_chart", methods=["GET", "POST"])
def plot_chart():
    path = "./app/static/saved_model/"
    history = session.get('history', None) 
    historyJSON = json.dumps(history)
    print("history taken")
    print("----------------", len(historyJSON))

    if request.method == "POST":
        # print("history taken")
        return render_template("train.html", history=(historyJSON))
    return render_template("train.html")



@app.route('/recordtest', methods=['GET','POST'])
def recordtest():
    modellist = session.get('modellist', None)
    print("sayfa açılırken",request.url)
    filename ="audio.wav" 
    parent_dir ="./app/static/audio/test/"
    # classnametest = session.get('classnametest', None)
    # print("clasnametest=====>", classnametest)
    savetopath = path = os.path.join(parent_dir,filename)
    print(savetopath)
    if (request.method == "POST"):       
        with open(savetopath, 'wb') as f:
            f.write(request.data)
        return redirect(request.url)
    else: 
        pass
    return render_template('predict.html',modellist = modellist)

@app.route("/predict", methods=['GET','POST'])
def predict():
    modellist = createmodellist()
    session["modellist"] = modellist
    path = "./app/static/audio/test/"
    conv = 0
    if (request.method == "POST"):

        test = path +  "audio.wav"
        print(test)

        labels = session.get('labels', None)
        modelname = session.get('modelnameforprediction', None) 
        print("-----", test, labels, modelname, "-----")
        # conv model is not used, so below lines are not required
        if "conv" in modelname:
            conv=1
        from app.load_model import load_model
        loaded_model = load_model(modelname)
        print("----------- model loaded -----------")
        if conv==1:
            from app.predict_embed import make_prediction_embed
            print (test,loaded_model,labels,conv)
            prediction = make_prediction_embed(test,loaded_model,labels,conv)
      
        else:
            from app.predict_model import make_prediction_mffc
            print (test,loaded_model,labels)
            prediction = make_prediction_mffc(test,loaded_model,labels)
        print(prediction)
        print("-----------prediction made-----------")
        
        return render_template("predict.html",modellist = modellist,prediction=prediction)
    return render_template("predict.html",modellist = modellist)


@app.route('/selectmodel', methods=['GET','POST'])
def selectmodel():
    modellist = session.get('modellist', None)
    selectedmodels=[]
    if request.method == "POST":
        rbutton = request.form['submit_button']
        if rbutton == "select" :
            for i,m in enumerate(modellist):
                r = request.form.getlist(m)
                if r:
                    modelnameforprediction = modellist[i]
                    print("selected model is : ----",modelnameforprediction)
                    session["modelnameforprediction"] = modelnameforprediction
                    return render_template("predict.html",modellist = modellist,modelnameforprediction=modelnameforprediction) 
        elif rbutton == "delete" :
            print("delete selected")
            for i,m in enumerate(modellist):
                r = request.form.getlist(m)
                if r:
                    selectedmodels.append(modellist[i])
            print(selectedmodels)
            session["selectedmodels"]=selectedmodels      
            deletemodeldirectoryrecord(selectedmodels)
            modellist = createmodellist()
            session["modellist"] = modellist
            return render_template("predict.html",modellist = modellist) 
        # return render_template("predict.html",modellist = modellist,modelnameforprediction=modelnameforprediction) 
    return render_template("predict.html",modellist = modellist) 
#--------------------------------------------------
# @app.route('/selectmodel', methods=['GET','POST'])
# def selectmodel():
#     modellist = session.get('modellist', None)

#     if request.method == "POST":
#         for m in modellist:
#             r = request.form.getlist(m)
#             if r :
#                 modelnameforprediction = m
#                 print("selected model is : ----",modelnameforprediction)
#                 session["modelnameforprediction"] = modelnameforprediction

#         return render_template("predict.html",modellist = modellist,modelnameforprediction=modelnameforprediction) 
#     return render_template("predict.html",modellist = modellist,modelnameforprediction=modelnameforprediction) 


# -------------> this is not used <-----------------------
@app.route('/makeclassfoldertest', methods=['GET','POST'])
def makeclassfoldertest():
    parent_dir ="./app/static/audio/test/"
    modellist = session.get('modellist', None)

    if request.method == "POST":
        #classnametest =request.data.decode("utf-8")
        #session["classnametest"] = classnametest
        path = os.path.join(parent_dir, classnametest) 
        try:
            os.mkdir(path)
        except:
            pass
        return redirect(request.url) 
    return render_template('predict.html',modellist = modellist)