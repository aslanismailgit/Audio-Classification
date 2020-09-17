#%%
def train_model(modelname,emb):
    import os
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    import tensorflow.keras as keras
    import tensorflow as tf
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("imported")
    #%% path to json file that stores MFCCs and genre labels for each processed segment
    DATA_PATH = "./app/static/audio/data.json"

    with open(DATA_PATH, "r") as fp:
        data = json.load(fp)
    labels = data["mapping"]
    #%% convert lists to numpy arrays
    try:
        X = np.array(data["mfcc"])
    except:
        X = np.array(data["emb"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")
    print(X.shape)
    print(y.shape)
    #%% create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("Train - Test shapes--------------------------------------")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # add an axis to input sets
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_accuracy')>0.95):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()
    #%% build network topology
    # X shape = (500, 130, 13) there are 500 samples in dim of 130,30
    # so input shape is 130, 30
    numOfClasses = len (np.unique(y))
    if emb == 1:
        input_shape=(1, X.shape[1])
    else:
        input_shape=(X.shape[1], X.shape[2])

    

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(numOfClasses, activation='softmax')
    ])

    optimiser = keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    print("------------------- model created ---------------------")
    history = model.fit(X_train, y_train, 
            validation_data=(X_test, y_test), 
            batch_size=32, 
            epochs=100)
            #callbacks=[callbacks])

    pathtomodel = 'app/static/saved_model/'+ modelname
    model.save(pathtomodel) 
    #%%
    with open(pathtomodel+'/history1.js', 'w') as outfile:
        outfile.write("var history_data =")
        json.dump(history.history, outfile)
    #%%
    history = history.history
    jsonFormattedScores = []
    for i in range(len(history["loss"])):
        dict_temp = {"index":i,
                    'loss': history["loss"][i],
                    "accuracy": history["accuracy"][i],
                    "val_loss":history["val_loss"][i],
                    "val_accuracy":history["val_accuracy"][i]}
        jsonFormattedScores.append(dict_temp)
   
    with open(pathtomodel+'/history2.js', 'w') as outfile:
        outfile.write("var history_data =")
        json.dump(jsonFormattedScores, outfile)

    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))



    return labels, history

# %%
