#%%
def load_model(modelname):
    import os
    import json
    import numpy as np
    import tensorflow.keras as keras
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # %% ============>  load model  <===================
    pathtomodel = 'app/static/saved_model/' + modelname
    #%%
    loaded_model = keras.models.load_model(pathtomodel)
    #print("------------  model loaded ----------------")
    return loaded_model
# %% ============>  predict  <===================
