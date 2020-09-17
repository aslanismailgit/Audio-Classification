#%%
def make_prediction_mffc(filename,loaded_model,labels):
    import os
    import numpy as np
    import tensorflow.keras as keras
    import librosa
    import math
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # %% ============>  predict  <===================
    SAMPLE_RATE = 22050
    SEGMENT_LENGTH = 3
    HOP_LENGTH = 512
    NUM_FFC=13
    N_FFT=2048 
    testFilePath = filename

    #%%
    signal_test, sample_rate_test = librosa.load(testFilePath, sr=SAMPLE_RATE)

    signal_test_length = signal_test.shape[0]
    num_segments = int(signal_test_length / (SAMPLE_RATE*SEGMENT_LENGTH))

    samples_per_segment = SEGMENT_LENGTH * SAMPLE_RATE
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    #%% process all segments of audio file
    tot_pred = 0
    mean_pred = 0
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal_test[start:finish], 
                sample_rate_test, 
                n_mfcc=NUM_FFC, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
#            print(" Segment ", d, "-------", mfcc.shape)
            X = mfcc[np.newaxis, ...] # array shape (1, 130, 13, 1)
            # perform prediction
            prediction = loaded_model.predict(X)
            # get index with max value
            predicted_index = np.argmax(prediction, axis=1)
            pred_label = labels[predicted_index[0]]
            true_label = testFilePath.split("/")[-1].split(".")[0]
            tot_pred += predicted_index[0]
            mean_pred = int(round(tot_pred/(d+1)))
            mean_pred_label = labels[(mean_pred)]
            print("Target: {}, Predicted label: {}, Predicted Index: {}".format(true_label, pred_label, predicted_index[0]))
            print(predicted_index[0],tot_pred,mean_pred,mean_pred_label,"--------------")
    prediction = {"class": mean_pred,
                    "label": mean_pred_label }
    #   print("tot pred: {}, mean_pred: {}, mean_pred label {}".format(tot_pred,mean_pred,mean_pred_label))
    return prediction

    # %%
    #prediction = make_prediction(filename,loaded_model,labels)
    #print(prediction)
# %%
