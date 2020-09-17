def save_mfcc():
    # (dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, seg_length=3):

    import json
    import os
    import math
    import librosa

    #%%
    JSON_PATH = "./app/static/audio/data.json"
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    SEGMENT_LENGTH = 3
    HOP_LENGTH = 512
    N_FFT = 2048

    DATASET_PATH = "./app/static/audio/train/"
    #%%

    # Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

    #     :param dataset_path (str): Path to dataset
    #     :param json_path (str): Path to json file used to save MFCCs
    #     :param num_mfcc (int): Number of coefficients to extract
    #     :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    #     :param hop_length (int): Sliding window for FFT. Measured in # of samples
    #     :param: seg_length (int): length of segments we want to divide sample tracks into

    # dictionary to store mapping, labels, and MFCCs
    data = {"mapping": [], "labels": [], "mfcc": []}

    samples_per_segment = SEGMENT_LENGTH * SAMPLE_RATE
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        # ensure we're processing a genre sub-folder level
        if (dirpath is not DATASET_PATH):

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            #print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                if f[0] != ".":
                    # load audio file
                    #print(f)
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path,
                                                       sr=SAMPLE_RATE)

                    signal_length = signal.shape[0]
                    num_segment = int(signal_length /
                                      (SAMPLE_RATE * SEGMENT_LENGTH))
                    #print(num_segment)

                    # process all segments of audio file
                    for d in range(num_segment):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        #print("{}, segment:{}".format(file_path, d))
                        mfcc = librosa.feature.mfcc(signal[start:finish],
                                                    sample_rate,
                                                    n_mfcc=NUM_MFCC,
                                                    n_fft=N_FFT,
                                                    hop_length=HOP_LENGTH)
                        mfcc = mfcc.T
                        #print(mfcc.shape)

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            # i is -1 because the first one is the main dir
                            data["labels"].append(i - 1)
                            #print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)
    print("DATA is READY")
