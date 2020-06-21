import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


def load_data(train_audio_path):
    
    #train_audio_path = path

    labels=os.listdir(train_audio_path)
    #labels = ['stop', 'on', 'four']
    #print(labels)

    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/'+ label+ '/') if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if(len(samples)== 8000) : 
                all_wave.append(samples)
                all_label.append(label)


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y=le.fit_transform(all_label)
    classes= list(le.classes_)


    y=np_utils.to_categorical(y, num_classes=len(labels))

    all_wave = np.array(all_wave).reshape(-1,8000,1)

    return np.array(all_wave), np.array(y)


def get_dataloader(x,y,batch_size=32):
    x = x.transpose(0,2,1)
    labels = np.argmax(y,axis=1)

    x_tr, x_val, y_tr, y_val = train_test_split(x,labels,
                                               stratify=labels,test_size=0.2,random_state=777,shuffle=True)

    print("Training Data: ",x_tr.shape)
    print("Validating Data: ", x_val.shape)

    features_train = torch.from_numpy(x_tr).float()
    targets_train = torch.from_numpy(y_tr).long()

    features_val = torch.from_numpy(x_val).float()
    targets_val = torch.from_numpy(y_val).long()

    train_set = TensorDataset(features_train,targets_train)
    val_set = TensorDataset(features_val,targets_val)

    train_loader = DataLoader(train_set,batch_size,True)
    val_loader = DataLoader(val_set,batch_size,False)

    print(np.unique(labels))
    
    return train_loader, val_loader