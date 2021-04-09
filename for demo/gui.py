#
# This is the final GUI and set creation tool for this project
#

# importing dependencies
import tkinter as tk
import random
import webbrowser
from tkinter import font as tkFont
from tkinter import filedialog
from tkinter import TOP, BOTTOM, LEFT, RIGHT, IntVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa, librosa.display
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# supress warnings
import warnings

warnings.filterwarnings('ignore')

# creates the application window
window = tk.Tk()

# initialises variable that will hold the filename of the uploaded audio
filename = ''

# dictionary to get from the label output from the classifier to the actual readable value of key
key_dict = {
    0: 'A major',
    1: 'A minor',
    2: 'B minor',
    3: 'C major',
    4: 'D major',
    5: 'D minor',
    6: 'E major',
    7: 'E minor',
    8: 'F major',
    9: 'G major',
    10: 'G minor'
}
# dictionary to get from the label output from the classifier to the actual readable value of meter
meter_dict = {
    0: 'Hornpipe',
    1: 'Jig',
    2: 'Polka',
    3: 'Reel',
    4: 'Slide'
}
# dictionary of key values to:
#   at index 0: the weights of the clusters
#   from indices 1 - end: the centroids of the clusters
key_cluster_dict = {
    'A major': ([30, 35, 14, 21],
                ['Amajor', 'Amajor'],
                ['Dmajor', 'Dmajor'],
                ['Gmajor', 'Gmajor'],
                ['Dmajor', 'Amajor']
                ),
    'A minor': ([40, 18, 10, 32],
                ['Dmajor', 'Gmajor']
                , ['Gmajor', 'Dmajor']
                , ['Aminor', 'Aminor']
                , ['Dmajor', 'Dmajor']
                ),
    'B minor': ([17, 23, 18, 15, 17, 10],
                ['Bminor', 'Bminor'],
                ['Dmajor', 'Amajor'],
                ['Dmajor', 'Dmajor'],
                ['Dmajor', 'Eminor'],
                ['Dmajor', 'Gmajor'],
                ['Amajor', 'Amajor']
                ),
    'C major': ([47, 33, 13, 7],
                ['Gmajor', 'Dmajor'],
                ['Gmajor', 'Gmajor'],
                ['Fmajor', 'Dmajor'],
                ['Amajor', 'Emajor']
                ),
    'D major': ([18, 32, 32, 18],
                ['Gmajor', 'Gmajor'],
                ['Dmajor', 'Dmajor'],
                ['Gmajor', 'Dmajor'],
                ['Dmajor', 'Gmajor']

                ),
    'D minor': ([33, 16, 16, 23, 12],
                ['Dminor', 'Dminor'],
                ['Gmajor', 'Gmajor'],
                ['Dmajor', 'Amajor'],
                ['Fmajor', 'Gmajor'],
                ['Amajor', 'Gmajor']
                ),
    'E major': ([31, 38, 31],
                ['Gmajor', 'Dmajor'],
                ['Amajor', 'Amajor'],
                ['Dmajor', 'Eminor']
                ),
    'E minor': ([24, 25, 15, 15, 21],
                ['Dmajor', 'Gmajor'],
                ['Gmajor', 'Dmajor'],
                ['Gmajor', 'Eminor'],
                ['Dmajor', 'Dmajor'],
                ['Gmajor', 'Gmajor']
                ),
    'F major': ([50, 50],
                ['Dmajor', 'Dmajor'],
                ['Cmajor', 'Cmajor']
                ),
    'G major': ([22, 26, 26, 26],
                ['Gmajor', 'Dmajor'],
                ['Dmajor', 'Gmajor'],
                ['Dmajor', 'Dmajor'],
                ['Gmajor', 'Gmajor']
                ),
    'G minor': ([25, 50, 25],
                ['Fmajor', 'Fmajor'],
                ['Dmajor', 'Eminor'],
                ['Dminor', 'Dmajor']
                )
}
# dictionary of meter values to:
#   at index 0: the weights of the clusters
#   from indices 1 - end: the centroids of the clusters
meter_cluster_dict = {
    'Reel': ([96, 2, 1, 1],
             ['reel', 'reel'],
             ['reel', 'jig'],
             ['reel', 'hornpipe'],
             ['jig', 'reel']),
    'Jig': ([87, 8, 2, 2, 1],
            ['jig', 'jig'],
            ['jig', 'reel'],
            ['jig', 'slip jig'],
            ['jig', 'slide'],
            ['reel', 'jig']),
    'Polka': ([92, 6, 2],
              ['polka', 'polka'],
              ['reel', 'reel'],
              ['reel', 'polka']),
    'Hornpipe': ([76, 17, 4, 3],
                 ['hornpipe', 'hornpipe'],
                 ['reel', 'reel'],
                 ['reel', 'hornpipe'],
                 ['jig', 'jig']),
    'Slide': ([57, 30, 7, 5],
              ['slide', 'slide'],
              ['jig', 'jig'],
              ['jig', 'slide'],
              ['slide', 'jig'])
}

#method that opens a browser window to a given URL

def callback(url):
    webbrowser.open_new(url)

# method that makes the application window wait 1000 ms before updating
def waiting():
    var = IntVar()
    window.after(1000, var.set, 1)
    window.wait_variable(var)

#method takes the centroids of each cluster for meter and key
#   * filters tune database down to the meter and key for the first tune
#   * selects three tunes from the filtered data at random
#   * makes labels with the tune names visible
#   * attaches a link to the sheet music from The Session so that when clicked a browser window opens
#   * wait 1 s
#   * repeats same process for the next tune
def get_tunes(cluster_key, cluster_meter):
    df = pd.read_csv(r"data/tunes.csv")
    tune_type_1 = [cluster_meter[0]]
    tune_key_1 = [cluster_key[0]]
    tunes_1 = df.loc[(df['type'].isin(tune_type_1) & df['mode'].isin(tune_key_1)), ['tune', 'name']]
    first_tune = tunes_1.sample(n=3)
    print(first_tune)
    t1_sugg_frame['bg'] = '#f7b5f2'
    t1_l1['text'] = first_tune.iloc[0, :]['name']
    t1_l1_str = 'https://thesession.org/tunes/' + str(first_tune.iloc[0, :]['tune'])
    t1_l1.bind("<Button-1>", lambda e: callback(t1_l1_str))
    t1_l1['bg'] = '#f7b5f2'
    t1_l2['text'] = first_tune.iloc[1, :]['name']
    t1_l2_str = 'https://thesession.org/tunes/' + str(first_tune.iloc[1, :]['tune'])
    t1_l2.bind("<Button-1>", lambda e: callback(t1_l2_str))
    t1_l2['bg'] = '#f7b5f2'
    t1_l3['text'] = first_tune.iloc[2, :]['name']
    t1_l3_str = 'https://thesession.org/tunes/' + str(first_tune.iloc[2, :]['tune'])
    t1_l3.bind("<Button-1>", lambda e: callback(t1_l3_str))
    t1_l3['bg'] = '#f7b5f2'
    window.update()
    waiting()
    tune_type_2 = [cluster_meter[1]]
    tune_key_2 = [cluster_key[1]]
    tunes_2 = df.loc[(df['type'].isin(tune_type_2) & df['mode'].isin(tune_key_2)), ['tune', 'name']]
    second_tune = tunes_2.sample(n=3)
    print(second_tune)
    t2_sugg_frame['bg'] = '#fadeaf'
    t2_l1['text'] = second_tune.iloc[0, :]['name']
    t2_l1_str = 'https://thesession.org/tunes/' + str(second_tune.iloc[0, :]['tune'])
    t2_l1.bind("<Button-1>", lambda e: callback(t2_l1_str))
    t2_l1['bg'] = '#fadeaf'
    t2_l2['text'] = second_tune.iloc[1, :]['name']
    t2_l2_str = 'https://thesession.org/tunes/' + str(second_tune.iloc[1, :]['tune'])
    t2_l2.bind("<Button-1>", lambda e: callback(t2_l2_str))
    t2_l2['bg'] = '#fadeaf'
    t2_l3['text'] = second_tune.iloc[2, :]['name']
    t2_l3_str = 'https://thesession.org/tunes/' + str(second_tune.iloc[2, :]['tune'])
    t2_l3.bind("<Button-1>", lambda e: callback(t2_l3_str))
    t2_l3['bg'] = '#fadeaf'
    window.update()

# when the upload button is pressed:
#   * open the user's file explorer filtered only to .wav files
#   * if a file is selected:
#       * activate the submit button
#       * set the filename label to the name of the uploaded file
#   *else:
#       * set filename label to 'No file selected'
#       * deactivate the submit button
#   * hide all labels with suggested tunes, key and meter etc.
def upload_action(event=None):
    global filename
    filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav ")])
    print('Selected:', filename)
    if len(filename) != 0:
        slashes = [pos for pos, char in enumerate(filename) if char == '/']
        name = filename[max(slashes) + 1:len(filename)]
        submit_file['state'] = 'active'
    else:
        name = 'No file selected'
        submit_file['state'] = 'disabled'
    file_name_label['text'] = name
    key_label['text'] = ''
    meter_label['text'] = ''

    t1_l1['text'] = ''
    t1_l1['bg'] = '#C4C4F5'
    t1_l2['text'] = ''
    t1_l2['bg'] = '#C4C4F5'
    t1_l3['text'] = ''
    t1_l3['bg'] = '#C4C4F5'
    t2_l1['text'] = ''
    t2_l1['bg'] = '#C4C4F5'
    t2_l2['text'] = ''
    t2_l2['bg'] = '#C4C4F5'
    t2_l3['text'] = ''
    t2_l3['bg'] = '#C4C4F5'
    t1_sugg_frame['bg'] = '#C4C4F5'
    t2_sugg_frame['bg'] = '#C4C4F5'
    window.update()

# when submit button is pressed:
#   * load in the models for classifying key and meter
#   * convert the send the file to be made into a chromagram and tempogram
#       * hop_length = 256
#   * load in the temporary files of the images of the chromagram and tempogram
#   * use the images as inputs to predict key and meter using the classifier models
#   * set the key label to the predicted key
#   * wait 1 s
#   * set the meter label to the predicted meter
#   * wait 1 s
#   * pass the key and meter values to the get_tunes method
def submit_action(event=None):
    data_k = []
    data_m = []
    print(filename)
    meter_model = keras.models.load_model('data/meter model')
    key_model = keras.models.load_model('data/key model')
    make_chromagram(filename, 256)
    make_tempogram(filename, 256)
    img_height = 288
    img_width = 432
    image_ch = tf.keras.preprocessing.image.load_img('temp/chroma.png',
                                                     color_mode='rgb',
                                                     target_size=(img_width, img_height))
    key = np.array(image_ch)
    data_k.append(key.reshape((1, 432, 288, 3)))
    K = key_dict[key_model.predict_classes(data_k)[0]]
    print(K)
    key_label['text'] = K
    window.update()
    waiting()
    image_tp = tf.keras.preprocessing.image.load_img('temp/tempo.png',
                                                     color_mode='rgb',
                                                     target_size=(img_width, img_height))
    meter = np.array(image_tp)
    data_m.append(meter.reshape((1, 432, 288, 3)))
    M = meter_dict[meter_model.predict_classes(data_m)[0]]
    print(M)
    meter_label['text'] = M
    window.update()
    waiting()
    get_tunes(key_cluster(K), meter_cluster(M))

# gets the value from the dictionary key - being the key of the music
# reads in the weights from the first array in the value
# populates a list with the cluster indices however many times the weights dictate for those centroids
# randomly selects a value from the list of indices
# returns the centroids present at that index
def key_cluster(key):
    all_clusters = key_cluster_dict[key]
    weights = all_clusters[0]
    ratios = []
    for x in range(0, len(weights)):
        for y in range(0, weights[x]):
            ratios.append(x)
    y = random.randint(0, len(ratios) - 1)
    x = ratios[y]
    print('Key cluster:')
    print(all_clusters[x + 1])
    return all_clusters[x + 1]


# gets the value from the dictionary key - being the meter of the music
# reads in the weights from the first array in the value
# populates a list with the cluster indices however many times the weights dictate for those centroids
# randomly selects a value from the list of indices
# returns the centroids present at that index
def meter_cluster(meter):
    all_clusters_m = meter_cluster_dict[meter]
    weights_m = all_clusters_m[0]
    ratios = []
    for x in range(0, len(weights_m)):
        for y in range(0, weights_m[x]):
            ratios.append(x)
    y = random.randint(1, len(ratios) - 1)
    x = ratios[y]
    print('Meter cluster:')
    print(all_clusters_m[x + 1])
    return all_clusters_m[x + 1]

# uses librosa to compute a chromagram
# saves the chromagram as an image file in the temp folder
def make_chromagram(f, h):
    save_ch = 'temp/chroma.png'
    a, sr = librosa.load(f)
    chromagram = librosa.feature.chroma_stft(a, sr=sr, hop_length=h)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p_chroma = librosa.display.specshow(chromagram, x_axis=None, y_axis=None, hop_length=h, ax=ax)
    fig.savefig(save_ch)
    window.update()

# uses librosa to compute a tempogram
# computes 8 s duration with a 3 s delay of the audio
# saves the tempogram as an image file in the temp folder
def make_tempogram(f, h):
    save_tp = 'temp/tempo.png'
    a, sr = librosa.load(f, duration=8, offset=3)
    oenv = librosa.onset.onset_strength(y=a, sr=sr, hop_length=h)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=h)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p_tempo = librosa.display.specshow(tempogram, x_axis=None, y_axis=None, hop_length=h, ax=ax)
    fig.savefig(save_tp)
    window.update()

# GUI info
window.title("Set Creator")
window['background'] = '#ffcc99'
window.geometry("600x400")
tahoma16 = tkFont.Font(family='Tahoma', size=16)
tahoma12 = tkFont.Font(family='Tahoma', size=12)
title = tk.Label(window, text='Creating Sets with AI', font=tahoma16, bg='#ffcc99', fg='white', height=2)
title.grid(row=1, column=2)
empty_frame1 = tk.Frame(window, bg='#ffcc99', width=30)
empty_frame1.grid(row=2, column=1)
info_frame = tk.Frame(window, bg='#C4C4F5')
type_frame = tk.Frame(info_frame, bg='#C4C4F5')
key_label = tk.Label(type_frame, text='', width=20, bg='#C4C4F5', font=tahoma16, fg='white')
key_label.pack(side=LEFT, padx=5, pady=5)
meter_label = tk.Label(type_frame, text='', width=20, bg='#C4C4F5', font=tahoma16, fg='white')
meter_label.pack(side=RIGHT, padx=5, pady=5)
type_frame.pack(side=TOP, padx=2, pady=2)
suggestions_frame = tk.Frame(info_frame, bg='#C4C4F5')
t1_sugg_frame = tk.Frame(suggestions_frame, bg='#C4C4F5')
t2_sugg_frame = tk.Frame(suggestions_frame, bg='#C4C4F5')
t1_l1 = tk.Label(t1_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t1_l1.pack(side=TOP, padx=2, pady=2)
t1_l2 = tk.Label(t1_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t1_l2.pack(side=BOTTOM, padx=2, pady=2)
t1_l3 = tk.Label(t1_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t1_l3.pack(side=BOTTOM, padx=2, pady=2)
t2_l1 = tk.Label(t2_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t2_l1.pack(side=TOP, padx=2, pady=2)
t2_l2 = tk.Label(t2_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t2_l2.pack(side=BOTTOM, padx=2, pady=2)
t2_l3 = tk.Label(t2_sugg_frame, text='', bg='#C4C4F5', font=tahoma12, fg='black')
t2_l3.pack(side=BOTTOM, padx=2, pady=2)
t1_sugg_frame.pack(side=LEFT, padx=5, pady=5)
t2_sugg_frame.pack(side=RIGHT, padx=5, pady=5)
suggestions_frame.pack(side=BOTTOM, padx=5, pady=5)
info_frame.grid(row=2, column=2)
empty_frame2 = tk.Frame(window, bg='#ffcc99', width=30)
empty_frame2.grid(row=2, column=3)
file_frame = tk.Frame(window, bg='#ffcc99')
file_frame.grid(row=3, column=2)
t_frame = tk.Frame(file_frame, bg='#ffcc99')
t_frame.pack(side=TOP)
upload_file = tk.Button(t_frame, text='Upload', command=upload_action, font=tahoma16, background='#ff9999', fg='white')
submit_file = tk.Button(t_frame, text='Submit', command=submit_action, font=tahoma16, background='#C4C4F5', fg='white',
                        state='disabled', activebackground='#C4C4F5')
upload_file.pack(side=LEFT, padx=5, pady=5)
submit_file.pack(side=LEFT, padx=15, pady=5)
b_frame = tk.Frame(file_frame, bg='#ffcc99')
b_frame.pack(side=BOTTOM)
file_name_label = tk.Label(b_frame, text='No file selected', bg='#ffcc99')
file_name_label.pack(side=BOTTOM)

suggestions_frame = tk.Frame(window, bg='#ffcc99')
file_frame.grid(row=3)

#constantly looping so the program is always active
while (True):
    window.update()

