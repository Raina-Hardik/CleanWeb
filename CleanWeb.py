import os
import time
from multiprocessing import Process, Manager

import numpy as np
import tensorflow as tf
import webdataset as wds
from detoxify import Detoxify

image_size = 260
batch_size = 1024

data_dir = "./laion400m-dat-release/"
SHARDS = "{00000..00002}.tar"

target_dir1 = "./drawings/"
target_dir2 = "./hentai/"
target_dir3 = "./neutral/"
target_dir4 = "./porn/"
target_dir5 = "./sexy/"

try:
    os.mkdir(target_dir1)
    os.mkdir(target_dir2)
    os.mkdir(target_dir3)
    os.mkdir(target_dir5)
    os.mkdir(target_dir4)
except:
    pass


def get_class_string_from_index(index):
    for class_string, class_index in generator.class_indices.items():
        if class_index == index:
            return class_string


def filter_dataset(item):
    if 'txt' not in item:
        return False
    if 'jpg' not in item:
        return False
    return True


def image_classifier(caption_list, prediction_list, datadir):
    import tensorflow as tf
    import tensorflow_hub as hub

    ds = wds.WebDataset(datadir + SHARDS, handler=wds.ignore_and_continue) \
        .select(filter_dataset) \
        .decode('rgb') \
        .to_tuple('jpg', 'txt')

    dl = wds.WebLoader(ds,
                       shuffle=False,
                       num_workers=16,
                       batch_size=batch_size,
                       prefetch_factor=4 * batch_size)
    c = 0
    start = time.time()

    model = tf.keras.models.load_model('model.h5', custom_objects={"KerasLayer": hub.KerasLayer})

    c = 0
    start = time.time()

    print("starting loader")
    for im_arr, txt in dl:
        start = time.time()
        c += 1
        im_arr = tf.image.resize(im_arr, [260, 260], antialias=True)
        # print (im_arr.shape)
        prediction_scores = model.predict(im_arr)
        prediction_list.append(prediction_scores)
        captions = []
        txt_list = list(txt)
        for e in txt_list:
            captions.append(e[:200])  # captions are cut off after 200 characters, to avoid OOM errors

        caption_list.append(captions)
        print(c)
        print("image predition time")
        print(time.time() - start)
    del model
    tf.keras.backend.clear_session()


start = time.time()

n_drawings = 0
n_hentai = 0
n_neutral = 0
n_porn = 0
n_sexy = 0
manager = Manager()
prediction_list = manager.list()
caption_list = manager.list()
p = [Process(target=image_classifier, args=(caption_list, prediction_list, data_dir))]
p[0].start()
p[0].join()

model_txt = Detoxify('multilingual', device='cuda')
os.system("nvidia-smi")

for i in range(len(caption_list)):
    # start = time.time()
    # print(type(caption_list[i]))

    text_res = model_txt.predict(caption_list[i])

    predicted_indices = []
    for j in range(len(caption_list[i])):

        predicted_indices.append(np.argmax(prediction_list[i][j]))
        # print(prediction_list[i].shape)
        dist = np.array(tf.nn.softmax(prediction_list[i][j]))
        dist[1] = dist[1] + text_res["sexual_explicit"][j] + text_res["toxicity"][j]
        dist[3] = dist[3] + text_res["sexual_explicit"][j] + text_res["toxicity"][j]
        dist[4] = dist[4] + text_res["sexual_explicit"][j] + text_res["toxicity"][j]

        predicted_index = np.argmax(dist)

        if predicted_index == 0:
            n_drawings += 1

        if predicted_index == 1:
            n_hentai += 1
        if predicted_index == 2:
            n_neutral += 1

        if predicted_index == 3:
            n_porn += 1
        if predicted_index == 4:
            n_sexy += 1
            # print("n_sexy: "+str(n_sexy))
    print(i)

print("n_drawings: " + str(n_drawings))
print("n_hentai: " + str(n_hentai))
print("n_neutral: " + str(n_neutral))
print("n_porn: " + str(n_porn))
print("n_sexy: " + str(n_sexy))
print(time.time() - start)
