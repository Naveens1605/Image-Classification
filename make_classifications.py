import matplotlib.pyplot as plt
import create_n_train
import numpy as np
import os
from create_n_train import MODEL_NAME, test_data, LR, IMG_SIZE, train_data

model2 = create_n_train.create_model()

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model2.load(MODEL_NAME)
    print('model loaded!')

fig=plt.figure()

for num,data in enumerate(test_data[36:54]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,6,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
    model_out = model2.predict([data])[0]

    if np.argmax(model_out) == 0: str_label='Bacterial'
    elif np.argmax(model_out) == 1: str_label='Healthy'
    elif np.argmax(model_out) == 2: str_label='Late Blide'
    else: str_label='Viral'

    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
