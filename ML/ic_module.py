import glob
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPoolinging2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils

FileNames = ["yellow.npy", "blue.npy"]
ClassNames = ["イエベ", "ブルベ"]
hw = {"height":16, "width":16}


###########################
#### 画像データの前処理 ####
###########################
def PreProcess(dirname, filename, var_amount=3):

    num = 0
    arrlist = []

    files = glob.glob(dirname + "/*.jpg")

    for imgfile in files:
        img = load_img(imgfile, target_size=(hw["height"], hw["width"])) # 画像ファイルの読み込み
        array = img_to_array(img) / 255                                  # 画像ファイルのnumpy化
        arrlist.append(array) # numpy型データをリストに追加
        num += 1

    nplist = np.array(arrlist)
    np.save(filename, nplist)
    print(">>" + dirname + "から" + str(num) + "個のファイル読み込み成功")


############################
###### モデルの構築 #########
############################
def BuildCNN(ipshape=(32, 32, 3), num_classes=3):

    model = Sequential()

    # 層1
    model.add(Conv2D(24, 3, padding='same', input_shape=ipshape)) # 3×3のフィルターでたたみ込み処理24回
    model.add(Activation('relu')) # 活性化関数

    # 層2
    model.add(Conv2D(48, 3)) # 畳み込み処理48回
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) # pool_size(2×2)の中の最大値を出力
    model.add(Dropout(0.5)) # 過学習防止(入力の50%を0に置き換え)

    # 層3　層4
    model.add(Conv2D(96, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(96, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # 層5
    mdoel.add(Flatten()) # Flatten()とDense(128)で要素128個の一次元配列へ
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 層6
    model.add(Dense(num_classes))
    model.add(Activation('softmax')) # 出力の個数を読み込んだフォルダの個数

    # 構築
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model

#########################
######### 学習 ##########
#########################
def Learning(tsnum=30, nb_epoch=50, batch_size=8, learn_schedule=0.9):
    # データの整理
    X_TRAIN_list = []; Y_TRAIN_list = []; X_TEST_list = []; Y_TEST_list = [];
    target = 0
    for filename in FileNames :
        data = np.load(filename)   # 画像のnumpyデータを読み込み
        trnum = data.shape[0] - tsnum
        X_TRAIN_list += [data[i] for i in range(trnum)]      # 画像データ
        Y_TRAIN_list += [target] * trnum                     # 分類番号
        X_TEST_list  += [data[i] for i in range(trnum, trnum+tsnum)] # 学習しない画像データ
        Y_TEST_list  += [target] * tsnum;                            # 学習しない分類番号

        target += 1
    
    X_TRAIN = np.array(X_TRAIN_list + X_TEST_list)  # 連結(学習するデータは後ろに連結)
    Y_TRAIN = np.array(Y_TRAIN_list + Y_TEST_list)  # 連結（同上）
    print(">> 学習サンプル数：", X_TRAIN.shape)
    y_train = np_utils.to_categorical(Y_TRAIN, target)  # 自然数をベクトルに変換(学習のしやすさのため)
    valrate = tsnum * target * 1.0 / X_TRAIN.shape[0]   # tsnum枚を精度確認用にする計算式

# epoch数が増える度に学習率を減らす
# initが最初の学習率
# lrが計算後つまり現在適用すべき学習率
# 学習が進むにつれ重みを収束しやすくする
    class Schedule(object):
        def __init__(self, init=0.001):    # 初期定義
            self.init = init
        def __call__(self, epoch):         # 現在値計算
            lr = self.init
            for i in range(1, epoch+1):
                lr *= learn_schedule
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    # 学習準備
    lrs = LearningRateScheduler(get_schedule_func(0.001))           # 学習率変換関数
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')  # val_lossが学習途中で最も小さくなる度に重みを保存する関数
    model = BuildCNN(ipshape=(X_TRAIN.shape[1], X_TRAIN.shape[2], X_TRAIN.shape[3]), num_classes=target)          # modelは構築した学習モデル

    # 学習
    print(">> 学習開始")
    hist = model.fit(X_TRAIN, y_train,          # 学習はfit, 学習に使用するデータX_TRAIN,y_trainを指定
                     batch_size=batch_size,     # batch_sizeは入力データをまとめて平均化する大きさ
                     verbose=1,
                     epochs=nb_epoch,           # epochsは学習の繰り返し回数
                     validation_split=valrate,  # valrateは精度確認用データの割合
                     callbacks=[lrs, mcp])      # callbacksは学習中に利用する関数

    # 保存
    json_string = mdoel.to_json()  # 学習モデルはjsonの形式で保存
    json_string += '##########' + str(ClassNames)  # jsonはテキストなので画像の分類名も付記して保存
    open('model.json' + 'w').write(json_string)    
    model.save_weights('last.hdf5')
                                                   # 重みもsave_weightsで保存

# 試行・実験
def TestProcess(imgname):
    # 読み込み
    modelname_test = open("model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])        # josn形式からモデルを読み込む
    model.load_weights("last.hdf5")  # best.hdf5 で損失最小のパラメータを使用, 重みの保存ファイル読み込み
    img = load_img(imgname, target_size=(hw["height"], hw["width"])) # 画像読み込み
    TEST = img_to_array(img) / 255                                   # 画像の数値化

    # 画像分類
    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)  # predictで学習結果を用いた計算ができる
    print(">> 計算結果↓\n" + str(pred))
    print(">> この画像は「" + textlist[np.argmax(pred)].replace(",", "") + "」です。")