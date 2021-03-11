import ic_module as ic

# 関数実行
# tsnumで各分類から何枚を精度確認用にするかを指定
# learn_scheduleで1epochごとに学習率をどのくらい減衰させるかを指定
# nb_epochで学習を何回繰り返すかを指定
ic.LearningRateScheduler(tsnum=30, nb_epoch=50, batch_size=8, learn_schedule=0.9)