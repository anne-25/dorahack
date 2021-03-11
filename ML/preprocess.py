# フォルダを読み込む
# ic_module で始めに書いたClassNamesの順にディレクトリを指定

import ic_module as ic
import os.path as op

i = 0
for filename in ic.FileNames :
    # ディレクトリ名入力
    while True:
        dirname = input(">>「" + ic.ClassNames[i] + "」の画像のあるディレクトリ：")
        if op.isdir(dirname) :
            break
        print(">> そのディレクトリは存在しません")

    # 関数実行
    ic.PreProcess(dirname, filename, var_amount=3)
    i += 1