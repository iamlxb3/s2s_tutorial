bytecup 最初级的版本

这个版本可能还是有BUG, 不过也很难测试吧，反正是跑通了，我先介绍下几个文件夹。
当然每个文件都会有每个文件的详细注释的，这里就比较粗的说一哈。

byte_cup/
这个就是主文件夹了，里面有3个文件，config是整个项目调参数的地方，run_train_val就是训练用的函数，整个项目的入口就是这个文件，run_test是
测试用的。

data/bytecup2018/
就是放数据的文件夹
里面放了2个超级缩小版的train和test，vocab的长度是8W多，OOV的词都用<UNK>代替。

funcs/
放了一下常用的函数，这里不细说了

model_pkls/
放置最佳模型的地方，每一个epoch都会在validation上做一次预测，然后保存在validation上最好的模型

utils/
就是放了一些比较通用的帮助函数


使用方法：

cd byte_cup
训练:
python -m run_train_val.py
测试:
python -m run_test.py

imoto特别注意下我写TODO的地方，你可能要在那里改参数