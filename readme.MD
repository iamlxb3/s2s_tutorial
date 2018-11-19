python 版本: Python 3.6.4

bytecup 最初级的版本

这个版本可能还是有BUG, 不过也很难测试吧，反正是跑通了，我先介绍下几个文件夹。
当然每个文件都会有每个文件的详细注释的，这里就比较粗的说一哈。

byte_cup/
这个就是主文件夹了，里面有3个文件，config是整个项目调参数的地方，run_train_val就是训练用的函数，整个项目的入口就是这个文件，run_test是
测试用的。

data/bytecup2018/
就是放数据的文件夹
里面放了2个超级缩小版的train和test，vocab的长度是8W多，OOV的词都用<UNK>代替。

字典创建的方式：
把训练集，验证集，测试集的文本都读进来，处理的方式就是先转成小写，然后用nltk.word_tokenize来把一个句子里的词提取出来，
接着计算词频，取词频前8W的作为字典，out of vocabulary的词一律认为是<UNK>, 字典里是一个List, 字典里还有<EOS>, <SOS>, <PAD>，
分别代表句子开始，句子结束，padding字符

下面说说输入文字转换为index的过程，就是比如I am a pig -> 10, 2, 1, 3 (数字代表了这个词在vocab里的index)
body和title在提取词的时候一点不同，body的内容在做word_tokenize之后，还会有一个Lemmatize的过程，用的也是ntlk，
title目前仅仅用word_tokenize提取词，当然都会转成小写。不在字典里的词全部转换成<UNK>，所以现在还有一步没有做，
就是如何把<UNK>再还原回去。

小数据集就是大数据集的子集，我手动生成的。

帮助函数就是封装一些项目里常用的函数，让代码看起来简洁一点

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