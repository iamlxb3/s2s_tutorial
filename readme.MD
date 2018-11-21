**This is a playground for sequence to sequence learning in Pytorch.**
**Still in progress...**

Quick start:
The default dataset is a self-copy toy dataset. For other default configs, see s2s_tutorial/s2s_config.py

```
cd s2s_tutorial
python run_train_val.py
python run_test.py
```

To run for english-french translation task,

```
cd s2s_tutorial
python run_train_val.py --e exp1
python run_test.py --e exp1
```
You can modify experiments configs in s2s_tutorial/exp_config.py.

