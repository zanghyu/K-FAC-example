# KFAC 示例

#### 训练集和测试集

分别为：

```python
train_data0 = np.random.rand(200, 10000)
test_data0 = np.random.rand(10, 10000)
```

即[0,1]的均匀分布。

使用均方误差损失函数进行训练。



#### 训练结果

**KFAC算法**

iteration=10, lr=0.2

```
Step   0 loss 10.526428223
Step   1 loss 6.528557301
Step   2 loss 3.392151833
Step   3 loss 0.522906780
Step   4 loss 0.159504890
Step   5 loss 0.122227915
Step   6 loss 0.112411827
Step   7 loss 0.107950374
Step   8 loss 0.105365150
Step   9 loss 0.103511676
```

![](.\pic\kfac_result.png)

**SGD算法**

iteration=100, lr=0.2

```
...
Step  96 loss 8.312950134
Step  97 loss 8.312684059
Step  98 loss 8.312415123
Step  99 loss 8.312146187
```



![1571833770726](.\pic\sgd_result_01.png)

iteration=3000, lr=0.2

```
...
Step 2994 loss 4.116673946
Step 2995 loss 3.978591681
Step 2996 loss 3.882165194
Step 2997 loss 3.917830467
Step 2998 loss 4.034480095
Step 2999 loss 4.183311462
```

![1571833717304](.\pic\sgd_result_02.png)

iteration=10000, lr=0.2

![1571832395100](D:\学习\github\K-FAC-example\pic\sgd_result_03.png)



**由此看来，KFAC的方法还是比SGD快很多的。**



#### Modified from https://github.com/yaroslavvb/kfac_pytorch