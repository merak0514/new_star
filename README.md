# 超新星

1. 提交文件的时候注意一下添加gitignore，不要把大文件搞上去了；我目前加了zip文件，和训练集测试集的名字，如果改名了需要再加。
2. 有新的收获可以记录在这里。


## 图像预处理

### 去除过大/过小点
1. 一开始尝试是把上下限定在100/150，但是最小值非常难调，90在一些图片完全无法滤掉噪声，120又会把噪声滤掉。
2. 随后尝试改成每张图片按照比例，把全图的平均值乘以一个权重。
3. 最后发现可以考虑把新图（b）过滤的狠一点，反正差值得到的负值最后都会被置为0，可能产生过滤掉星星的问题
4. 最后设置为均值的1.5倍, 两张图相等.

### 中值滤波
把每九个点的中位数作为中间点的值，这样可以直接滤掉孤立噪声点。

### 加入了均值相等的滤波:
把后一张图(b)的均值乘上一个系数, 让均值取成和c相等

### 问题：
若是出现大块的噪声，且亮度高于星星，则无法过滤。
整体效果还不错.