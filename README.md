# implement-of-RePr
尝试复现RePr


论文看了好几遍，也看了一些讨论，讨论在：https://www.reddit.com/r/MachineLearning/comments/ayh2hf/r_repr_improved_training_of_convolutional_filters/eozi40e/

参考了这个复现：https://github.com/siahuat0727/RePr/blob/master/main.py

最后的结果就是：没达到论文效果，但是有一点点提升。

思考：

1.首先一点是，在ranking的时候是进行全局的ranking，就是将所有的filters放在一起prune。但是O（公式2）是通过层内的计算而来的。生成W（公式1）是先将flatten之后的filter进行了归一化。详细内容可以看论文的第五部分。要注意的是：在讨论中，作者提到，在进行rank时不考虑第一个卷积层。

2.重新初始化
论文中的方法是用QR分解。我这里产生过一个问题，假如filters（全局）的个数远大于flat后的权重，或者每一层的权重尺寸不一样，后面的QR分解怎么操作。因为文章说了，在重新初始化时新的权重是与原来被prune的权重和当前新的权重同时正交的。

3.论文中的figure1
![image1](https://github.com/SweetWind1996/implementation-of-RePr/blob/master/%E5%9B%BE1.png)
该训练图像很稳定，并且在reinitialize的时候没有出现下降的情况，在论文中的figure7中是出现了下降的。我在实验中也是出现下降的。

4.说一下结果：我重复训练了几次，结果不是很稳定，可能是我还没完全理解作者的思想或者代码写的存在问题，但这个工作也算是告一段落了。
