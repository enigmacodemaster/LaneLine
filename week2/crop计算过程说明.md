## VGG-FCN的crop计算说明
### 计算up2times的特征融合crop大小
假设输入图像尺寸为ixi，经过padding=100之后，以及第一个3x3卷积之后，尺寸是`$i+198$`，一次经过Pooling1，pooling2，pooling3，pooling4，pooling5，每一次大小缩小`$1 \over 2$`，经过Pooling5之后，大小变为`$i + 198 \over 32$`，经过7x7卷积之后，尺寸变为`$i + 6 \over 32$`。

此时，我们进行上采样反卷积一次，得到的大小为`$i + 38 \over 16$`,而需要融合的特征图尺寸为`$i + 198 \over 16$`，像素相差10。所以crop的大小为上下左右各5，从而有：

```python
h = h[:, :, 5 : 5 + up2_featuremap.size()[2], 5 : 5 + up2_featuremap.size()[3]]
```

### 计算up4times的特征融合crop大小
我们对刚才计算得到的h，尺寸为`$i + 38 \over 16$`在进行上采样，参数步长为2，kernel size为4，根据转置卷积计算公式`$output_{size} = (intput_{size} - 1) \times S + K$`得到的尺寸为`$i + 54 \over 8$`，对应的融合对象的特征图尺寸为`$i + 198 \over 8$`，尺寸相差为18，所以上下左右各crop的尺寸为9，所以有：

```
h = h[:, :, 9 : 9 + up4_featuremap.size()[2], 9 : 9 + up4_featuremap.size()[3]]
```

### 计算up8times的特征融合crop大小
继续对刚才得到的融合图进行上采样2倍，也就是总共上采样8倍。得到的尺寸为`$i + 62 \over 4$`，而对应的融合对象特征图尺寸为`$i + 198 \over 4$`，相差34，所以上下左右各crop的尺寸为17。

所以有：
```
h = h[:, :, 17 : 17 + up8_featuremap.size()[2], 17 : 17 + up8_featuremap.size()[3]]
```

### 计算up32times的特征融合crop大小
最后，把刚才得到的上采样8倍的特征图，上采样4倍，得到最后的预测结果。同样的，上采样8倍之后，尺寸大小根据转置卷积计算公式，可以得到`$$`
