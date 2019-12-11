# ocr_chinese
Keras实现自然场景下图像文字检测和识别，EAST/CRNN/CTC.

代码都是使用Keras+后端TensorFlow实现，方便生产环境部署和维护；

- EAST模型实现文字检测，文字方向支持90° ~ -90°间任意角度，包含中字/英文/数字/符号等，目标检测的方式画框定位，返回文本框的四个坐标。
- CRNN模型实现不定长文字识别，输出使用CTC算法模块。
- TensorFlow自带CTC，算法原理可以参考我的博客[CTC算法在OCR文字识别上的应用]( https://felixbrave.github.io/ )，还有[其他博文]( https://xiaodu.io/ctc-explained/ )



### 开发环境

```shell
python3.6 + tensorflow1.14.0 + keras2.1.6
# 或使用pip环境复制
pip install -r environment.txt
# GPU环境
NVIDIA Drivers/CUDA/cuDNN
```

###  How to use

从输入一张图片到端到端的检测及识别文字

```python
python predict.py
```

EAST模型检测文本框

```
cd east
python predict.py
```

CRNN模型识别文字

```
cd crnn
python predict.py
```



### 代码结构



### Documents

Writing ......

#### Training



### 参考

- [AdvancedEAST]( https://github.com/huoyijie/AdvancedEAST )
- [Pytorch_CRNN]( https://github.com/AstarLight/Lets_OCR/tree/master/recognizer/crnn )
- [CHINESE-OCR]( https://github.com/xiaofengShi/CHINESE-OCR )