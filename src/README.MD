### 一、前言：

之前都在研究如何用TensorFlow_federated框架做一个实体的模型，但是折腾了好久发现由于TensorFlow没有认真学过，直接用tff做的话难度实在是太大。所以转了个方向，开始尝试不使用tff而是用朴实无华的TensorFlow直接实现联邦学习。尝试后发现好像还行。

之前有说过tff没有区分端代码，而是通过变量声明来控制执行端。这个设定好是好，就是对于新手过于不友好。使用朴素的TensorFlow后甚至能实现将Client和Server的代码分离开，理论上添加网络传输之后甚至可以分布式学习。

这个魔改的代码是基于AlexNet模型。

使用环境如下：

- python 3.7
- TensorFlow v1.14（如果想要复现最好就安装这个版本，不要安装最新的，有很多语法不一样）
- tqdm（一个python的库）

### 二、安装：
如果还没有安装，可以按照如下步骤：

1. 安装pip
2. 安装virutalenv
3. 启动virtualenv后在虚拟环境中安装TensorFlow和tqdm

#### 安装pip和python：
这两个步骤没啥好说的，也没有特殊要求，pip版本尽量新一些即可。网上有许多现成的教程可以参考，这里不多做赘述。

#### 安装virtualenv：
安装好pip后可以直接用pip安装virtualenv，Windows环境下先以管理员权限启动cmd后输入如下指令：
```
pip install -U pip virtualenv
```
#### 确认你是否都安装好了：
```
python --version
pip --version
virtualenv --version
```
#### 创建虚拟环境：
我以前是不屑于用虚拟环境的，知道有一天不知道安装了什么奇奇怪怪的python库导致出现了奇奇怪怪的bug...

创建虚拟环境前建议先创建一个专用的文件夹，不用有中文，然后在cmd中cd到这个文件夹下。

输入如下代码创建：
```
virtualenv --system-site-packages -p python3 ./venv
```

激活：
```
.\venv\Scripts\activate
```
如果成功的话你会看到命令行最前面出现了（venv）

在虚拟环境中升级pip

```
pip install --upgrade pip
```
如果你想退出虚拟环境：

```
deactivate #没事别瞎退
```

#### 在虚拟环境中安装TensorFlow：

由于我用的是1.14版本，命令如下：

```
pip install --upgrade tensorflow==1.14
```

验证：

```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

### 三、整活：

#### Client端：

client要做的事情包括
1. 把server发来的**模型变量**加载到模型上
2. 用自己的所有数据更新模型
3. 把**模型变量**发给server

client需要具备的功能：
1. 训练模型
2. 加载server的模型变量
3. 提取当前模型的变量值发给Server

其实就是比普通的TensorFlow模型多了个模型变量提取和赋值的操作，是不是够朴素的。

#### Server端：

server的任务包括：

1. 用一组模型变量给某个client更新，更新完后拿回来
2. 每一轮更新之后收集多个client更新后的模型并整合成新一轮的模型。

启动只要输入python server.py 即可

### 四、目前的进度：

1. 这个模型实在是过于简陋，但是能用了
2. 模型本身没有做任何优化，可操作空间大
3. 代码是基于AlexNet做的，但只要替换client端的函数就能更换模型
4. 本次代码没有设置任何加密，甚至没有传模型梯度，但是从官方文档来看是可以传递的，后续考虑增加。
5. 目前训练效果巨慢，而且效果很差，因为没有任何优化。
