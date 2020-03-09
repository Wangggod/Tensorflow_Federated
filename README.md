> **NOTE:**  这里使用经典的MINST训练来介绍Tensorflow Federated Learning(FL)的API。
2020.3.9 更新至colab在线版

---
### 一、准备工作：

##### 1、确保你的电脑已经安装了python和tensorflow_federated

On Ubuntu:

```
sudo apt update
sudo apt install python3-dev python3-pip  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```
On macOS:

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install --upgrade virtualenv  # system-wide install
```

##### 2、创建虚拟运行环境

```
virtualenv --python python3 "venv"
source "venv/bin/activate"
pip install --upgrade pip
```
Note: run deactivate 可以退出环境

##### 3、安装TensorFlow Federated （pip方式）

```
pip install --upgrade tensorflow_federated
```

##### 4、测试是否成功

```
python -c "import tensorflow_federated as tff; print(tff.federated_computation(lambda: 'Hello World')())"
```
如果成功显示：:"Hello World"则说明成功安装了TensorFlow Federated 

### 二、准备开始：
加载模块：
```
#@test {"skip": true}
!pip install --quiet --upgrade tensorflow_federated

# NOTE: Jupyter requires a patch to asyncio.
!pip install --quiet --upgrade nest_asyncio
import nest_asyncio
nest_asyncio.apply()

%load_ext tensorboard
```
测试是否加载成功：
```
import collections
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

tff.federated_computation(lambda: 'Hello, World!')()
```
如果成功则会看到输出“Hello world”
#### 准备输入数据：
数据集来源：

1. 经过Leaf处理过的用于FL的mnist数据集：femnist
2. tff已经集成的emnist数据集


> MNIST 的原始数据集为 NIST，其中包含 81 万张手写的数字，由 3600多个志愿者提供，目标是建立一个识别数字的 ML 模型。
通过调用 TFF 的 FL API，使用已由 GitHub 上的Leaf项目处理的 NIST 数据集版本来分隔每个数据提供者所写的数字

数据集构成为：
1. by_write：以用户划分，每个用户一个ID，共3600+数据，每个用户4份文件，分别是数字，大写字母，小写字母，混合字母。
2. by_class：按字符类别划分，每个文件夹对应一个字符，以字符的十六进制acsii码明明，如30-39对应数字0-9



```
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
print(len(emnist_train.client_ids))
```
第一次运行上述代码会自动开始下载数据，并输出3383。

查看数据信息：

```
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

example_element = iter(example_dataset).next()

print(emnist_train.element_type_structure)
print(example_element['label'].numpy())

plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.show()
```
得到结果如下：

```
(OrderedDict([('pixels', tf.float32), ('label', tf.int32)]), OrderedDict([('pixels', TensorShape([28, 28])), ('label', TensorShape([]))]))
```
即以字典形式存储，pixels是图片的key，label是对应图片标签的key，后面预处理将他们改为x和y

#### 数据预处理
1. 将28*28的图片展开成784像素的数组
2. 随机排序
3. 重组为batches
4. 将原本的pixels和label转换为x和y使得数据能够用于Keras


```
NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

print(sample_batch)
```
上述sample_batch输出结果如下，每个batches包含20对数据

```
OrderedDict([('x', array([[1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       ...,
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.]], dtype=float32)), ('y', array([[7],
       [7],
       [4],
       [0],
       [9],
       [1],
       [9],
       [5],
       [4],
       [8],
       [0],
       [4],
       [0],
       [9],
       [7],
       [0],
       [6],
       [7],
       [4],
       [1]], dtype=int32))])
```

#### 选择用户并生成对应用户的数据集
> 在模拟中向TFF提供联合数据的一种方法是简单地将其作为一个Python列表，该列表的每个元素都包含单个用户的数据，不管是作为列表还是tf.data.Dataset。
既然我们已经有了提供后者的接口，让我们使用它。下面是一个简单的帮助函数，它将构造来自给定用户集的数据集列表，作为一轮培训或评估的输入。 ————官方语录。


```
def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]
```
> 当然，我们是在一个模拟环境中，所有的数据都是本地可用的。通常情况下，当运行模拟时，我们会简单地对每一轮训练中涉及的客户的随机子集进行抽样，通常在每一轮中是不同的。 ————官方语录。


```
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print(len(federated_train_data))
print(federated_train_data[0])
```

> 为了简化，这里做的是对客户端集进行一次抽样（取10个客户端），然后每一轮都重复使用相同的这10个客户端，以加速收敛，故意对这几个用户的数据进行过度拟合)。
我们把它作为一个练习留给读者来修改本教程来模拟随机抽样——这是相当容易做到的(一旦你这样做了，记住如果每轮选择不同客户端，让模型收敛可能需要一段时间)。 ————官方语录。

### 3、用Keras创建模型

```
def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
      10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

  model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model
```
> 关于编译的一个重要注意事项。
如下所示，在联邦平均算法中使用时，优化器只是整个优化算法的一半，因为它只用于计算每个客户机上的本地模型更新。算法的其余部分涉及如何在客户机上平均执行这些更新，以及如何将它们应用到服务器上的全局模型。
特别是，这意味着这里使用的优化器和学习率的选择可能需要不同于您在标准的i.i.d.数据集上训练模型的选择。
我们建议从常规的SGD开始，学习速度要比平时慢。我们在这里使用的学习速度没有经过仔细调整，请随意尝试。
为了使用TFE的任何模型。它需要包装在一个 tff.learning.Model 接口的实例中。
与Keras类似，它公开了对模型的前向传递、元数据属性等进行标记的方法，但也引入了其他元素，例如控制计算联邦度量的过程的方法。
现在我们先不要担心这个，如果你有一个编译过的Keras模型，就像我们上面定义的那样，你可以通过调用 tf.learning.from_compiled_keras_model 让TFF为你包装它。from_compiled_keras_model，将模型和样本数据批处理作为参数传递，如下所示 ————官方语录。


```
def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)
```

### 4、训练模型
FedSGD和FedAVG

![image](https://upload-images.jianshu.io/upload_images/5750276-89c79073a8899d0e.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

> 参数说明：
C：每轮执行计算的client的比例fraction
E：每轮客户端对其本地数据集的训练遍数epochs
B：用于客户端更新的本地mini-batch大小。
当B取无穷，E取1时，代表每个client使用本地所有数据集作为一个batch，并且在一轮中训练了一个epoch，就变成了FedSGD

![image](https://upload-images.jianshu.io/upload_images/5750276-53b97fd3249c8f8d.png?imageMogr2/auto-orient/strip|imageView2/2/w/611/format/webp)

核心句：

```
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
```
调用 tff.learning.build_federated_averaging_process() ，将会返回一个 IterativeProcess 的实例，包含两个函数：initialize() 和 next()

- initialize() 用于初始化，返回的是训练开始时的state
- next() 输入当前的state，执行一轮计算，得到新的state

> next，代表了一轮Federated Averaging，它包括将服务器状态(包括模型参数)推给客户机，对它们的本地数据进行设备上的培训，收集和平均模型更新，并在服务器上生成一个新的更新模型


```
state = iterative_process.initialize()

# 训练10轮，并输出每轮精度
for round_num in range(1, 11):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))
```

### 5、测试精度


```
evaluation = tff.learning.build_federated_evaluation(model_fn)

train_metrics = evaluation(state.model, federated_train_data)
print(train_metrics)

federated_test_data = make_federated_data(emnist_test, sample_clients)

test_metrics = evaluation(state.model, federated_test_data)
print(test_metrics)
```



