# tentips

- 开着梯子pip会报错
- tensor在产生时会自带维度的，reshape是为了保证其维度是我们想要的，符合某个你想使用的函数的特定要求
- 当光标在某一行中间时，按shift+enter可以跳转到下一行开始并且本行不发生变化
- ctrl+/会整块注释

- Pycharm给Python程序传递参数:https://blog.csdn.net/counte_rking/article/details/78837028
- Pycharm给Python程序传递参数:https://www.bilibili.com/video/BV14u411k7ws/?spm_id_from=333.788   后半段

![image-20220517163742490](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220517163742490.png)



- pycharm忽略大小写设置，即使输入小写r也可以提示Resize函数。

  setting->Editor->General->Code Completion,把第一行的Match case取消勾选即可。
  
- __init__和--call--的区别

  https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call

  ```py
  class Foo:
      def __init__(self, a, b, c):
          # ...
  
  x = Foo(1, 2, 3) # __init__
  
  ############
  
  class Foo:
      def __call__(self, a, b, c):
          # ...
  
  x = Foo()
  x(1, 2, 3) # __call__
  
  ####
  the __init__ method is used when the class is called to initialize the instance, while the __call__ method is called when the instance is called
  ```


- pycharm在函数括号中ctrl+p快捷键可以看括号中需要填写什么参数





# 一、Dataset类练习

目录结构如下：

![image-20220322105302762](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322105302762.png)

![image-20220322105316478](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322105316478.png)



```py
from torch.utils.data import Dataset
from PIL import Image
import os

class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        
    	#这两行定义了两个变量，并把参数传进来
        self.root_dir = root_dir  
        self.label_dir = label_dir
        
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        #这一步只是为了把路径中的斜杠都统一一下，其实统一不统一都可以
        #img_item_path = os.path.normpath(img_item_path)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "hymenoptera_data/train"
label_dir = "ants"
ants_set = Mydata(root_dir, label_dir)

#凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样  p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
#当实例对象通过[] 运算符取值时，会调用它的方法__getitem__
img, label = ants_set[1]
img.show()
```

运行结果是：读出第二张图片



上面路径的normpath的说明：

```py
import os
a = "hymenoptera_data/train"
b = "ants"
c = os.path.join(a, b)
#输出c为
#'hymenoptera_data/train\\ants'
#上面这个路径就可以直接用

d = os.path.join(a, b)
os.path.normpath(d)
#输出的d为
'hymenoptera_data\\train\\ants'
```





![image-20220322113641040](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322113641040.png)

上图代码中的ants_image就是上面的ants改名过来的（里面存的蚂蚁图片）



# 二、TensorBoard的使用

tensor数据类型的图片可以用TensorBoard来展示

## 1.SummaryWriter类的使用

本来是这样的：

![image-20220323100631253](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220323100631253.png)



运行下面代码之后：

```py
from torch.utils.tensorboard import SummaryWriter

# 在logs文件夹下放入事件文件
writer = SummaryWriter("logs")

# writer.add_image()
for i in range(100):
    writer.add_scalar("title", i, i)

#千万别忘记close
writer.close()
```





多了一个logs文件夹，里面放了一个事件文件：

![image-20220323100711897](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220323100711897.png)



首先在终端terminal使用下图命令（conda activate pytorch）启动pytorch环境

![image-20220905112659084](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220905112659084.png)

然后在终端terminal下运行

```py
tensorboard --logdir=logs
```

结果是：

![image-20220323103446724](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220323103446724.png)



浏览器输入上面的网址之后：

![image-20220323103703505](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220323103703505.png)



还可以改变端口名字(注意还是在终端terminal下输入的命令)：

```py
tensorboard --logdir=logs --port=6007
#把端口号改为了6007
```



```py
from torch.utils.tensorboard import SummaryWriter

# 在logs文件夹下放入事件文件
writer = SummaryWriter("logs")

# writer.add_image()
# scalar是标量的意思
for i in range(100):
    writer.add_scalar("title", 2*i, i) # 先y轴再x轴，这里显示的图像就是y=2x

writer.close()
```



## 2.add_image



![image-20220518181746677](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220518181746677.png)





![image-20220518181823487](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220518181823487.png)

![image-20220518181942536](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220518181942536.png)

# 三、torchvision中的transforms

torchvision.transforms : 常用的图像预处理方法，提高泛化能力 • 数据中心化 • 数据标准化 • 缩放 • 裁剪 • 旋转 • 翻转 •



```py
from PIL import Image
from torchvision import transforms

# 通过transforms.ToTensor解决两个问题
# 1.如何使用transforms
# 2.为什么我们需要Tensor数据类型。tensor数据类型：包装了神经网络所需要的理论基础参数
# tensor_img = tensor_trans(img)调用了call方法，相当于tennsor_img = tensor_tras.__call__(img)

img_path = "train/ants/6743948_2b8c096dda.jpg"
img = Image.open(img_path)

# 第一行是实例化对象，后面一行是直接调用对象的__call__方法
tensor_trans = transforms.ToTensor() # ToTensor是类名
tensor_img = tensor_trans(img) # 对象tensor_trans因为传入了img参数，所以自动调用了ToTensor()类中的__call__方法，__call__方法一般是对象来调动的
print(tensor_img)

#######
tensor([[[0.6549, 0.5451, 0.5765,  ..., 0.7451, 0.7451, 0.7647],
         [0.4078, 0.4471, 0.5373,  ..., 0.8118, 0.8431, 0.8627],
         [0.3529, 0.5804, 0.7490,  ..., 0.6824, 0.8314, 0.8824],
         ...,
         [0.5490, 0.5412, 0.4353,  ..., 0.6510, 0.6706, 0.6275],
         [0.8824, 0.5020, 0.8353,  ..., 0.6745, 0.6706, 0.5608],
         [0.6235, 0.3961, 0.7765,  ..., 0.7765, 0.6784, 0.6196]],
...
```

# 四、常见的transforms

```py
# Image.open()   对应PIL类型
# ToTensor()     对应tensor类型
# cv.imread()    对应narray类型

```



## 内置函数call

```py
### __call__
# init是构造函数  call是将类作为函数来使用 直接 类名(参数)
class Person:
    def __call__(self, name):
        print("__call__"+"hello"+name)

    def hello(self, name):
        print("hello"+name)

person=Person()
person("zhangsan") # 对象调用call函数
person.hello("lisi")

####
__call__hellozhangsan
hellolisi
```

## 归一化Normalize

```py
# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # 设定每个通道的均值和标准差都是0.5
img_norm = trans_norm(tensor_img) # tensor_img是tensor格式的图片
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm) #输入网址和端口号可以找到图片

writer.close()

#####
tensor(0.6549)
tensor(0.3098)
```

## Resize

```py
# Resize
trans_resize = transforms.Resize((512,512)) # 左边是对象，右边是类名
img_resize = trans_resize(img) # 这里使用的是Resize类中的forward(self, img)方法
writer.close()
# 接下来可以使用writer.add_image来查看图片，详看2.2
```

## Compose

```py
# Compose
trans_resize_2 = transforms.Resize(512) # 正方形裁剪
trans_compose = transforms.Compose([trans_resize_2, tensor_trans]) # Compose的类型必须是transforms类型，且为列表形式，这俩参数都是transforms类型，其中第一个参数的输出类型必须是第二个参数的输入类型，也就是说参数一个个的来对图片进行操作
## PIL->PIL->tensor类型
img_compose = trans_compose(img) # 这里的参数img，类型是PIL类型的，因为上一行中的Compose两个transforms参数要求的参数类型是PIL
writer.add_image("Compose", img_compose)
```

## RandomCrop

![image-20220519162216541](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519162216541.png)

# 五、torchvision的数据集的使用

pytorch.org官网的数据集

最新版的在Search Docs左上角选择版本为0.9.0就一样了



举例为https://pytorch.org/vision/0.9/datasets.html#cifar下的

![image-20220519175616866](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519175616866.png)



```py
import  torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True) # 因为没有提前准备数据集，所以download设为True
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True) # 因为没有提前准备数据集，所以download设为True
```

运行代码之后会下载数据集：（感觉下载慢的话，可以把运行框中的下载链接放在迅雷中进行下载）

![image-20220519180127530](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519180127530.png)



标记断点之后，显示classes总共是这么多

![image-20220519181546533](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519181546533.png)



```py
import  torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True) # 因为没有提前准备数据集，所以download设为True
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True) # 因为没有提前准备数据集，所以download设为True
print(test_set[0]) #两个输出，第一个是图片，第二个是label
print(test_set.classes) #看看全部的label

img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show() # 显示图片

########
Files already downloaded and verified
Files already downloaded and verified
(<PIL.Image.Image image mode=RGB size=32x32 at 0x20AC11ACBA8>, 3)
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
<PIL.Image.Image image mode=RGB size=32x32 at 0x20AC11AC9E8>
3
cat

Process finished with exit code 0
```



加了一个dataset_transform进行处理

```py
import  torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True) # 因为没有提前准备数据集，所以download设为True
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True) # 因为没有提前准备数据集，所以download设为True

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

print(test_set[0])

##########
Files already downloaded and verified
Files already downloaded and verified
(tensor([[[0.6196, 0.6235, 0.6471,  ..., 0.5373, 0.4941, 0.4549],
         [0.5961, 0.5922, 0.6235,  ..., 0.5333, 0.4902, 0.4667],
         [0.5922, 0.5922, 0.6196,  ..., 0.5451, 0.5098, 0.4706],
         ...,
         [0.2667, 0.1647, 0.1216,  ..., 0.1490, 0.0510, 0.1569],
         [0.2392, 0.1922, 0.1373,  ..., 0.1020, 0.1137, 0.0784],
         [0.2118, 0.2196, 0.1765,  ..., 0.0941, 0.1333, 0.0824]],

        [[0.4392, 0.4353, 0.4549,  ..., 0.3725, 0.3569, 0.3333],
         [0.4392, 0.4314, 0.4471,  ..., 0.3725, 0.3569, 0.3451],
         [0.4314, 0.4275, 0.4353,  ..., 0.3843, 0.3725, 0.3490],
         ...,
         [0.4863, 0.3922, 0.3451,  ..., 0.3804, 0.2510, 0.3333],
         [0.4549, 0.4000, 0.3333,  ..., 0.3216, 0.3216, 0.2510],
         [0.4196, 0.4118, 0.3490,  ..., 0.3020, 0.3294, 0.2627]],

        [[0.1922, 0.1843, 0.2000,  ..., 0.1412, 0.1412, 0.1294],
         [0.2000, 0.1569, 0.1765,  ..., 0.1216, 0.1255, 0.1333],
         [0.1843, 0.1294, 0.1412,  ..., 0.1333, 0.1333, 0.1294],
         ...,
         [0.6941, 0.5804, 0.5373,  ..., 0.5725, 0.4235, 0.4980],
         [0.6588, 0.5804, 0.5176,  ..., 0.5098, 0.4941, 0.4196],
         [0.6275, 0.5843, 0.5176,  ..., 0.4863, 0.5059, 0.4314]]]), 3)

Process finished with exit code 0
```



tensor数据类型可以使用tensorBoard进行显示

```py
import  torchvision
from torch.utils.tensorboard import SummaryWriter

# 可以组合，但是这里只使用了一个
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]) # 这里只用了一个transform

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True) # 因为没有提前准备数据集，所以download设为True，其实下没下好都建议设置为True
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True) # 因为没有提前准备数据集，所以download设为True

writer = SummaryWriter("dataset_logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()
```



运行之后在terminal中打开日志文件

![image-20220519184032756](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519184032756.png)

运行之后

![image-20220519184107729](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519184107729.png)



浏览器打开链接，显示如下

![image-20220519184213217](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220519184213217.png)



# 六、Dataloader

![image-20220520093734306](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520093734306.png)



num_workers（进程数？可以看下英文）>0时在windows下可能出现如下错误（BrokenPipeEeeor）：这时把num_workers设为0

![image-20220520094014420](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520094014420.png)



点开CIFAR10，可以看到getitem函数返回的是：

![image-20220520095337361](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520095337361.png)

img，target（label）

![image-20220520095446884](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520095446884.png)



所以可以直接写

```py
img,target = test_set[0]
```



shuffle先打乱，后选取，随机取4个(弹幕说的，我也不清楚？？？？？？？？？)

打乱是为了随即抓取批量

```py
import torchvision
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10("./dataset", False, torchvision.transforms.ToTensor()) # 测试的数据集
test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img,target = test_set[0]
print(img.shape) # 三通道，尺寸是32*32
print(target)
print(test_set.classes[target])

##################
torch.Size([3, 32, 32])
3
cat

Process finished with exit code 0
```



接着上面的程序写

```py
for data in test_loader:  # 注意是test_loader，不是test_set
    imgs, targets = data
    print(imgs.shape)
    print(targets)
    
######
torch.Size([4, 3, 32, 32]) # batch_size=4，3channel，32*32像素， 四张图片的打包
tensor([1, 5, 9, 6])       # 4张图片的label（target）放在一块，打包
torch.Size([4, 3, 32, 32])
tensor([9, 2, 5, 4])
torch.Size([4, 3, 32, 32])
tensor([5, 3, 2, 1])
torch.Size([4, 3, 32, 32])
tensor([3, 9, 7, 2])
torch.Size([4, 3, 32, 32])
tensor([0, 3, 0, 3])
torch.Size([4, 3, 32, 32])
tensor([1, 8, 1, 1])
torch.Size([4, 3, 32, 32])
tensor([8, 2, 6, 3])
torch.Size([4, 3, 32, 32])
tensor([2, 5, 6, 7])
torch.Size([4, 3, 32, 32])
tensor([6, 1, 3, 2])
....没有写完
```



显示tensor数据类型的图片

```py
writer = SummaryWriter("test_loader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("testLoader", imgs, step)  # 注意这里是add_images,不是add_image
    step += 1
writer.close()
```



然后terminal输入

```py
(base) C:\Users\zgliang\Desktop\learn_torch>tensorboard --logdir="test_loader"
运行之后浏览器输入网址即可显示图片
```



加个epoch,因为上面的DataLoader中的shuffle=True，所以这里是每个epoch就shuffle一次。如果shuffle=False的话，那么两轮都是一样的。

```py
writer = SummaryWriter("test_loader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch{}".format(epoch), imgs, step)
        step += 1
writer.close()
# 运行for循环就会重新调用test_loader,如果shuffle是True的话，那么bactch_size中的图片都不一样
```

# 七、神经网络基本骨架nn.Module

## call函数中调用了forward函数

```py
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input):
        output = input + 1
        return output

m=Model()
x = torch.tensor(1.0)
output = m(x) # 这里没调用forward函数，但是其实用的call函数中调用了forward函数
print(output)


# 在Model继承的父类nn.Module里应该有一个__call__()直接自动调用了forward函数，而在Model这个nn.Module的子类里我们又重写了forward函数，所以我们间接地通过父类nn.Module里的__call__()调用了子类Model里的forward函数
# 解释可以看https://blog.csdn.net/xu380393916/article/details/97280035
# 和https://zhuanlan.zhihu.com/p/392233393
```

查看nn.Module类中的--call--函数，可以看到这么一行简短的定义：

说明：因为__call__()方法的实现_call_impl()中调用了forward()方法

![image-20220520151252672](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520151252672.png)



这么一行定义什么意思可以参考https://ppday.cn/2020/09/14/Python%E4%B8%ADtyping%E6%A8%A1%E5%9D%97%E4%B8%8E%E7%B1%BB%E5%9E%8B%E6%B3%A8%E8%A7%A3%E7%9A%84%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95/

或者https://cuiqingcai.com/7071.html

上面已经摘录在./笔记/python类型注解.md

# 八、卷积

卷积核，不赘述。

参数说明：

- stride是卷积每次移动的步长
- padding主要是在图片的四周填充

![image-20220520162210623](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520162210623.png)

![image-20220520163924568](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520163924568.png)



代码如下：

![image-20220520163404011](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520163404011.png)

这里面conv2d(N,C,H,W)里面的四个是 N就是batch size也就是输入图片的数量，C就是通道数这只是一个二维张量所以通道为1，H就是高，W就是宽，所以是1 1 5 5

(B,C,H,D)=(batch_size,channel,高度，宽度)，二维卷积操作需要输入一个四元素的向量

运行结果为：

![image-20220520163539644](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520163539644.png)

输出结果时四维张量，是因为reshap后就是四为张量了，输入的就是一个四维，输出的自然也是四维的？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

# 九、卷积层

![image-20220520164523923](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220520164523923.png)

二维卷积的一些参数如上。



```py
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True) # 转成tensor的形式
dataloader = DataLoader(dataset, batch_size=64)

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__() # 这个super函数的python2的写法，直接用super().__init__()就可以
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0) # ctrl+p显示参数的列表
## 卷积核中数据是自动生成的，在训练过程中自动更新，也就是向后传播
        
    def forward(self, x):
        self.conv1(x)
        return x

m = Model1()
for data in dataloader:
    imgs, targets = data
    output = m(imgs)
    print(imgs.shape)
    print(output.shape) #这里没输出出来，不知道为什么？？？？？？？？？？？？？？正常应该输出torch.Size([64, 6, 30, 30])
    
#########
torch.Size([64, 3, 32, 32])  ## batch_size=64，3通道
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
...
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([64, 3, 32, 32])
torch.Size([16, 3, 32, 32])
torch.Size([16, 3, 32, 32])
```







本来是叠起来，现在摊开（将output  reshape一下）

![image-20220521153833293](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220521153833293.png)



# 十、最大池化（也叫做下采样）

## 池化层目的：

## 对特征图进行稀疏处理，减少数据运算量。

![image-20221119150737917](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20221119150737917.png)



![image-20221119150829042](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20221119150829042.png)

第三条：池化核的大小和步距是相同的。





池化层夹在连续的卷积层中间，压缩数据和参数的量，减小过拟合，池化层并没有参数，它只不过是把上层给它的结果做了一个下采样（数据压缩）。下采样有**两种**常用的方式：

**Max pooling**：选取最大的，我们定义一个空间邻域（比如，2x2 的窗口），并从窗口内的修正特征图中取出最大的元素，最大池化被证明效果更好一些。

**Average pooling**：平均的，我们定义一个空间邻域（比如，2x2 的窗口），并从窗口内的修正特征图算出平均值



- 最大池化的作用是保留输入数据的特征并且把数据量减小（牺牲信息换效率），池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。本质是 降采样，可以大幅减少网络的参数量。



![微信图片编辑_20220522155415](C:\Users\zgliang\Desktop\微信图片编辑_20220522155415.jpg)



常用的是MaxPool2d

![image-20220522155530901](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220522155530901.png)

![image-20220522155628857](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220522155628857.png)



注意stride默认值是kernel_size，跟卷积核有所不同。



ceil_mode有ceil（天花板）模式和floor（地板）模式，floor模式要舍去，不做计算。

![image-20220522160509178](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220522160509178.png)





![image-20220522172422680](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220522172422680.png)

MaxPool的输入格式如上所示，N为batchsize，C是channel，H高，W宽。

```py
import torch
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

# MaxPool2d对输入格式有要求，所以reshape一下
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)  ##输出为 torch.Size([1, 1, 5, 5])


```



```py
import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

# MaxPool2d对输入格式有要求，所以reshape一下
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.MaxPool1(input)
        return output

nn1 = Model()
output = nn1(input)  ###此句也可改写为 output = nn1.MaxPool1(input)
print(output)

######这里报错
RuntimeError: "max_pool2d" not implemented for 'Long'

将input数据类型改一下就可以运行，如下
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)

########################
torch.Size([1, 1, 5, 5])
tensor([[[[2., 3.],
          [5., 1.]]]])

Process finished with exit code 0


######################
把torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)的ceil_mode改为False
结果为：

torch.Size([1, 1, 5, 5])
tensor([[[[2.]]]])

Process finished with exit code 0
```



1、./是当前目录  试验了一下放在跟py文件平级的目录下

2、../是父级目录 

3、/是根目录 根目录指逻辑驱动器的最上一级目录     实验了一下直接放在c盘下了（因为我的py程序在c盘的桌面上）

```py

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set = torchvision.datasets. CIFAR10("./data1", train=True, transform=torchvision.transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=data_set, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.MaxPool1(input)
        return output

nn1 = Model()
writer = SummaryWriter("logs_maxpool")
step = 0
for data in data_loader:
    imgs, targets = data
    writer.add_images("imgs", imgs, step)
    output = nn1.MaxPool1(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

###########
运行结果放在tensorboard里面即可
```

# 十一、非线性激活Non-linear Activations (weighted sum, nonlinearity)

激活函数作用：没有激活函数的话,你的神经网络不管多少层,都是线性变换.

没有激活函数的神经网络实际上是线性可加的，那么多线性层其实可以归为一层。只具有线性的神经网络表达能力极其有限。

所以增加非线性的激活函数实际上是给模型增加非线性的表达能力或者因素，有了非线性函数模型的表达能力就会更强。整个模型就像活了一样，而不是想机器只会做单一的线性操作。激活函数是加入非线性因素，线性并不能很好的拟合现实的情况，加入非线性因素可以增强拟合能力



二分类输出层用sigmod，隐藏层用ReLu，ReLU对自然语言处理的处理比较明显 是NLP常用的激活函数

多分类sigmoid换softmax



inplace参数的含义：True就是原地操作

![image-20220523152536282](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523152536282.png)







![image-20220523153822036](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523153822036.png)

批的大小就是batch_size. 每取一次就是一个step，step跟batch_sizes 是有关系的。因为数据集通常很大，没有办法一次性全取出来，单个单个取又很浪费时间效率（跟IO操作有关系）。所以是按照批取得。



# 十二、线性层及其他层介绍

https://pytorch.org/docs/stable/nn.html#



## Normalization Layers

这个叫批处理归一化，不是正则化，也叫标准化归一化

有论文表示，Normalization可以增加训练速度

![image-20220523154513700](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523154513700.png)



## Recurrent Layers

文字识别可能会用到

![image-20220523154911929](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523154911929.png)



## Transformer Layers

![image-20220523155108242](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523155108242.png)

## Linear Layers

线性层用到的比较多

这里出错是因为drop_last默认是False，保留了最后16张图，可以设置成drop_last=True就正常运行了



![image-20220523155142924](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523155142924.png)

![image-20220523155253491](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523155253491.png)



## Dropout Layers

按概率dropout，主要防止过拟合

![image-20220523155714300](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523155714300.png)

# 十三、搭建一个简单的网络

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523190926328.png)



Flatten展开后变为1024（1024=64 x 4 x 4）的向量



搭建在nn_seq.py文件上

```py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class nn_seq(nn.Module):
    def __init__(self):
        super(nn_seq, self).__init__()
        # 根据第一层的图片来写参数，padding是根据上面截图的公式计算得来的，stride和dilation都是默认值
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        # 步长为1时，padding只与卷积核尺寸有关.就是保持图片卷积之后的尺寸不变时padding到底取多少，想一想图片很容易理解
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten() # 这个不需要写参数,展开后的大小应该是1024（64*4*4=1024）
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self, input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)
        input = self.conv3(input)
        input = self.maxpool3(input)
        input = self.flatten(input)
        input = self.linear1(input)
        input = self.linear2(input)
        return input

nn_seq1 = nn_seq()
print(nn_seq1)

###############
nn_seq(
  (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=1024, out_features=64, bias=True)
  (linear2): Linear(in_features=64, out_features=10, bias=True)
)

Process finished with exit code 0
```





给网络加上一个输入x

```py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class nn_seq(nn.Module):
    def __init__(self):
        super(nn_seq, self).__init__()
        # 根据第一层的图片来写参数，padding是根据上面截图的公式计算得来的，stride和dilation都是默认值
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        # 步长为1时，padding只与卷积核尺寸有关.就是保持图片卷积之后的尺寸不变时padding到底取多少，想一想图片很容易理解
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten() # 这个不需要写参数
        self.linear1 = Linear(in_features=1024, out_features=64)
        self.linear2 = Linear(in_features=64, out_features=10)

    def forward(self, input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.conv2(input)
        input = self.maxpool2(input)
        input = self.conv3(input)
        input = self.maxpool3(input)
        input = self.flatten(input)
        input = self.linear1(input)
        input = self.linear2(input)
        return input

nn_seq1 = nn_seq()
print(nn_seq1)
x = torch.ones((64, 3, 32, 32))
output = nn_seq1(x) # 这个x应该是神经网络里面的input
print(output)
print(output.shape)  ## torch.Size([64, 10])

####################
nn_seq(
  (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear1): Linear(in_features=1024, out_features=64, bias=True)
  (linear2): Linear(in_features=64, out_features=10, bias=True)
)
tensor([[ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745],
        [ 0.0537,  0.1085, -0.0877,  0.0228,  0.0401, -0.0143,  0.1482, -0.0769,
         -0.0906, -0.0745]], grad_fn=<AddmmBackward0>)

Process finished with exit code 0
```



## Sequential简化代码

```py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class nn_seq(nn.Module):
    def __init__(self):
        super(nn_seq, self).__init__()
        # # 根据第一层的图片来写参数，padding是根据上面截图的公式计算得来的，stride和dilation都是默认值
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1)
        # self.maxpool1 = MaxPool2d(kernel_size=2)
        # # 步长为1时，padding只与卷积核尺寸有关.就是保持图片卷积之后的尺寸不变时padding到底取多少，想一想图片很容易理解
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # self.maxpool2 = MaxPool2d(kernel_size=2)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        # self.maxpool3 = MaxPool2d(kernel_size=2)
        # self.flatten = Flatten() # 这个不需要写参数
        # self.linear1 = Linear(in_features=1024, out_features=64)
        # self.linear2 = Linear(in_features=64, out_features=10)
        self.model1 = Sequential(
            # 直接把上面右边的模型抄过来就可以
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, input):
        # input = self.conv1(input)
        # input = self.maxpool1(input)
        # input = self.conv2(input)
        # input = self.maxpool2(input)
        # input = self.conv3(input)
        # input = self.maxpool3(input)
        # input = self.flatten(input)
        # input = self.linear1(input)
        # input = self.linear2(input)
        output = self.model1(input)
        return output

nn_seq1 = nn_seq()
print(nn_seq1)
x = torch.ones((64, 3, 32, 32))
output = nn_seq1(x)
print(output.shape)
```





后面用tensorboard打开图片

```py
nn_seq1 = nn_seq()
print(nn_seq1)
x = torch.ones((64, 3, 32, 32))
output = nn_seq1(x)
print(output.shape)

writer = SummaryWriter("./logs_seq")
writer.add_graph(nn_seq1, x) # ctrl+p查看参数类型
writer.close()
```





![image-20220523203539615](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523203539615.png)

![image-20220523204218844](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523204218844.png)

可以双击放大查看网络具体情况



# 十四、损失函数和反向传播

如果你看过一点机器学习的推到的话，优化器会有一个优化函数，那个函数你可以定性的理解为最小二乘，不同的优化函数对应不同的损失函数，mse是比较经典用于回归的损失函数

## Loss Function

![image-20220524112553694](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220524112553694.png)



```py
import torch
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=float)

inputs = torch.reshape(inputs, [1, 1, 1, 3]) # batchsize=1 channel=1 1*3的pic
targets = torch.reshape(targets, [1, 1, 1, 3])

loss = L1Loss() # 默认是mean
# loss = L1Loss(reduction='sum')
loss1 = MSELoss()

res = loss(inputs, targets)
res1 = loss1(inputs, targets)
print(res,res1)
################
tensor(0.6667) tensor(1.3333, dtype=torch.float64)

Process finished with exit code 0
```



## 交叉熵

### 熵

熵表示一个模型里面的混乱程度。熵越大，代表系统的不确定性越大，混乱程度越高。

这里信息量的定义是：-log（p）

系统的熵：信息量*p之后再求和

![image-20220524164802738](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220524164802738.png)



系统的熵定义如下：

![image-20220525154914087](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220525154914087.png)

### 相对熵（KL散度）

通俗说法是相对熵用来衡量同一个随机变量的两个不同分布之间的距离。

相对熵是两个概率分布的不对称性度量，即

![img](https://bkimg.cdn.bcebos.com/formula/dc79c85b7e013878852a30d63ebdcf9f.svg)

在优化问题中，若

![img](https://bkimg.cdn.bcebos.com/formula/8026d7da8e67d58e75fd0be846ee8861.svg)

 表示[随机变量](https://baike.baidu.com/item/随机变量/828980)的真实分布，

![img](https://bkimg.cdn.bcebos.com/formula/54e824c54066bd91eccd5b38b7a36816.svg)

 表示理论或拟合分布，则

![img](https://bkimg.cdn.bcebos.com/formula/d5966c797d30ff5132cfdd90956dee66.svg)

 被称为前向KL散度（forward KL divergence），

![img](https://bkimg.cdn.bcebos.com/formula/31bb9a04529f40fba9e5625d2ef6a2c2.svg)

 被称为后项KL散度



KL散度如下：

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220525160725498.png)

pi是随机变量的概率。



由吉布斯不等式可知，KL散度是大于零的。也就是说交叉熵越小，模型越接近。

![image-20220525195503836](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220525195503836.png)



![image-20220525200351503](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220525200351503.png)













多分类的公式

![image-20220524113807231](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220524113807231.png)

第二个式子的括号里是一个softmax，然后对整个式子求一个1-hot的交叉熵



未化简的时候公式含义很明显，这个公式是把softmax函数和多分类交叉熵的式子叠加在一起写成这样子了 有不明白的可以分别搜下softmax 和交叉熵各自的公式。这个output相当于网络最终输出logits，然后输入到softmax，得到score，3个类别的score之和为1

这里是交叉熵，只不过这里简化了，默认了狗分类对应的标签值y是1，非狗的都是0，所以非狗的项都不需要写出来。而且这里还考虑了如果x不是概率值的话，这里也进行了归一计算。



```py
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3))
loss_cross = nn.CrossEntropyLoss()
res2 = loss_cross(x, y)
print(res2)

##############
tensor(1.1019)
```

这里没看懂 P23

？？？？？？？？？？？？？？？？？

# 十五、优化器

## TORCH.OPTIM的使用

链接https://pytorch.org/docs/stable/optim.html

### Constructing it

```py
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # lr是学习率 momentum是特定的优化器算法中设定的参数
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

### Per-parameter options

```py
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

### Taking an optimization step

```py
for input, target in dataset:
    optimizer.zero_grad()  # 把上一个循环中每个参数对应的梯度清零，不清零，梯度会累加
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()  # 反向传播，得到每个要更新参数对应的梯度
    optimizer.step()  # 每个参数会根据上一步得到的梯度进行优化
```



实例

```py
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./data_optim", train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset,batch_size=64)

class nn_optim(nn.Module):
    def __init__(self):
        super(nn_optim, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1, dilation=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)  ## 注意这里没有逗号
        )

    def forward(self, input):
        output = self.model1(input)
        return output

loss = nn.CrossEntropyLoss() # 交叉熵
nn_opt = nn_optim()
optim = torch.optim.SGD(nn_opt.parameters(), lr = 0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = nn_opt(imgs)
        loss_res = loss(output, targets)
        optim.zero_grad()
        loss_res.backward()
        optim.step()
        running_loss += loss_res
    print(running_loss)
    
###########
tensor(1672.1947, grad_fn=<AddBackward0>)
...
```



# 十六、现有模型的使用和修改

如何使用pytorch提供的网络模型，以及修改

## 加层

```py
import torchvision

# train_dataset = torchvision.datasets.ImageNet("./pretrain_dataset", split="train", download=True, transform=torchvision.transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 参数未初始化
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 参数已初始化

print(vgg16_true) # 可以看到网络的结构


######
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to C:\Users\zgliang/.cache\torch\hub\checkpoints\vgg16-397923af.pth
100.0%
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

Process finished with exit code 0
```



最后一层的out_features=1000，我们再加一层，分为10类

```py
import torchvision

# train_dataset = torchvision.datasets.ImageNet("./pretrain_dataset", split="train", download=True, transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 参数是初始化的，没有经过训练
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 参数已被训练

vgg16_true.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))
print(vgg16_true)

############# 最后多了一行的Linear
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
  (add_linear): Linear(in_features=1000, out_features=10, bias=True)
)

```



如果想加入到classifier里面的话

```py
vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))

###
...
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    (add_linear): Linear(in_features=1000, out_features=10, bias=True)
  )
)

Process finished with exit code 0
```

## 改层

```py
print(vgg16_false)

#####
...
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

Process finished with exit code 0
```

把最后一行的Linear里面的out_features改一下

```py
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_false)

####
...
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)
```

# 十七、网络模型的保存与修改

## 模型的保存

在moedl_save.py文件中

```py
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False) # 设置为False意思是这个网络模型的参数是没有经过训练的，是初始化的参数
# 保存方式一
# 方式一不仅保存了这个网络模型，而且保存了这个模型的参数
torch.save(vgg16, 'vgg16_method1.pth') # 引号里面的是保存路径， .pth是常用的后缀格式，写什么都可以
```

然后在model_load.py中：

```py
import torch
# import model_save

# 方式一-》保存方式一，加载模型
model = torch.load("vgg16_method1.pth")
print(model)
```

先运行moedl_save.py，会出现：

![image-20220720155107621](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220720155107621.png)



再运行model_load.py，把模型打印一下，会出现如下运行结果(这里显示的只是模型结构，其实参数也被保存下来了，可以设置断点debug看一下)：

```py
`VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

Process finished with exit code 0
```







pytorch看完之后学习：https://blog.csdn.net/weixin_42632271/article/details/107683469



# 十八、完整的模型训练套路（CIFAR10数据集为例）

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220523190926328.png)

## model.py

首先是model.py，搭建上图的神经网络

```py
# 搭建神经网络
import torch
from torch import nn
from torch.nn import Sequential


class ZL(nn.Module):
    def __init__(self):
        super(ZL, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2), #只设置kernelsize
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(), #展平
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# if __name__ == '__main__':
#     model1 = NN()
#     input = torch.ones((64,3,32,32))
#     output = model1(input)
#     print(output.shape)
```

## train.py

其次是train.py

```py
import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


# 添加tensordboard
writer = SummaryWriter("../logs_train")

# 靠，这里ToTensor忘记加括号，找了很久
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(), download=False)

train_data_size = len(train_data)
test_data_size = len(test_data)

#  print(f"训练数据集长度为{train_data_size}")
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))


# 利用DataLoader加载数据集
train_DataLoader = DataLoader(train_data, batch_size=64)
test_DataLoader = DataLoader(test_data, batch_size=64)

# 创建网络模型,NN网络定义在model.py文件中
zl = ZL()

# 损失函数
loss_fn = nn.CrossEntropyLoss() # 参数暂时用不到

# 优化器
learn_rate = 0.01
optimizer = torch.optim.SGD(params=zl.parameters(), lr = learn_rate) #第一个参数必须写模型的参数

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epoch = 10  # 训练的轮数

for i in range(epoch):  # 0-9
    print("------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    zl.train() # 这个对一些特定的层有作用，有这些层的话就一定要写上去，但是没有这些特定层的话写上也没问题，具体参见官网说明
    for data in train_DataLoader:
        imgs, targets = data
        outputs = zl(imgs)
        loss = loss_fn(outputs, targets)

        #优化器调优
        # 梯度清零、反向传播、参数优化、变量加一
        optimizer.zero_grad() #梯度清零
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 == 0:
            print("训练次数：{},Loss:{}".format(total_train_step, loss))  #更正规的写法是将loss改写为loss.item()
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 在测试集上测试当前的模型
    zl.eval() # 这个也只对特定的层有作用，有这些层的话就一定要写上去，但是没有这些特定层的话写上也没问题，具体参见官网说明
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_DataLoader:
            imgs, targets = data
            outputs = zl(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            accuracy = accuracy.item()  # 其实这一步无所谓的，因为.sum()已经转成普通数字类型了
            total_test_accuracy = total_test_accuracy + accuracy

    print(f"整体测试集的loss为{total_test_loss}")
    print(f"整体测试集的正确率为{total_test_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮（第i+1轮）的模型
    torch.save(zl, f"zl_{i+1}.pth")
    print("模型已保存")

writer.close()
```



关于train.py第63行item的解释为，在Scratches下建一个临时的test来测试,loss和loss.item()输出不一样

![image-20220905104615038](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220905104615038.png)

```py
import torch

a = torch.tensor(5)
print(a)
print(a.item())
#################
tensor(5)
5

Process finished with exit code 0
```





## 二分类的小例子

![image-20220905162244657](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220905162244657.png)

### test2.py

```py
import torch

outputs = torch.tensor([[0.1,0.2,0.3],
                        [0.3,0.4,0.6]])

print(outputs.argmax(1))  # .argmax参数写1的话，意思是比较横向比较大的坐标
####
tensor([2, 2])

Process finished with exit code 0
```

### test3.py(按照上面的图片说的)

```py
import torch

outputs = torch.tensor([[0.1,0.2],
                        [0.3,0.4]])

print(outputs.argmax(1))  # .argmax参数写1的话，意思是比较横向最大数的坐标
preds = outputs.argmax(1)
targets = torch.tensor([0,1])
print(preds == targets)

print((preds == targets).sum())  # False算0，True算1

#########
tensor([1, 1])
tensor([False,  True])
tensor(1)

Process finished with exit code 0
```

# 十九、使用GPU训练

## 方式一

train_gpu1.py



# 二十、完整的模型验证套路

## 利用训练好的模型，给它提供输入

## test.py

![image-20220906193049539](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220906193049539.png)![image-20220906200054738](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220906200054738.png)

dog.png跟test.py和上面十八保存的模型在同一个目录下。

```py
from PIL import Image

img_path = "./dog.png"
image = Image.open(img_path)
print(image)
####
结果如下图
```

![image-20220906192948871](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220906192948871.png)

下面是test.py

```py
import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "./dog.png"
image = Image.open(img_path)
print(image)

image = image.convert('RGB') # 因为png格式图片是四通道（r，g，b，透明度），当然如果图片本来就是三个颜色通道，经过此操作，不变。加上这一步后，可以适应png jpg各种格式的图片
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()]) # compose是把几个transform的操作链接在一起
image = transform(image)
print(image.shape)  # 结果是 torch.Size([3, 32, 32])

# 网络模型
class ZL(nn.Module):
    def __init__(self):
        super(ZL, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2), #只设置kernelsize
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(), #展平
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("zl_1.pth") # 同一个目录可以这么写.  另外：用方式1保存的模型要这样声明，方式2不用。
print(model)
image = image.reshape([1,3,32,32]) # 四维的原因[batch_num,channel,length,width]
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
######
<PIL.PngImagePlugin.PngImageFile image mode=RGB size=205x188 at 0x1CF6DAE77F0>
torch.Size([3, 32, 32])
ZL(
  (model): Sequential(
    (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=1024, out_features=64, bias=True)
    (8): Linear(in_features=64, out_features=10, bias=True)
  )
)
tensor([[-2.7927, -0.4143,  1.0657,  1.2183,  1.3533,  1.2238,  1.8717,  0.9010,
         -4.0387, -1.4674]])
tensor([6])

Process finished with exit code 0
```

对应的是第6个类别，预测错误哈哈，因为这是第一轮的模型，准确度还不怎么高。

![image-20220906203925599](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220906203925599.png)

