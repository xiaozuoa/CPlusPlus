一.tips

### 多版本python共存

本地已经安装了python3.6版本，现在想要安装python3.9版本，来学习一下python新版本特性。

#### 下载安装python3.9

[官网下载](https://www.python.org/downloads/windows/)

下载成功后，点击安装，选择安装路径，我C盘大，安装在提前创建好的文件夹`C:\Python39`中，你可自行选择安装路径。安装中可以自动添加path, 前提你需要打勾，没打的话自己去也可以在环境变量中配置。

添加环境变量path：

#### 修改python名命

安装完成后，怎么区分3.6和3.9呢。

python3.6就不修改了，默认保持以前安装的设置。

这里主要是对 `python3.9` 修改，进入刚才的安装路径`C:\Python39`， 将文件下的 python.exe 改为 pyhon39.exe ，pythonw.exe 改为 pythonw39.exe

结果如图：

然后在CMD中输入 `python39` 能够进入python3.9解释器，则成功。

#### pip共存

python版本区分了，那如何区分各自的pip呢，使pip能够共存？

进入`C:\Python39\Scripts`文件下，把 pip.exe 改名为 pip39.exe

#### pip报错处理

然后在CMD中输入 `pip39 list`测试，报错: Fatal error in launcher

解决方法：重新下载pip

```css
python39 -m pip install --upgrade pip --force-reinstall
```

再输入`pip39 list`测试，如果还报错，进入`C:\Python39\Scripts`文件下，会发现多了一个 `pip.exe`，然后我们将之前修改的pip39.exe删掉，再将 pip.exe 改为 pip39.exe。

在CMD中输入 `pip39 list`测试，成功:

这样以后就可以使用`pip39 install `下载自己需要的python3.9模块包了。

以上。

### tar.gz

- gz是UNIX系统中的压缩文件，ZIP的Gnu版本，功能和WINRAR一样压缩文件的扩展名
- Unix和类Unix系统上的压缩打包工具，可以将多个文件合并为一个文件，打包后的文件后缀亦为“*tar*”



### 循环（for和while）的else

```py
//
i = 1
while i<5:
    print("循环内，i的值为",i)
    if i==2:
        break
    i += 1
else:
    print("循环外，i的值为",i)
//运行结果为：

循环内，i的值为 1
循环内，i的值为 2
```

![image-20220313180623411](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313180623411.png)

break跳出循环（**else也算是循环的一部分**）

这样设计，十分容易的看出循环的退出状况。

另一个例子：

```py
for n in range(2, 10):
    for x in range(2, n):
        if n%x == 0:
            print(n,"=",x,"*",n//x)
            break#这里的break跳出的是里层的for循环，直接进行下一个n的判断
    else:
        print(n, "是一个素数")
        
//运行结果如下：
2 是一个素数
3 是一个素数
4 = 2 * 2
5 是一个素数
6 = 2 * 3
7 是一个素数
8 = 2 * 4
9 = 3 * 3
```



### //和/

```py
6/3  #2.0
6//3  #2
```



### 引号

https://blog.csdn.net/woainishifu/article/details/76105667

### python标识符

不能以数字开头，以字母或者下划线开头

### import

import可以导入任何py文件（当然py文件不能以数字开头），然后可以使用被导入模块（文件）的函数和全局变量。并且被导入的模块（py文件）并不是一行一行解释执行的，python会把它编译为.pyc文件（字节码），成为一个二进制文件，直接编译执行。

### 大驼峰命名法（给类取名字）

每个单词的首字母大写（第一个单词也需要）；单词与单词之间没有下划线。

### 导入方式

1、import：所有导入的类使用时需加上模块名的限定。

2、 from XX import * ：所有导入的类不需要添加限定，直接用。

### print输出多个参数时，要用元组（加上括号）

![image-20220316100046967](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316100046967.png)

### python实例化的时候要传入__init__的所有参数，因为创建对象自动调用init方法

### python避免浮点数精度损失

https://zhuanlan.zhihu.com/p/62538963

### print自动换行

python的print会自动换行

```py
str = 'abcdefg'
for word in str:
    #print(word) # a/n b/n c/n d/n e/n f/n g/n
    print(each,end='') #输出abcdefg end=参数不设置，默认为末尾换行/n，end=''末尾为空所以不换行

```

### 成块注释

ctrl+/

### format {}格式化输出数据&&解包打包

https://www.runoob.com/python/att-string-format.html

#### *可以为元组或者列表解包，**为字典解包

如果字典解包一次，那么输出的就是键。

如果不解包，那么需要在{}中加一个0； 如果解包了，直接用下标即可。

```py
my_list = ['菜鸟教程', 'www.runoob.com']
my_tuple = ('菜鸟教程', 'www.runoob.com')
my_dir = {'a':"菜鸟教程", 'b':'www.runoob.com'}
print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的???
print("网站名：{0}, 地址 {1}".format(*my_list))

print("网站名：{0[0]}, 地址 {0[1]}".format(*my_tuple))

print("网站名：{0[0]}, 地址 {0[1]}".format(my_tuple))
print("网站名：{0}, 地址 {1}".format(*my_tuple))
print("网站名：{a}, 地址 {b}".format(**my_dir))
```

输出结果如下：

![image-20220322190747356](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322190747356.png)



![image-20220408214557731](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220408214557731.png)



![image-20220408214630886](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220408214630886.png)

![image-20220408214705094](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220408214705094.png)

![image-20220408214723023](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220408214723023.png)



__name__

### 1. __name__的理解

1.1 为什么使用__name__属性？

Python解释器在导入模块时，会将模块中没有缩进的代码全部执行一遍（模块就是一个独立的Python文件）。开发人员通常会在模块下方增加一些测试代码，为了避免这些测试代码在模块被导入后执行，可以利用__name__属性。

1.2 __name__属性。

__name__属性是Python的一个内置属性，记录了一个字符串。

- 若是在当前文件，__name__ 是__main__。

- - 在hello文件中打印本文件的__name__[属性值](https://www.zhihu.com/search?q=属性值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1654722335})，显示的是__main__

![img](https://pic4.zhimg.com/50/v2-ab797099d03e19a01348af6b84d86dc0_720w.jpg?source=1940ef5c)![img](https://pic4.zhimg.com/80/v2-ab797099d03e19a01348af6b84d86dc0_720w.jpg?source=1940ef5c)

- 若是导入的文件，__name__是模块名。

- - test文件导入hello模块，在test文件中打印出hello模块的__name__属性值，显示的是hello模块的模块名。

![img](https://pic2.zhimg.com/50/v2-f6141627bc99770b7b906e17dc6a9005_720w.jpg?source=1940ef5c)![img](https://pic2.zhimg.com/80/v2-f6141627bc99770b7b906e17dc6a9005_720w.jpg?source=1940ef5c)

因此__name__ == '__main__' 就表示在当前文件中，可以在if __name__ == '__main__':条件下写入测试代码，如此可以避免测试代码在模块被导入后执行。



# 二、列表（最后一个下标是-1）

- 列表可容纳不同类型的对象

## 1.列表-左闭右开

```py
rhyme = [1,2,3,4,5,"上山打老虎"]
rhyme[:3] #[1,2,3]
rhyme[3:] #[4,5,"上山打老虎"]
//只要是区间，左边端点是包括的，右边端点是不包括的（上面第三行这种：肯定是到达最后的啦）
rhyme[::-1] #直接倒叙输出 前面两个空是起点下标和终点下标 第三个数是步长
```

## 2.1列表-增

### - append、extend

![image-20220313192403289](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313192403289.png)

### - 切片加列表

![image-20220313192638244](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313192638244.png)

![image-20220313192856163](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313192856163.png)

## 2.2列表-删

![image-20220313193133075](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313193133075.png)

![image-20220313193149491](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313193149491.png)

- 如果列表中存在多个匹配的元素，那么remove只会删除第一个
- 如果指定元素不存在，那么程序会报错

![image-20220313193545715](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313193545715.png)

![image-20220313193619730](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313193619730.png)

## 2.3列表-改

![image-20220313194118395](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194118395.png)



![image-20220313194307808](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194307808.png)



![image-20220313194334725](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194334725.png)

## 2.4列表-查

![image-20220313194643349](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194643349.png)

- count（）查个数；

- index（）查下标；

  

  ![image-20220313194805406](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194805406.png)

index只会返回第一个下标，后面两个参数是开始下标和结束下标



![image-20220313194951050](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313194951050.png)

## 2.5列表运算

![image-20220313195146088](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313195146088.png)

## 2.6列表嵌套

![image-20220313195252517](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313195252517.png)





![image-20220313200045455](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313200045455.png)

print（）函数输出一个之后就自动换行。



![image-20220313200531249](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313200531249.png)



![image-20220313200726008](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313200726008.png)

上图B这种定义的写法会出现一些意想不到的bug



![image-20220313200923881](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313200923881.png)

这两个x和y都是指针，只是FishC只在计算机中存在一个，而列表则开辟了两个不同的空间。

![image-20220313201121492](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313201121492.png)

![image-20220313201208849](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313201208849.png)

A的第一个指针指错了。

## 2.7引用、拷贝

![image-20220313201537121](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313201537121.png)

这个就是引用





![image-20220313201635385](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313201635385.png)

copy方法不会修改到另一个



![image-20220313201734838](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313201734838.png)

切片方法跟上面的copy方法一样，都不会修改到另一个。但是这两种方法都被叫做浅拷贝。



**但是，到二维数组时，上面两种方法不灵了**

![image-20220313202242632](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313202242632.png)

这里修改一个，另一个也发生了改变。



导入copy模块

![image-20220313202456740](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313202456740.png)

用的是copy模块的copy方法，还是浅拷贝。



下面是深拷贝（使用copy模块的deepcopy方法）

![image-20220313202723596](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313202723596.png)

这是深拷贝。多层嵌套的话，深拷贝会全方位无死角的全部拷贝。

## 2.8列表推导式

列表的数都变为原来二倍，第一种方法：

![image-20220313204416080](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313204416080.png)





下面使用列表推导式

![image-20220313204754149](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313204754149.png)

列表推导式比第一种方法要快得多。



列表推导式的定义：

![image-20220313205052515](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313205052515.png)

![image-20220313205203007](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313205203007.png)



列表推导式转化为循环的写法：

![image-20220313205122525](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313205122525.png)





提取矩阵第二列：

![image-20220313205326424](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313205326424.png)

提取对角线元素：

![image-20220313205940893](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313205940893.png)

```py
//获取副对角线的元素
c = [b[i][len(b)-1-i] for i in range(len(b))]
```



列表推导式创建嵌套列表：

![image-20220313211200998](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211200998.png)



带if的列表推导式：

![image-20220313211325480](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211325480.png)



执行顺序如下：1，2，圈

![image-20220313211431898](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211431898.png)



![image-20220313211605788](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211605788.png)





![image-20220313211751038](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211751038.png)

相当于（上面是列表推导式写法，下面是循环写法）

![image-20220313211932856](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313211932856.png)





![image-20220313212634847](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313212634847.png)





![image-20220313212945014](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313212945014.png)



# 三、元组

- 元组定义用圆括号，且元组不可变

![image-20220313213359068](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313213359068.png)



- 元组也支持切片操作

![image-20220313213536122](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313213536122.png)

注意最后的跟reverse不一样，reverse是原地操作，而这个切片操作是重新导出了一个对象。



- 元组只让查，支持count（）和index（）方法
- 元组支持+  *  嵌套  迭代（举例如下）

![image-20220313214123623](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214123623.png)



- 可以对元组使用列表推导式，但是没有元组推导式

![image-20220313214251811](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214251811.png)



- 如何生成只有一个元素的元组（关键是逗号，tuple是元组的意思）

![image-20220313214420100](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214420100.png)



- 元组、列表、字符串的打包和解包：

  ![image-20220313214620854](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214620854.png)



![image-20220313214726158](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214726158.png)





- 元组中的元素是不可变的，但是如果元组中的元素是指向一个可变的列表，是可以改变的，如下：

  ![image-20220313214932697](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313214932697.png)

个人感觉是s和t都是引用，引用没发生改变，只是被引用指向的内存部分当中的数据发生了改变。



# 四、字符串



![image-20220314093957393](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314093957393.png)

方括号表示这为可选参数，可以有，也可以没有。



## 测试回文数

![image-20220313220439018](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313220439018.png)

## 大小写换来换去的函数

![image-20220313220623103](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313220623103.png)

字符串不可改变，这些函数之后都是重新生成一个新的字符串。

![image-20220313220837541](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313220837541.png)

分别是：首字母大写、全部变小写、每个单词的首字母大写、大小写反转、全部大写、全部小写。

其中casefold的小写能比lower的处理范围要多一点，lower只可以处理英文字母。



## 左中右对齐

![image-20220313221056191](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313221056191.png)

![image-20220313222332280](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313222332280.png)

![image-20220313222448175](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313222448175.png)



## 查找

![image-20220313223113348](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313223113348.png)

rfind是从右往左找下标。找不到的元素的话，find跟index也是找下标，但是找不存在的元素的话输出的结果不一样（如上）。

## 替换

![image-20220313223723054](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220313223723054.png)

code当中第一行使用的是tab，第二行是四个空格。expandtabs（4）是使用四个空格替换掉tab（制表符），注意这里code只是一个字符串。



![image-20220314093354264](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314093354264.png)

replace还有第三个参数（替换次数），默认为-1（全部替换）



![image-20220314093621311](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314093621311.png)

先用maketrans建立一个转换表格，再用translate对字符串进行替换。

也可以简写为：

![image-20220314093736077](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314093736077.png)



maketrans还有第三个参数，表示忽略掉这块：

![image-20220314093806760](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314093806760.png)

## startwith等函数

![image-20220314094248841](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314094248841.png)

endwith是左闭右开的。



![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314094449739.png)

istitle来判断每个单词的开头是否大写，isupper判断所有字母是否大写。



判断x字符串是否全是英文字母：

![image-20220314094630329](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314094630329.png)

返回False，因为空格不是字母。



![image-20220314094709939](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314094709939.png)

isspace判断是不是空白字符串，“换行，空格，tab”都属于是的。



## 判断是否是数字的函数

![image-20220314095029868](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314095029868.png)

## 截取字符串

![image-20220314101329213](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314101329213.png)





当截取掉参数指定的字符后，一直截取到没有指定的参数为止

![image-20220314101554275](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314101554275.png)

从字符串左边开始 ，看是否有w c o m . 这四个，一直到不是这四个的字符为止。



去除前缀prefix，后缀suffix

![image-20220314101757064](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314101757064.png)



切分为三元组：

![image-20220314102004776](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314102004776.png)



![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314104144415.png)

split默认切分空格隔开的，第二个参数是切分几次的意思（注意最后一个是把字符串切为了两部分）



splitlines可以将所有系统的换行符分割开来（包括\n \r 等）

![image-20220314104302078](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314104302078.png)



如果加上参数True，则把换行符也包含了进来。

![image-20220314104518043](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314104518043.png)



## 拼接join方法

![image-20220314104737265](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314104737265.png)

join方法效率比+拼接快。



## 格式化字符串format方法

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314105214772.png)



![image-20220314105313161](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314105313161.png)



想要单纯显示花括号：

![image-20220314105402529](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314105402529.png)



居中：

![image-20220314110101981](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314110101981.png)



![image-20220314110025676](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314110025676.png)

![image-20220314110045183](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314110045183.png)

![image-20220314110247898](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314110247898.png)



“=”强制在填充放置在符号（如果有）之后但在数字之前的位置

![image-20220314110716303](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314110716303.png)

![image-20220314111059990](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314111059990.png)



# 五、序列

列表 元组 字符串都属于序列。

分为可变序列（列表）和不可变序列（元组、字符串）。



简单的+ *：

![image-20220314112544433](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314112544433.png)



id函数

![image-20220314114315777](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314114315777.png)

这里的id应该是地址值，列表是可变的，所以直接在原地址上变化；而元组不可变，所以id发生了改变。



is    not is    in     not in

![image-20220314114529015](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314114529015.png)





del

![image-20220314114657592](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314114657592.png)



切片也可以实现上述的删除：

![image-20220314114731124](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314114731124.png)

![image-20220314115417926](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314115417926.png)



![image-20220314115502703](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314115502703.png)

```py
//直接del y的话就会把y列表给删除掉，而上面的则是y列表还在，只是清空了而已
del y
```



## 列表、元组、字符串的转化

分别是list（其他转为列表）、tuple、str

![image-20220314115730051](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314115730051.png)





## min、max

![image-20220314115940569](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314115940569.png)

![image-20220314120014845](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314120014845.png)



## len（）、sum（）

![image-20220314120138113](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314120138113.png)

len有最大值限制。sum有第二个参数。



## sorted（）和reversed（）

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314120412000.png)

sorted不是原地，sort是原地。



![image-20220314120546559](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314120546559.png)

第二个加key的是比较的len返回的结果。



注意：sort只可以处理列表方法，sorted可以处理列表，元组，字符串。



![image-20220314120901387](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314120901387.png)

reversed返回的是一个迭代器，这里统一用list来调用这个迭代器。

## all（）、any（）

![image-20220314133517673](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314133517673.png)

判断所有是否为真值，判断是否有一个为真值。

## enumerate（）



![image-20220314133636587](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314133636587.png)

后面这两步必须得分别执行，一下执行第三步是报错的。

还有一个开始参数：

![image-20220314133728446](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314133728446.png)



## zip（）函数和zip_longest()函数

![image-20220314141439376](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314141439376.png)

https://www.bilibili.com/video/BV1c4411e77t?p=36

看到8min，懒得看了。。



# 六、字典

## torch中补充的：

字典(Dictionary) get()方法：https://www.runoob.com/python/att-dictionary-get.html

字典(Dictionary) setdefault()方法：https://www.runoob.com/python/att-dictionary-setdefault.html

setdefault 如果不存在会在原字典里添加一个 key:default_value 并返回 default_value。

get 找不到 key 的时候不会修改原字典，只返回 default_value。







字典是python中唯一实现映射关系的内置类型。

摩斯密码：

![image-20220314143426002](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314143426002.png)

![image-20220314143512354](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314143512354.png)



字典形式：

![image-20220314151211705](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314151211705.png)





字典的定义形式：

![image-20220314151334292](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314151334292.png)

冒号左边叫做”键“，右边叫做”值“。

![image-20220314151511250](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314151511250.png)



### 创建字典

#### 形式一

上面花括号那种加上冒号的形式：

![image-20220314152424283](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314152424283.png)

#### 形式二

dict函数：

![image-20220314151935651](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314151935651.png)

即使键是字符串，但是这种定义方式中键不可以加引号。

#### 形式三

列表中的每个元素是使用元组包裹起来的键值对：

![image-20220314152152853](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314152152853.png)

#### 形式四

![image-20220314152239709](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314152239709.png)

#### 形式五

![image-20220314152316660](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314152316660.png)

#### 形式六

![image-20220314152401006](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314152401006.png)





![image-20220314155216975](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314155216975.png)

### 删

![image-20220314155317578](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314155317578.png)

一般使用pop函数，删除不存在的项时会报错，但是也可以指定提示词。



popitem（）是删除最后一个加入字典的键值对。

python3.7之后字典变得有序了。



del函数：

![image-20220314155625309](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314155625309.png)





clear函数：

![image-20220314155712302](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314155712302.png)

### 改

![image-20220314155941325](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314155941325.png)

https://www.bilibili.com/video/BV1c4411e77t?p=39不想看了

# 七、集合（没看完）

无序、唯一。

![image-20220314161821051](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314161821051.png)

注意s = set("FishC")，这里因为是小阔号不是花括号。





![image-20220314162142460](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314162142460.png)

因为集合会自动去重。





![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314162901079.png)

**自行判断**

# 八.函数

![image-20220314164906584](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314164906584.png)





![image-20220314170205803](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314170205803.png)

函数说明中的/  左边不能说明形参的名字（只能是位置参数），右边可以（比如start的关键字参数）

比如下面这种：

![image-20220314170556886](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314170556886.png)



*左边无所谓，星号的右边必须为关键字参数：

![image-20220314170657978](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314170657978.png)



### 收集参数（参数爱弄几个就弄几个）：

#### 打包为元组：

![image-20220314171016642](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314171016642.png)

**通过*将多个参数打包到一个元组里面。**







![image-20220314194156304](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314194156304.png)

通过这种方式来解包。





![image-20220314194239187](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314194239187.png)

表明为元组tuple。





![image-20220314172021009](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314172021009.png)

收集参数之外的参数，必须指明（关键字参数）。



#### **通过两个连续的*将参数打包为字典**：

注意：小甲鱼在字典构造那一节里留了一个悬念，使用dict构造字典时键不需要使用引号，在函数的时候会讲到。

![image-20220314195024789](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314195024789.png)





![image-20220314195345072](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314195345072.png)





![image-20220314195504179](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314195504179.png)

利用*进行解包，可以传入四个参数。



![image-20220314195603337](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314195603337.png)

利用**进行解包

# 九.作用域

正常的，局部变量跟全局变量重名时，优先使用局部变量：

![image-20220314200429132](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314200429132.png)



global作用域：

![image-20220314200257245](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314200257245.png)



## 嵌套函数：

![image-20220314200542872](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314200542872.png)

内部的函数funB无法直接被调用



## nonlocal语句

本来：

![image-20220314200938838](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314200938838.png)



调用nonlocal之后，就可以内部修改外部函数的值了：

![image-20220314201007895](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314201007895.png)



作用域：LEGB原则。

# 十、闭包closure

1.利用嵌套函数的外层作用域具有记忆能力这个特性（通过函数的嵌套，使外层函数返回内层函数名，赋值给新变量，新变量再调用内层函数，但外层函数的形成所带值不会消失，仍然可以被内层函数变量使用）

2.将内层函数作为返回值给返回



​		

![image-20220314202057046](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314202057046.png)

return函数时不用加括号，此时返回的时funB的一个引用，所以调用funB这么写：

![image-20220314202218351](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314202218351.png)





或者直接这么调用也行：

![image-20220314202312186](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314202312186.png)





仔细看下面这个：

![image-20220314203332600](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314203332600.png)

square指向的是exp_of函数，cube同理



![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314204022150.png)

内层函数的记忆作用，让数据保存在外部函数的参数中。





![image-20220314205525619](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314205525619.png)

超出边界就是撞墙反弹的效果。



# 十一、装饰器

将函数作为参数。

![image-20220314211306925](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314211306925.png)





![image-20220314211702996](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314211702996.png)

{.2f}当中表示小数点后保留两位，f表示是浮点型。





使用装饰器（**相当于把myfunc()这个函数作为一个参数塞到装饰器里面**）：

![image-20220314212024795](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314212024795.png)

相当于如下（就是timemaster内部定义的那块不看，直接就跳到最后一句，再进来跑callfunc）：

![image-20220314212937562](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314212937562.png)





多个装饰器可以用在同一个函数上（顺序是square-cube-add）：

![image-20220314213316276](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314213316276.png)





如何给装饰器传递参数：

![image-20220314214424369](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314214424369.png)



相当于：

![image-20220314215101468](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314215101468.png)



# 十二、lambda表达式

匿名函数



![image-20220314215511490](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314215511490.png)





![image-20220314215825099](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314215825099.png)

lambda表达式作第一个元素，y[1]当作表达式的参数。



![image-20220314220027165](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314220027165.png)



# 十三.生成器generator

![image-20220314221555385](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314221555385.png)

yield i与函数中的return i类似，返回上层程序并给出返回值，但是函数在return后不保存当前状态，生成器则是一个迭代器，每次引用时从上一次的结束状态开始运行

![image-20220314222038042](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314222038042.png)





元组的生成器表达式时一个生成器对象：

![image-20220314222205927](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220314222205927.png)

与列表推导式进行比较（列表推导式会一下子把所有数据弄出来，生成器表达式则是一个个的出来）。

# 十四、函数说明

![image-20220315103639015](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220315103639015.png)





之后看的黑马，参考讲义在D:\笔记\Python讲义

# 十五、类

类三要素：类名、属性、方法。

![image-20220316102003399](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316102003399.png)



![image-20220316115330750](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316115330750.png)

pass是保证程序的正确运行。



## 多态

java的多态：

```java
animal a = new cat();//animal是cat的父类
```

python

```py
class dog():
    def __init__(self,name):
        self.name = name
    def play(self):
        print("普通狗玩")

class xtq(dog):
    def play(self):
        print("%s哮天犬玩"% self.name)

class ren():
    def __init__(self, name):
        self.name = name

    def play_with_dog(self, dog):
        print("%s和%s玩"% (self.name, dog.name))
        dog.play()

d = dog("sss")
r = ren("rrr")
r.play_with_dog(d)

运行结果如下：

```

![](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316145334547.png)

这个执行的是父类中的play方法。





```py
`class dog():
    def __init__(self,name):
        self.name = name
    def play(self):
        print("普通狗玩")

class xtq(dog):
    def play(self):
        print("%s哮天犬玩"% self.name)

class ren():
    def __init__(self, name):
        self.name = name

    def play_with_dog(self, dog):
        print("%s和%s玩"% (self.name, dog.name))
        dog.play()

d = xtq("sss")
r = ren("rrr")
r.play_with_dog(d)

运行结果如下：
```

![image-20220316145425401](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316145425401.png)

这个执行的是子类xtq的play方法。





![image-20220316150903733](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220316150903733.png)





# 十六、类方法，静态方法

## 类方法

类方法可以使用类.方法名进行访问，上面加上@classmethod修饰符，第一个参数为cls（必须的），表明当前类

## 静态方法

静态方法不需要访问类方法、类属性，也不需要访问实例方法、实例属性，上面加上@staticmethod，没有默认参数



## ____new____方法

![image-20220317201004946](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317201004946.png)





![image-20220317202215574](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317202215574.png)new

new方法必做的事情是2分配空间和3返回对象的引用，这里2用的是object的new方法来分配空间，使用pycharm时候用new方法，自动弹出cls，*args（表示元组），**kwargs（表示字典）这三个参数，但是cls是不可或缺的。（其实这里new方法是静态方法，不知道为啥非要写一个cls参数）

弹幕解释（感觉第二条靠谱）：

静态方法传递的cls，是重写MusicPlayer类的new方法中的cls，是MusicPlayer这个类的引用；

__new__至少要有一个参数cls，代表要实例化的类，此参数在实例化时由Python解释器自动提供；



## 单例设计模式（只有一个对象）：

![image-20220317205539554](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317205539554.png)

![image-20220317205610749](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317205610749.png)





想要init只执行一次，找个标志位就可以了。



# 十七、异常

![image-20220317213020803](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317213020803.png)



![image-20220317213043246](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317213043246.png)

![image-20220317213315911](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317213315911.png)被选中的就是异常类型





### 捕获未知错误

![image-20220317213717046](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317213717046.png)

result是变量名，可以随便写。



![image-20220317215210000](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317215210000.png)





异常具有传递性，会一层层的抛出给调用者。



主动抛出异常：

![image-20220317221127475](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220317221127475.png)



# 十八、模块、包等

![image-20220318094007444](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220318094007444.png)

 外面是一个大的模块（不知道这么叫模块是否合适），可以包含python包（必须要有一个--init--.py的文件，这个文件下没有什么内容，文件名必须这么命名），也可以直接下括py文件（每个py文件就是一个模块）。

![image-20220318094413368](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220318094413368.png)

当另外一个导入包时，包中需要提供给外界的需要在init中提供声明。

# 十九、文件

![image-20220322150041228](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322150041228.png)

cpu想要使用文件，第一步先把文件加载到内存。文本文件和二进制文件本质都是二进制方式来存储的。



![image-20220322150334845](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322150334845.png)

![image-20220322150548461](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322150548461.png)

![image-20220322150812938](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322150812938.png)

![image-20220322151029159](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322151029159.png)

![image-20220322151117391](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322151117391.png)

分割线下面的就不能再打印出来了。。





![image-20220322151250875](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322151250875.png)

输出结果（上面的len是48，下面的len是0）：

![image-20220322151315313](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322151315313.png)



## 打开文件的方式

![image-20220322151435438](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322151435438.png)



本来README文件有这么多内容

![image-20220322152729772](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322152729772.png)

![image-20220322152745645](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322152745645.png)

运行之后README文件只剩下hello，所以w方式会覆盖

![image-20220322152804201](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322152804201.png)

a相当于append，直接在后面追加。



## 读取文件

![image-20220322153237792](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322153237792.png)





![image-20220322154825293](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322154825293.png)

![image-20220322154807947](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322154807947.png)



运行结果：

![image-20220322154857532](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322154857532.png)

注意：readline函数不可以读取中文。



### readline的换行问题

结论：readline读出来的东西应该是本行内容加上一个换行符



测试txt文件本身是这样

![image-20220322171020844](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322171020844.png)





```py
file = open(".idea文件说明.txt")
while True:
    text = file.readline()#readline读出来的东西应该是本行内容加上一个换行符
    if not text:
        break
    print(text)#print具有自动换行的效果
file.close()
```

结果是：

![image-20220322171002370](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322171002370.png)





```py
file = open(".idea文件说明.txt")
while True:
    text = file.readline()#readline读出来的东西应该是本行内容加上一个换行符
    if not text:
        break
    print(text, end="")
file.close()
```

结果是：

![image-20220322171156202](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322171156202.png)



```py
file = open(".idea文件说明.txt")
while True:
    text = file.readline()#readline读出来的东西应该是本行内容加上一个换行符
    if not text:
        break
    print(text, end="   ")
file.close()
```

结果是：

![image-20220322171248560](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322171248560.png)





### with as用法：

demo.txt内容就两行：

```py
35durant
teamGSW
```



```py
with open("demo.txt", "r") as f:
    data = f.read()
    print(data)
    print(type(data))
f.close()

output[1]:
35durant
teamGSW
<class 'str'>
```



## 文件操作

### 小文件复制

![image-20220322174337613](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322174337613.png)

### 大文件复制

![image-20220322175056396](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322175056396.png)

## 文件目录的常用管理操作

![image-20220322175220955](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220322175220955.png)



# 二十、定义

# 二十一、csv文件

![image-20220907192706918](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220907192706918.png)





![image-20220907193635210](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220907193635210.png)
