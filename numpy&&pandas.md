

# 0.简介

numpy是用c语言写的，运用到了矩阵的计算（比单独的一条条的计算来的快），计算速度比较快，pandas是在numpy的基础上写的。

# 1. numpy

```python
[[1,2,3],
 [2,3,4]]
```

这只是一个列表，我们要把它转化为numpy可以识别的矩阵的话

需要如下操作：

```py
array = np.array([[1,2,3],
				  [4,5,6]])
```

这时候就可以输出一些关于矩阵的属性。

```py
print('num of dim:', array.ndim)#num of dim:2
print('shape:', array.shape)#shape:(2,3)
print('size:', array.size)#size:6

ndim这里说的是几维数组，2X2 3X3的矩阵都是二维数组表示，不是线性代数里的几维矩阵，即“二维数组”中的“二”。
size指的是数据总数。
```

## 1.1 array

==注意==：经过array处理后的好像还是数组，不是矩阵，不过已经可以用好多矩阵的性质和运算符号了，具体参考下面的链接（自己还没仔细看）

https://blog.csdn.net/alxe_made/article/details/80492649

```py
import numpy as np

a = np.array([2,23,4], dtype = np.int)#第二个参数规定只可以是int类型的

print(a)
#输出结果是
#[2 23 4]

print(a.dtype)
#int64   意为64位的int格式（这是默认的），还可以就把np.int改为np.int32就改成了32位的int
```

## 1.2 zeros、empty、arange、linespace

```py
a = np.zeros((3,4))#定义三行四列的0矩阵

a = np.empty((3,4))#定义三行四列的空矩阵，其实是有数的，是多个非常接近于0的数，可以print来看一下，empty是占用的未初始化的内存，值是随机的

a = np.arange(10, 20, 2)
print(a)
#[10 12 14 16 18]

a = np.arange(12).reshape((3,4))#里面的那个括号代表以元组的形式存放行数和列数
print(a)
#输出
[[0 1 2 3]
 [4 5 6 7]
 [8 9 10 11]]
 
a = np.linspace(1, 10, 5)#1到10这个线段分成五个数(四段)
print(a)
#输出
[1.  3.25  5.5  7.75  10.]

a = np.linspace(1, 10, 6).reshape((2, 3))
print(a)
#输出
[[1.  2.8  4.6]
 [6.4  8.2  10.]]
```

### arange和linesapce官方文档

```py
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```

当`arange`与浮点参数一起使用时，由于有限的浮点精度，通常不可能预测所获得的元素的数量。出于这个原因，通常最好使用`linspace`函数来接收我们想要的元素数量的函数，而不是步长（step）：

```py
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
>>> f = np.sin(x)
```

## 1.3 numpy的基础运算

### 矩阵

```py
a = np.array([[1,1],
             [0,1]])
b = np.arange(4).reshape((2, 2))
print(a)
print('---')
print(b)
#输出
[[1 1]
 [0 1]]
---
[[0 1]
 [2 3]]
 
 c = a*b #矩阵对应元素相乘
 c_dot = a@b #矩阵的乘法，还可以写为c_dot = np.dot(a, b)
 #输出是：
 [[0 1]
 [0 3]]
 
[[2 4]
 [2 3]]
```

### 矩阵基本的运算

```py
a = np.random.random((2,4)) #random模块下的random方法
print(a)
print(np.sum(a))
print(np.min(a))
print(np.max(a))

[[0.42528475 0.37491425 0.43960813 0.59886992]
 [0.99654409 0.29671001 0.38396884 0.87791251]]
4.393812493024049
0.2967100119585907
0.9965440938788492


a = np.random.random((2,4))
print(a)
print(np.sum(a, axis=1)) #1代表行
print(np.min(a, 0)) #0代表列
print(np.max(a))
#输出：
[[0.67799966 0.51037881 0.92948068 0.73457376]
 [0.56426944 0.67852965 0.47073671 0.55296585]]
[2.85243291 2.26650164]
[0.56426944 0.51037881 0.47073671 0.55296585]
0.9294806783627417
```



```py
import numpy as np

A = np.arange(2,14).reshape((3,4))

print(A)
print(np.argmin(A)) #最小值的下标
print(np.argmax(A)) #最大值的下标
print(np.mean(A)) #均值，还可以写为print(A.mean())
print(np.median(A)) #中位数
print(np.cumsum(A)) #累加的过程
print(np.diff(A)) #相邻元素的差值，原数据为3*4，差值输出矩阵为3*3.
print(np.nonzero(A)) #nonzero输出的是每个非0值的位置，第一个元组是行，第二个元组是列
#输出：
[[ 2  3  4  5]
 [ 6  7  8  9]
 [10 11 12 13]]
0
11
7.5
7.5
[ 2  5  9 14 20 27 35 44 54 65 77 90]

[[1 1 1]
 [1 1 1]
 [1 1 1]]
 
 (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
```



![image-20220411185117227](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411185117227.png)

输出是：

![image-20220411185136258](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411185136258.png)

可以看出是sort是逐行的排序。

矩阵转置：

![image-20220411201937488](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411201937488.png)

![image-20220411201817636](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411201817636.png)

clip函数：

![image-20220411203112375](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411203112375.png)

A中小于5的等于5，大于9的等于9.

![image-20220411203146207](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411203146207.png)





![image-20220411204225665](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411204225665.png)

![image-20220411204237039](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220411204237039.png)

axis=0表示是对列进行mean均值计算。

## 1.4 numpy索引

```py
import numpy as np
A = np.arange(2,14).reshape((3,4))

print(A)
print(A[2])
print(A[2][1])
print(A[2,1])
print(A[2,:])#第二（3）行
print(A[:,1])#第一（2）列
#输出
[[ 2  3  4  5]
 [ 6  7  8  9]
 [10 11 12 13]]
[10 11 12 13]
11
11
[10 11 12 13]
[ 3  7 11]
```



```py
import numpy as np
A = np.arange(2,14).reshape((3,4))

print(A)

for i in A:
    print(i)#按行输出
    
for j in A.T:
    print(j)#按列输出，对矩阵转置的行输出就是对原矩阵的列输出

#输出：
[[ 2  3  4  5]
 [ 6  7  8  9]
 [10 11 12 13]]
[2 3 4 5]
[6 7 8 9]
[10 11 12 13]
[ 2  6 10]
[ 3  7 11]
[ 4  8 12]
[ 5  9 13]

```



```py
A = np.arange(2,14).reshape((3,4))

print(A)
print(A.flat)
print(A.flatten())
for item in A.flat: ##A.flat是一个迭代器
	print(item)
#输出
[[ 2  3  4  5]
 [ 6  7  8  9]
 [10 11 12 13]]
<numpy.flatiter object at 0x000002BD03C339C0>
[ 2  3  4  5  6  7  8  9 10 11 12 13]
2
3
4
5
6
7
8
9
10
11
12
13
```

## 1.5 array的合并

```py
import numpy as np

A = np.array([1,1,1])
B = np.array([2,2,2])

C = np.vstack((A,B)) #vertical stack
D = np.hstack((A,B)) #horizontal stack
print(A.shape)
print(C.shape)
print("-------")
print(C)
print(D)
#输出
(3,)
(2, 3)
-------
[[1 1 1]
 [2 2 2]]
[1 1 1 2 2 2]
```



```py
A = np.array([1,1,1])
print(A.T)
#输出
[1 1 1]

没有转置过来，因为一个数列是一维的 而矩阵是二维的 你没法通过转置提升数列的维度

解决办法1：reshape（（1，3））
办法2：
print(A[np.newaxis,:]) #在行上面加了一个维度，相当于变为了（1，3）的矩阵
A[:,np.newaxis] #在列上面加了一个维度，相当于变成了（3，1）的矩阵
```



多个array的合并：

```py
concatenate  "连接"
A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]

C = np.concatenate((A,B,B,A),axis=0) #等于0表示纵向合并
print(C)
#输出
[[1]
 [1]
 [1]
 [2]
 [2]
 [2]
 [2]
 [2]
 [2]
 [1]
 [1]
 [1]]
 
 C = np.concatenate((A,B,B,A),axis=1) #等于1表示横向合并
print(C)
 #输出
 [[1 2 2 1]
 [1 2 2 1]
 [1 2 2 1]]
```

## 1.6 array的分割

```py
A = np.arange(12).reshape((3,4))
print(A)
print(np.split(A,2,axis=1)) #第二个参数是分成几块，第三个是按照列来分块
#输出（竖向分割成两块）
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2,  3],
       [ 6,  7],
       [10, 11]])]
       
A = np.arange(12).reshape((3,4))
print(A)
print(np.split(A,3,axis=0)) #第二个参数是分成几块，第三个是按照行来分块
#输出（横向分割成三块）
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
```

如果想要进行不等的分割,使用array_split：

```py
A = np.arange(12).reshape((3,4))
print(A)
print(np.array_split(A,3,axis=1)) #总共四列进行三行的分割
print(np.array_split(A,(1,2,1),axis=1)) #可以按照1 2 1来分割
np.split(A,[3],axis=1)这样就可以分割成第0，1，2列和第3列，中间那部分列表看成索引
再比如np.split(A,[1,3],axis=1)就是分割成第0列，第1，2列和第三列，1跟3算是断点把0123分成了0 12 3（左闭右开）
而np.split(A,[1,4],axis=1)你会发现还是分为了三部分，0，123和一个空array
#输出：
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2],
       [ 6],
       [10]]), array([[ 3],
       [ 7],
       [11]])]
```



```py
A = np.arange(12).reshape((3,4))
print(A)
print(np.vsplit(A,3)) #纵向分成三块，但是刀是横着的
print(np.hsplit(A,2)) #横向分成两块，这里的参数2也可以用列表来表示索引

#输出
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
[array([[0, 1],
       [4, 5],
       [8, 9]]), array([[ 2,  3],
       [ 6,  7],
       [10, 11]])]
```

## 1.7 numpy的copy和deep copy

注意看下面这三种情况

```py
a=1
b=a
c=a
d=b

a=2
print(b)
print(d)
#输出
1
1
```



```py
a=np.array([1,2,3])
b=a
c=a
d=b

a=np.array([0,2,3])
print(b)
print(c)
print(d)
#输出
[1 2 3]
[1 2 3]
[1 2 3]
```



```py
#a就是b，c，d#

a=np.array([1,2,3])
b=a
c=a
d=b

a[0]=0
print(b)
print(c)
print(d)
#输出
[0 2 3]
[0 2 3]
[0 2 3]

d[1:3]=[22,33]
print(a)
#输出
[0 22 33]
```



```py
弹幕1：直接赋值是浅拷贝，是直接将新的变量指向同一内存地址，但是copy会进行一个较深度的复制，会开辟一个新的内存空间，将原有数据赋值进去，再将新的变量指向新的内存地址
弹幕2：都错了 在numpy中a=b 不拷贝 view()浅拷贝 copy()深拷贝
弹幕3：copy是浅拷贝，deepcopy是深拷贝
自己观点：copy是浅拷贝

a=np.array([1,2,3])
b=a.copy()
a[0]=0
print(a)
print(b)

#输出
[0 2 3]
[1 2 3]
```

# 2.pandas

## 2.1生成DataFrame

```py
下面的程序都默认导入了numpy和pandas
import numpy as np
import pandas as pd
```



```py
s = pd.Series([1,3,6,np.nan,44,1])
print(s)

#输出
0     1.0
1     3.0
2     6.0
3     NaN
4    44.0
5     1.0
dtype: float64
```



```py
dates = pd.date_range('20220425',periods=6)
print(dates)

#输出
DatetimeIndex(['2022-04-25', '2022-04-26', '2022-04-27', '2022-04-28',
               '2022-04-29', '2022-04-30'],
              dtype='datetime64[ns]', freq='D')
```



```py
dates = pd.date_range('20220425',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
#random模块下的randn是产生随机数，这些随机数服从正态分布


#输出
                   a         b         c         d
2022-04-25 -0.396696 -1.038431 -0.409615 -1.341674
2022-04-26  2.688878  0.575523 -1.321671 -1.306394
2022-04-27  1.942196 -2.137942 -0.118466  0.454236
2022-04-28  0.416243  1.048462  0.573706 -0.939586
2022-04-29 -0.357441 -0.685873 -0.288174 -0.294214
2022-04-30  0.380663  1.071531 -0.164563 -0.171182
```



```py
df1=pd.DataFrame(np.arange(12).reshape((3,4)))
print(df1)

#输出
   0  1   2   3
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
```



### 字典形式

```py
df2=pd.DataFrame({'A':1.,
                  'B':pd.Timestamp('20220425'),
                  		'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                  'D':np.array([3]*4,dtype='int32'),
                 'E':pd.Categorical(["test","train","test","train"]),
                 'F':'foo'})
print(df2)
print(df2.dtypes)
print(df2.index) #行
print(df2.columns) #列
print(df2.values) #值
print(df2.describe())
#输出
     A          B    C  D      E    F
0  1.0 2022-04-25  1.0  3   test  foo
1  1.0 2022-04-25  1.0  3  train  foo
2  1.0 2022-04-25  1.0  3   test  foo
3  1.0 2022-04-25  1.0  3  train  foo
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
Int64Index([0, 1, 2, 3], dtype='int64')
Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')
[[1.0 Timestamp('2022-04-25 00:00:00') 1.0 3 'test' 'foo']
 [1.0 Timestamp('2022-04-25 00:00:00') 1.0 3 'train' 'foo']
 [1.0 Timestamp('2022-04-25 00:00:00') 1.0 3 'test' 'foo']
 [1.0 Timestamp('2022-04-25 00:00:00') 1.0 3 'train' 'foo']]
         A    C    D
count  4.0  4.0  4.0
mean   1.0  1.0  3.0
std    0.0  0.0  0.0
min    1.0  1.0  3.0
25%    1.0  1.0  3.0
50%    1.0  1.0  3.0
75%    1.0  1.0  3.0
max    1.0  1.0  3.0
```



```py
df2.T #转置
#输出
                     0  ...                    3
A                  1.0  ...                  1.0
B  2022-04-25 00:00:00  ...  2022-04-25 00:00:00
C                  1.0  ...                  1.0
D                    3  ...                    3
E                 test  ...                train
F                  foo  ...                  foo
```



```py
df2.sort_index(axis=1)#1表示对列的名称进行排序
#输出
     A          B    C  D      E    F
0  1.0 2022-04-25  1.0  3   test  foo
1  1.0 2022-04-25  1.0  3  train  foo
2  1.0 2022-04-25  1.0  3   test  foo
3  1.0 2022-04-25  1.0  3  train  foo

df2.sort_index(axis=1，ascending=False)#第二个参数意为倒序排序
#输出
     F      E  D    C          B    A
0  foo   test  3  1.0 2022-04-25  1.0
1  foo  train  3  1.0 2022-04-25  1.0
2  foo   test  3  1.0 2022-04-25  1.0
3  foo  train  3  1.0 2022-04-25  1.0

df2.sort_index(axis=0,ascending=False)
#输出
     A          B    C  D      E    F
3  1.0 2022-04-25  1.0  3  train  foo
2  1.0 2022-04-25  1.0  3   test  foo
1  1.0 2022-04-25  1.0  3  train  foo
0  1.0 2022-04-25  1.0  3   test  foo
```



```py
print(df2.sort_values(by='E')) #按照E列排序
#输出
     A          B    C  D      E    F
0  1.0 2022-04-25  1.0  3   test  foo
2  1.0 2022-04-25  1.0  3   test  foo
1  1.0 2022-04-25  1.0  3  train  foo
3  1.0 2022-04-25  1.0  3  train  foo
```

## 2.2 pandas选择数据

```py
             A   B   C   D
2022-04-26   0   1   2   3
2022-04-27   4   5   6   7
2022-04-28   8   9  10  11
2022-04-29  12  13  14  15
2022-04-30  16  17  18  19
2022-05-01  20  21  22  23
```



```py
dates=pd.date_range('20220426',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D']) #别忘记加引号
print(df)
print(df['A']) #别忘记加引号
print(df.A)
#输出
             A   B   C   D
2022-04-26   0   1   2   3
2022-04-27   4   5   6   7
2022-04-28   8   9  10  11
2022-04-29  12  13  14  15
2022-04-30  16  17  18  19
2022-05-01  20  21  22  23
2022-04-26     0
2022-04-27     4
2022-04-28     8
2022-04-29    12
2022-04-30    16
2022-05-01    20
Freq: D, Name: A, dtype: int32
2022-04-26     0
2022-04-27     4
2022-04-28     8
2022-04-29    12
2022-04-30    16
2022-05-01    20
Freq: D, Name: A, dtype: int32
```



```py
切片（按行）
print(df[0:3],df['20220426':'20220429'])
不知道为什么后面的切片不是左闭右开了
###
            A  B   C   D
2022-04-26  0  1   2   3
2022-04-27  4  5   6   7
2022-04-28  8  9  10  11              A   B   C   D
2022-04-26   0   1   2   3
2022-04-27   4   5   6   7
2022-04-28   8   9  10  11
2022-04-29  12  13  14  15
```



标签形式

```py
#select by label:loc
print(df.loc['20220427'])
print(df.loc[:,['A','B']]) #保留所有行的信息
print(df.loc['20220426',['A','B']])
###
A    4
B    5
C    6
D    7
Name: 2022-04-27 00:00:00, dtype: int32

             A   B
2022-04-26   0   1
2022-04-27   4   5
2022-04-28   8   9
2022-04-29  12  13
2022-04-30  16  17
2022-05-01  20  21

A    0
B    1
Name: 2022-04-26 00:00:00, dtype: int32
```



```py
# select by position:iloc
print(df.iloc[3])#显示第3行的数据（从0开始计）
print(df.iloc[3,1])
print(df.iloc[3:5,1:3])#切片
print(df.iloc[[1,3,5],1:3])#不连续的筛选
###
A    12
B    13
C    14
D    15
Name: 2022-04-29 00:00:00, dtype: int32

13

             B   C
2022-04-29  13  14
2022-04-30  17  18

             B   C
2022-04-27   5   6
2022-04-29  13  14
2022-05-01  21  22
```



```py
# mixed selction:ix
# 把label和loc一起用
不过ix已经被弃用了，现在loc就可以标签位置双指了QwQ
```



```py
# Boolean indexing
print(df)
print(df[df.A>8])
###
             A   B   C   D
2022-04-26   0   1   2   3
2022-04-27   4   5   6   7
2022-04-28   8   9  10  11
2022-04-29  12  13  14  15
2022-04-30  16  17  18  19
2022-05-01  20  21  22  23
             A   B   C   D
2022-04-29  12  13  14  15
2022-04-30  16  17  18  19
2022-05-01  20  21  22  23
```

## 2.3 pandas设置值

```py
print(df)
df.iloc[1,1]=22225
print(df)

df.loc['20220427','B']=2222 #按照标签label修改
print(df)

df[df.A>4]=0 #这个好像只能用在列标上，行标不太行（A换成‘20220428’报错）
print(df)  ##这个命令把满足条件的所有列都改变了

将上面一行命令删掉用下面的替换
df.A[df.A>4]=0
print(df)  #实现了只对A这一列进行替换数据

df.B[df.A>4]=0 #把B列中df.A大于4的全改为0
print(df)

df['F']=np.nan #新增一列F
print(df)

df['E']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20220426',periods=6))
print(df)  #新增一列E，注意index要跟DataFrame的index保持一致
####
             A   B   C   D
2022-04-26   0   1   2   3
2022-04-27   4   5   6   7
2022-04-28   8   9  10  11
2022-04-29  12  13  14  15
2022-04-30  16  17  18  19
2022-05-01  20  21  22  23

             A      B   C   D
2022-04-26   0      1   2   3
2022-04-27   4  22225   6   7
2022-04-28   8      9  10  11
2022-04-29  12     13  14  15
2022-04-30  16     17  18  19
2022-05-01  20     21  22  23

             A     B   C   D
2022-04-26   0     1   2   3
2022-04-27   4  2222   6   7
2022-04-28   8     9  10  11
2022-04-29  12    13  14  15
2022-04-30  16    17  18  19
2022-05-01  20    21  22  23

            A     B  C  D
2022-04-26  0     1  2  3
2022-04-27  4  2222  6  7
2022-04-28  0     0  0  0
2022-04-29  0     0  0  0
2022-04-30  0     0  0  0
2022-05-01  0     0  0  0

            A     B   C   D
2022-04-26  0     1   2   3
2022-04-27  4  2222   6   7
2022-04-28  0     9  10  11
2022-04-29  0    13  14  15
2022-04-30  0    17  18  19
2022-05-01  0    21  22  23

             A     B   C   D
2022-04-26   0     1   2   3
2022-04-27   4  2222   6   7
2022-04-28   8     0  10  11
2022-04-29  12     0  14  15
2022-04-30  16     0  18  19
2022-05-01  20     0  22  23

             A     B   C   D   F
2022-04-26   0     1   2   3 NaN
2022-04-27   4  2222   6   7 NaN
2022-04-28   8     0  10  11 NaN
2022-04-29  12     0  14  15 NaN
2022-04-30  16     0  18  19 NaN
2022-05-01  20     0  22  23 NaN

             A     B   C   D   F  E
2022-04-26   0     1   2   3 NaN  1
2022-04-27   4  2222   6   7 NaN  2
2022-04-28   8     0  10  11 NaN  3
2022-04-29  12     0  14  15 NaN  4
2022-04-30  16     0  18  19 NaN  5
2022-05-01  20     0  22  23 NaN  6
```

## 2.4处理丢失数据

```py
初始化
import numpy as np
import pandas as pd

dates=pd.date_range('20220426',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])

df.iloc[0,1]=np.nan
df.iloc[1,2]=np.nan
print(df)
###
             A     B     C   D
2022-04-26   0   NaN   2.0   3
2022-04-27   4   5.0   NaN   7
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
```



下面的几个操作都是独立来对上面的df操作的，不是顺着操作下去的

```py
print(df.dropna(axis=0,how='any')) #dropna函数就是去掉带nan的行或者列，axis=0表示去掉带nan的行，how可以取any或者all

注意这个dropna只是临时隐藏掉数据，实际没有删除
###
             A     B     C   D
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
```



```py
print(df.dropna(axis=1,how='any'))
###
             A   D
2022-04-26   0   3
2022-04-27   4   7
2022-04-28   8  11
2022-04-29  12  15
2022-04-30  16  19
2022-05-01  20  23
```





```py
print(df.fillna(0)) # 替换nan为0

###
             A     B     C   D
2022-04-26   0   NaN   2.0   3
2022-04-27   4   5.0   NaN   7
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
             A     B     C   D
2022-04-26   0   0.0   2.0   3
2022-04-27   4   5.0   0.0   7
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
```



```py
print(df)
print(df.isnull())
###
             A     B     C   D
2022-04-26   0   NaN   2.0   3
2022-04-27   4   5.0   NaN   7
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
                A      B      C      D
2022-04-26  False   True  False  False
2022-04-27  False  False   True  False
2022-04-28  False  False  False  False
2022-04-29  False  False  False  False
2022-04-30  False  False  False  False
2022-05-01  False  False  False  False
```



```py
print(df)
print(np.any(df.isnull())==True)
###
             A     B     C   D
2022-04-26   0   NaN   2.0   3
2022-04-27   4   5.0   NaN   7
2022-04-28   8   9.0  10.0  11
2022-04-29  12  13.0  14.0  15
2022-04-30  16  17.0  18.0  19
2022-05-01  20  21.0  22.0  23
True
```

## 2.5 pandas导入导出数据

读取数据（各种格式的数据）

![image-20220504085626331](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220504085626331.png)



保存为文件

![image-20220504092816919](C:\Users\zgliang\AppData\Roaming\Typora\typora-user-images\image-20220504092816919.png)

```py
data=pd.read_csv('student.csv')
print(data)

data.to_pickle('student.pickle')
```

## 2.6 pandas合并concat

```py
# concatenating
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3=pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
print(df1)
print(df2)
print(df3)
print('-----------')

res=pd.concat([df1,df2,df3],axis=0) #0是竖向划分操作行，1是横向划分操作列
print(res)

###
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
     a    b    c    d
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
     a    b    c    d
0  2.0  2.0  2.0  2.0
1  2.0  2.0  2.0  2.0
2  2.0  2.0  2.0  2.0
-----------
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
0  2.0  2.0  2.0  2.0
1  2.0  2.0  2.0  2.0
2  2.0  2.0  2.0  2.0
```



```py
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3=pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
print(df1)
print(df2)
print(df3)
print('-----------')

res=pd.concat([df1,df2,df3],axis=0,ignore_index=True)
print(res)
####
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
     a    b    c    d
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
     a    b    c    d
0  2.0  2.0  2.0  2.0
1  2.0  2.0  2.0  2.0
2  2.0  2.0  2.0  2.0
-----------
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  1.0  1.0  1.0
4  1.0  1.0  1.0  1.0
5  1.0  1.0  1.0  1.0
6  2.0  2.0  2.0  2.0
7  2.0  2.0  2.0  2.0
8  2.0  2.0  2.0  2.0
```



```py
# join,['inner','outer']
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[0,1,2])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[1,2,3])
print(df1)
print(df2)
print('-----------')

res=pd.concat([df1,df2])默认是outer，相当于
#res=pd.concat([df1,df2]，join='outer')
print(res)
####
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
     b    c    d    e
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
3  1.0  1.0  1.0  1.0
-----------
     a    b    c    d    e
0  0.0  0.0  0.0  0.0  NaN
1  0.0  0.0  0.0  0.0  NaN
2  0.0  0.0  0.0  0.0  NaN
1  NaN  1.0  1.0  1.0  1.0
2  NaN  1.0  1.0  1.0  1.0
3  NaN  1.0  1.0  1.0  1.0
```

```py
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[0,1,2])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[1,2,3])
print(df1)
print(df2)
print('-----------')
res=pd.concat([df1,df2],join='inner')
print(res)

###
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
     b    c    d    e
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
3  1.0  1.0  1.0  1.0
-----------
     b    c    d
0  0.0  0.0  0.0
1  0.0  0.0  0.0
2  0.0  0.0  0.0
1  1.0  1.0  1.0
2  1.0  1.0  1.0
3  1.0  1.0  1.0
```

新版本的jupyter已经删除了join_axes，建议学习merge

pandas建议不用append，使用concat

```py
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2=pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
print(df1)
print(df2)
print('-----------')

res=df1.append(df2,ignore_index=True)
##res=df1.append([df2,df3],ignore_index=True)
print(res)


####
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
     a    b    c    d
0  1.0  1.0  1.0  1.0
1  1.0  1.0  1.0  1.0
2  1.0  1.0  1.0  1.0
-----------
	a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  1.0  1.0  1.0
4  1.0  1.0  1.0  1.0
5  1.0  1.0  1.0  1.0
```



```py
df1=pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
s1=pd.Series([1,2,3,4],index=['a','b','c','d'])
res=df1.append(s1,ignore_index=True)
print(res)
###
     a    b    c    d
0  0.0  0.0  0.0  0.0
1  0.0  0.0  0.0  0.0
2  0.0  0.0  0.0  0.0
3  1.0  2.0  3.0  4.0
```

## 2.7 pandas合并merge

```py
left=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
###
传入dataframe的是字典格式而字典格式本身是没有顺序的。但是IDLE可能做了一些处理使得列的出现是按照字典的顺序来的。
  key   A   B
0  K0  A0  B0
1  K1  A1  B1
2  K2  A2  B2
3  K3  A3  B3
  key   C   D
0  K0  C0  D0
1  K1  C1  D1
2  K2  C2  D2
3  K3  C3  D3
```



```py
left=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key':['K0','K1','K2','K3'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
res=pd.merge(left,right,on='key') # 这个on参数可以按照columns来merge，也可以按照index来merge
print(res)
###
  key   A   B
0  K0  A0  B0
1  K1  A1  B1
2  K2  A2  B2
3  K3  A3  B3
  key   C   D
0  K0  C0  D0
1  K1  C1  D1
2  K2  C2  D2
3  K3  C3  D3
  key   A   B   C   D
0  K0  A0  B0  C0  D0
1  K1  A1  B1  C1  D1
2  K2  A2  B2  C2  D2
3  K3  A3  B3  C3  D3
```



### consider two keys（how可以取left、right、inner、outer）

这里的left和right代表的数据名，inner和outer是方式

#### inner

```py
left=pd.DataFrame({'key1':['K0','K0','K1','K2'],
                   'key2':['K0','K1','K0','K1'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key1':['K0','K1','K1','K2'],
                   'key2':['K0','K0','K0','K0'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
res=pd.merge(left,right,on=['key1','key2']) # merge默认合并方式是inner,参数名是how
print(res)
####
  key1 key2   A   B
0   K0   K0  A0  B0
1   K0   K1  A1  B1
2   K1   K0  A2  B2
3   K2   K1  A3  B3
  key1 key2   C   D
0   K0   K0  C0  D0
1   K1   K0  C1  D1
2   K1   K0  C2  D2
3   K2   K0  C3  D3
  key1 key2   A   B   C   D
0   K0   K0  A0  B0  C0  D0
1   K1   K0  A2  B2  C1  D1
2   K1   K0  A2  B2  C2  D2
```

#### outer

```py
left=pd.DataFrame({'key1':['K0','K0','K1','K2'],
                   'key2':['K0','K1','K0','K1'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key1':['K0','K1','K1','K2'],
                   'key2':['K0','K0','K0','K0'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
res=pd.merge(left,right,on=['key1','key2'],how='outer')
print(res)

###
  key1 key2   A   B
0   K0   K0  A0  B0
1   K0   K1  A1  B1
2   K1   K0  A2  B2
3   K2   K1  A3  B3
  key1 key2   C   D
0   K0   K0  C0  D0
1   K1   K0  C1  D1
2   K1   K0  C2  D2
3   K2   K0  C3  D3
  key1 key2    A    B    C    D
0   K0   K0   A0   B0   C0   D0
1   K0   K1   A1   B1  NaN  NaN
2   K1   K0   A2   B2   C1   D1
3   K1   K0   A2   B2   C2   D2
4   K2   K1   A3   B3  NaN  NaN
5   K2   K0  NaN  NaN   C3   D3
```

#### left

```py
left=pd.DataFrame({'key1':['K0','K0','K1','K2'],
                   'key2':['K0','K1','K0','K1'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key1':['K0','K1','K1','K2'],
                   'key2':['K0','K0','K0','K0'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
res=pd.merge(left,right,on=['key1','key2'],how='left') #按照第一个left数据的columns来写的
print(res)

####
  key1 key2   A   B
0   K0   K0  A0  B0
1   K0   K1  A1  B1
2   K1   K0  A2  B2
3   K2   K1  A3  B3
  key1 key2   C   D
0   K0   K0  C0  D0
1   K1   K0  C1  D1
2   K1   K0  C2  D2
3   K2   K0  C3  D3
  key1 key2   A   B    C    D
0   K0   K0  A0  B0   C0   D0
1   K0   K1  A1  B1  NaN  NaN
2   K1   K0  A2  B2   C1   D1
3   K1   K0  A2  B2   C2   D2
4   K2   K1  A3  B3  NaN  NaN
```



#### right

```py
left=pd.DataFrame({'key1':['K0','K0','K1','K2'],
                   'key2':['K0','K1','K0','K1'],
                   'A':['A0','A1','A2','A3'],
                   'B':['B0','B1','B2','B3']})

right=pd.DataFrame({'key1':['K0','K1','K1','K2'],
                   'key2':['K0','K0','K0','K0'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']})
print(left)
print(right)
res=pd.merge(left,right,on=['key1','key2'],how='right')
print(res)

####
  key1 key2   A   B
0   K0   K0  A0  B0
1   K0   K1  A1  B1
2   K1   K0  A2  B2
3   K2   K1  A3  B3
  key1 key2   C   D
0   K0   K0  C0  D0
1   K1   K0  C1  D1
2   K1   K0  C2  D2
3   K2   K0  C3  D3
  key1 key2    A    B   C   D
0   K0   K0   A0   B0  C0  D0
1   K1   K0   A2   B2  C1  D1
2   K1   K0   A2   B2  C2  D2
3   K2   K0  NaN  NaN  C3  D3
```

#### indicator

```py
df1=pd.DataFrame({'col1':[0,1],
                   'col_left':['a','b']})
df2=pd.DataFrame({'col1':[1,2,2],
                   'col_right':[2,2,2]})
print(df1)
print(df2)
res=pd.merge(df1,df2,on='col1',how='outer',indicator=True)
print(res)

###
   col1 col_left
0     0        a
1     1        b
   col1  col_right
0     1          2
1     2          2
2     2          2
   col1 col_left  col_right      _merge
0     0        a        NaN   left_only
1     1        b        2.0        both
2     2      NaN        2.0  right_only
3     2      NaN        2.0  right_only
```



```py
#上面indicator参数的True改为indicator_column
res=pd.merge(df1,df2,on='col1',how='outer',indicator='indicator_column') #修改了列的名称
###
   col1 col_left
0     0        a
1     1        b
   col1  col_right
0     1          2
1     2          2
2     2          2
   col1 col_left  col_right indicator_column
0     0        a        NaN        left_only
1     1        b        2.0             both
2     2      NaN        2.0       right_only
3     2      NaN        2.0       right_only
```

#### merge by index

```py
left=pd.DataFrame({'A':['A0','A1','A2'],
                   'B':['B0','B1','B2']},
                   index=['K0','K1','K2'])
right=pd.DataFrame({'C':['C0','C1','C2'],
                   'D':['D0','D1','D2']},
                   index=['K0','K2','K3'])
print(left)
print(right)
res=pd.merge(left,right,left_index=True,right_index=True,how='outer') #这俩index参数默认是none，改为True之后就是按照index来merge
print(res)

###
     A   B
K0  A0  B0
K1  A1  B1
K2  A2  B2
     C   D
K0  C0  D0
K2  C1  D1
K3  C2  D2
      A    B    C    D
K0   A0   B0   C0   D0
K1   A1   B1  NaN  NaN
K2   A2   B2   C1   D1
K3  NaN  NaN   C2   D2
```

#### handle overlapping

```py
boys=pd.DataFrame({'k':['K0','K1','K2'],
                   'age':[1,2,3]})
girls=pd.DataFrame({'k':['K0','K0','K3'],
                   'age':[4,5,6]})
print(boys)
print(girls)
res=pd.merge(boys,girls,on='k',suffixes=['_boys','_girls'],how='outer')  #suffixes是后缀的意思
print(res)

####
    k  age
0  K0    1
1  K1    2
2  K2    3
    k  age
0  K0    4
1  K0    5
2  K3    6
    k  age_boys  age_girls
0  K0       1.0        4.0
1  K0       1.0        5.0
2  K1       2.0        NaN
3  K2       3.0        NaN
4  K3       NaN        6.0
```

outer改为inner

```py
boys=pd.DataFrame({'k':['K0','K1','K2'],
                   'age':[1,2,3]})
girls=pd.DataFrame({'k':['K0','K0','K3'],
                   'age':[4,5,6]})
print(boys)
print(girls)
res=pd.merge(boys,girls,on='k',suffixes=['_boys','_girls'],how='inner') 
print(res)

###
    k  age
0  K0    1
1  K1    2
2  K2    3
    k  age
0  K0    4
1  K0    5
2  K3    6
    k  age_boys  age_girls
0  K0         1          4
1  K0         1          5
```



