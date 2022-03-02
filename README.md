# Niuya's Page

---

### 基础知识
>张量：多维数组，0-n阶（维数，括号个数）  
>* 数据类型：tf.int tf.float tf.bool tf.string  
>* 创建张量：  
>    -  a = tf.constant(张量内容, dtype=数据类型)
>        - a.dtype 
>        - a.shape shape(2,3,4) 几个数字就是几维
>    -  numpy -> tensor:  b = tf.convert_to_tensor(数据内容, dtype)
>    -  c = tf.zeros(维度) d = tf.ones(维度) e = tf.fill(维度,指定值)
>        - 维度写法：一维直接写个数，二维以上[n,m,j,k....]
>    - 随机张量
>        - tf.random.normal(维度, mean=0.5, stddev=1)
>        - tf.random.truncated_normal(维度, mean=0.5, stddev=1) 更集中
>        - tf.random.unniform(维度, minval=最小值, maxval=最大值)   

>常用函数：
>    -  tf.cast(张量名, dtype=数据类型) 强制数据类型转换
>    -  求最小和最大 tf.reduce_min(张量名) tf.reduce_max(张量名)  
>    -  求平均和求和 tf.reduce_mean(张量名, axis=操作轴) tf.reduce_sum(张量名, axis=操作轴)
>        - 二维张量或数组中，axis=0表示对第一维度操作，纵向，经度；axis=1表示对第二维度，横向，维度；不设置则对所有元素  
>    -  tf.Variable() 将变量标记为"可训练"，被标记后会在反向传播中记录梯度信息
>    -  四则运算 tf.add(张量1,张量2) tf.subtract(张量1,张量2) tf.multiply(张量1,张量2) tf.divide(张量1,张量2)
>         -  只有维度相同才可以四则运算
>    -  矩阵运算 tf.matmul()
>    -  平方、次方、开方 tf.square(张量名) tf.pow(张量名，n) tf.sqrt(张量名)
>    -  标签与特征配对 tf.data.Dataset.from_tensor_slices((features,labels))
>    -  梯度 tf.GradientTape() 
>         - with结构记录计算过程，gradient求出张量对梯度
>             - with tf.GradientTape() as tape:   
>                   1.前向传播计算y 2.计算总loss   
>               grad=tape.gradient(函数,自变量)  
>    -  独热编码 tf.one_hot(待转换数据, depth=几分类) 将待转换数据以one-hot形式输出
>    -  softmax tf.nn.softmax(y) 使y的元素符合概率分布
>    -  自更新 w.assign_sub(w要自减的内容) 赋值操作，更新参数的值并返回
>         - 调用之前，需用tf.variable定义w为可训练(可自更新)
>    -  指定维度最大值索引 tf.argmax(张量名,axis=操作轴)

---

### 神经网络实现鸢尾花分类
>* 准备数据
>       - 数据集读入
>       - 数据集乱序
>       - 生成训练集，测试集
>       - x和y配对，读入batch
>* 搭建网络: 定义网络中可训练参数
>* 参数优化: 嵌套循环迭代,with结构更新参数，现实当前loss
>* 测试效果
>* acc/loss可视化

---

### 神经网络的优化
>* 预备知识  
>   - tf.where(条件语句,真返回A对应index上的值,假返回B对应index上的值) 
>   - np.random.RandomState.rand(维度) 返回[0,1)之间的随机数
>       - 设定随机种子 rdm = np.random.RandomState(seed=1)
>       - 设定维度 rdm.rand() rdm.rand(2,3)
>   - np.vstack(数组1,数组2) 将两个数组按垂直方向叠加
>   - 网格坐标点 
>       - np.mgrid[起始值:结束值:步长，起始值:结束值:步长，...]
>       = x.ravel() 将x变为一维数组
>       - np.c_[数组1,数组2，...] 使返回的间隔数值点配对
>   - 神经网络复杂度  
>       - 层数：隐藏层层数+1个输出层
>* 复杂学习率
>   - 先用较大学习率，快速得到最优解；再逐步减小,趋于稳定
>       - 指数衰减学习率 = 初始学习率 * 衰减率 **（当前轮数/多少轮衰减一次）
>* 激活函数
>   - sigmoid函数 tf.nn.sigmoid(x)
>       - 梯度0-0.25之间，容易梯度消失
>       - 输出0-1之间，非0均值，收敛慢
>       - 幂运算e-x复杂，训练时间长
>   - tanh函数 tf.math.tanh(x)
>       - 梯度0-1之间，容易梯度消失
>       - 输出0均值
>       - 幂运算e-x复杂，训练时间长
>   - Relu