#!/usr/bin/env python3
# a2scope.py
""" 作用域
分为 name_scope（命名空间），variable_scope（变量作用域）
name_scope：不同空间名，相同变量name
variable_scope：1)变量共享，2)tensorboard可视化封装变量
https://www.w3cschool.cn/tensorflow_python/tensorflow_python-61ue2ocp.html
"""
import tensorflow as tf


'''------- tf.name_scope（命名空间）-------
tf.name_scope(name, default_name=None, values=None)
与 tf.Variable()组合使用。
tf.get_variable() 不受name_scope约束，不存在组合。

注意，这里的 with 和 python 中其他的 with 是不一样的
执行完 with 里边的语句之后，这个 name1/ 和 name2/ 空间还是在内存中的。
这时候如果再次执行上面的代码就会再生成其他命名空间：name1_1, name2_1
'''
# 命名空间作用：不同空间名，相同变量name
def nsuse():
    with tf.name_scope('name1') as scope:
        weights1 = tf.Variable([1.0, 2.0], name='weights')
        #bias1 = tf.Variable([0.3], name='bias')
        bias1 = tf.get_variable(name='bias', shape=[1]) #不受name1约束

    with tf.name_scope('name2') as scope:
        weights2 = tf.Variable([4.0, 2.0], name='weights')
        bias2 = tf.Variable([0.33], name='bias')
        #bias2 = tf.get_variable(name='bias', shape=[1])

    #相同空间名，相同变量name时，空间名自动变更.
    with tf.name_scope('name2') as scope:
        weights3 = tf.Variable([4.0, 2.0], name='weights')

    print ("name_scope: 不同空间名，相同变量name")
    print (weights1.name)  # name1/weights:0
    print (weights2.name)  # name2/weights:0
    print (weights1 == weights2) # False 不同的变量
    print (bias1.name, " ", bias2.name) # bias:0  name2/bias:0
    print (weights3.name)  # name2_1/weights:0


'''------- tf.variable_scope（变量作用域）-------
与 tf.get_variable()结合使用，变量共享必须设置 reuse=True
tf.variable_scope(<scope_name>)

tf.get_variable(<name>, <shape>, <initializer>)
'''
def vsuse():
    with tf.variable_scope('v_scope') as scope1:
       Weights1 = tf.get_variable('Weights', shape=[2, 3])
       bias1 = tf.get_variable('bias', shape=[3])

    # note: 在下面的 scope 中的get_variable()变量必须已经定义过了，
    #才能设置 reuse=True，否则会报错
    with tf.variable_scope('v_scope', reuse=True) as scope2:
       Weights2 = tf.get_variable('Weights')
       Weights3 = tf.get_variable('Weights', [2, 3]) #shape如果不同会报错
    print (Weights2.name)  # v_scope/Weights:0
    print (Weights3.name)  # v_scope/Weights:0
    print (Weights1==Weights2, Weights1==Weights3)  #True True 都是同一个变量


# 创建一个新变量
def vsNew():
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"): # 等同于 "foo/bar"
            v = tf.get_variable("v", [1])
            w = tf.Variable([3.14], name='weights') 
            assert v.name == "foo/bar/v:0"
    print(v.name) # foo/bar/v:0
    print(w.name) # foo/bar/weights:0


# 共享变量AUTO_REUSE
def vsShare1():
    v1, w1 = foo()  # Creates v.
    v2, w2 = foo()  # Gets the same, existing v.
    assert v1 == v2
    print(v1.name, v2.name)  #foo/v:0 foo/v:0
    print(w1.name, w2.name)  #foo_1/weights:0 foo_2/weights:0

def foo():
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("v", [1])
        w = tf.Variable([3.14], name='weights') #会调整变量作用域名
    return v, w

# 共享变量:使用reuse=True
def vsShare2():
    #在foo()中已定义foo/v，并且变量名不会自动更新(如:foo_1),此处用foo2
    with tf.variable_scope("foo2"):
        v = tf.get_variable("v", [1])
    with tf.variable_scope("foo2", reuse=True):
        v1 = tf.get_variable("v", [1])
    #assert v1 == v
    print(v.name, v1.name)


# 共享变量：通过捕获范围并设置重用
def vsShare3():
    with tf.variable_scope("foo/egg") as scope:
        v = tf.get_variable("v", [1])
        scope.reuse_variables()
        v1 = tf.get_variable("v", [1])
    assert v1 == v
    print(v.name, v1.name)


# 为了防止意外共享变量,我们在获取非重用范围中的现有变量时引发异常.
def vsShareEx():
    try:
        with tf.variable_scope("foo"): # 不可重用
            v = tf.get_variable("v", [1])
            v1 = tf.get_variable("v", [1])
    except(ValueError) as ex:
        print("vsShareEx catch err1:", ex)

    try:
        with tf.variable_scope("fooE", reuse=True):  #未定义重用变量
            v = tf.get_variable("v", [1])
    except(ValueError) as ex:
        print("vsShareEx catch err2:", ex)



if __name__=='__main__':
    #nsuse();  print('-'*20)
    #vsuse();  print('-'*20)
    vsNew();     print('-'*20)
    vsShare1();  print('-'*20)
    vsShare2();  print('-'*20)
    vsShare3();  print('-'*20)
    vsShareEx();  print('-'*20)
