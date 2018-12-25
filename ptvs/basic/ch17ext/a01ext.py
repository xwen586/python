#!/usr/bin/env python
#-*- coding:UTF-8 -*-
# python 扩展
'''
Jython -- Java
IronPython -- C#、.net
'''

'''------java 代码-------
public class JythonTest {
	public void greeting() {
		System.out.println("Hello, Jython world!");
	}
}
设置环境变量：
set JAVA_HOME=C:\tools\Java\jdk1.8.0_92
set JYTHON_HOME=D:\tools\jython270
set PATH=%JAVA_HOME%\bin;%JYTHON_HOME%\bin;
编译: javac JythonTest.java
命令行调用
jython
>>>import JythonTest
>>>test = JythonTest()
>>>test.greeting()
'''

# Python 调用 java 程序
import jpype
import os

os.chdir(r'./ch17ext')

jvmPath = jpype.getDefaultJVMPath()
jpype.startJVM(jvmPath)
#jpype.java.lang.System.out.println("hello world!") 
javaClass = jpype.JClass('JythonTest')
t =  javaClass()
print( t.greeting() )

jObj = jpype.JClass('JavaObj')
t =  jObj( 'Tomcat' )
v = t.getValue()
print( v )
t.say()

jpype.shutdownJVM()
