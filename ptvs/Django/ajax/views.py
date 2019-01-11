from django.shortcuts import render
from django.http import HttpResponse
from datetime import datetime
from django.views.decorators.csrf import csrf_protect

# Create your views here.
# http://localhost:8000/aj/
def index(request):
    return HttpResponse(u"<h3>Ajax 学习专区</h3> 欢迎光临 ! WellCome !")

# 例1：显示
# http://localhost:8000/aj/a01
def step01(request):
    # indx.html 一定要放在 templates 目录下
    return render(request, 'a01.html')


# 例1 的ajax 请求响应
#
def step01req(request):
    now = datetime.datetime.now()
    stime = "It is now： %s." % now
    return HttpResponse('Hello Ajax! 异步测试成功!<br/>' + stime)


# 例2：
# http://localhost:8000/aj/a02
def step02(request):
    return render(request, 'a02.html')


# 例2：ajax请求响应
# @csrf_protect
def step02req(request):
    method = request.method
    if method == "GET":
        firstName = request.GET["firstName"]
        birthday = request.GET["birthday"]
        msg = "GET: firstName=[" + firstName + "], your birthday is [" + birthday + "]"
    elif request.POST:  # method == "POST":
        firstName = request.POST["firstName"]
        birthday = request.POST["birthday"]
        msg = "POST: firstName=[" + firstName + "], your birthday is [" + birthday + "]"
    return HttpResponse(msg)


