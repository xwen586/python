# -*- coding: UTF-8 -*-
# app级urls 目录配置

from django.urls import path, include
from ajax import views


urlpatterns = [
    path('', views.index, name='index'),  #默认目录
    path('a01', views.step01, name='step01'),
    path('a01req', views.step01req),
    path('a02', views.step02, name='step02'),
    path('a02.jsp', views.step02req),
]
