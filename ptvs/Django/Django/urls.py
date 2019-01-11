"""Django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

#import app.forms
import app.views

urlpatterns = [
    path('admin/', admin.site.urls),
    #url(r'^$', app.views.home, name='home'), #Django 1.11.17
    path('', app.views.home, name='home'),
    #url(r'^contact$', app.views.contact, name='contact'),  #Django 1.11.17
    path('contact/', app.views.contact, name='contact'),
    #url(r'^about', app.views.about, name='about'),  #Django 1.11.17
    path('about/', app.views.about, name='about'),

    path('app/', app.views.index, name='index'),  # app 的 welcome
    path('hello/', app.views.hello, name='hello'),  # app 的 hello

    # 引用ajax中的urls
    # url(r'^ajax/',include('ajax.urls', namespace=ajax))
    path('aj/', include('ajax.urls'))

]
