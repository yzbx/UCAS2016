QT += core
QT -= gui

CONFIG += c++11

TARGET = sift
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    yzbx_sift.cpp

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv

HEADERS += \
    yzbx_sift.h
