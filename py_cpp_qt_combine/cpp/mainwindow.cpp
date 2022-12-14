//
// Created by Zheng on 2022/5/14.
//

// You may need to build the project (run Qt uic code generator) to get "ui_mainwindow.h" resolved

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <windows.h>
#include <QString>
#include <QtCore>
#include <QtGui>
#include <QtWidgets/QWidget>
#include <iostream>
#include <fstream>


mainwindow::mainwindow(QWidget *parent) :
        QWidget(parent), ui(new Ui::mainwindow) {
    ui->setupUi(this);

//    const CHAR *name = "3D_Visualizer";
//    WId hwnd = (WId)FindWindowEx(0, 0, nullptr, name);

    std::ifstream file("winId.txt");
    if (!file.is_open())
    {
        std::cout << "open file failed" << std::endl;
    }
    std::string s;
    getline(file, s);
    std::cout<<s.c_str()<<std::endl;
    WId hwnd = (WId)(atoi(s.c_str()));
    QWindow *window = QWindow::fromWinId(hwnd);
    QWidget *my_widget = QWidget::createWindowContainer(window);
    ui->test_layout->addWidget(my_widget);
    this->setLayout(ui->test_layout);

    // 我的界面会通过下述代码生成一个winId, 写入winId.txt文件.
    // 通过上述代码读取winId, 生成widget, 嵌入到你的界面中
//    WId new_hwnd = this->winId();   // 写入txt文件.

}

mainwindow::~mainwindow() {
    delete ui;
}

