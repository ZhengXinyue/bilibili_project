#include <iostream>
#include <QApplication>
#include <QtGui>
#include "mainwindow.h"


int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    mainwindow w;
    w.show();
    QApplication::connect(&app, SIGNAL(lastWindowClosed()), &app, SLOT(quit()));
    return QApplication::exec();
}
