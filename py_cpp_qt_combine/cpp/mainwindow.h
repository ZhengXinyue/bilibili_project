//
// Created by Zheng on 2022/5/14.
//

#ifndef C__QT_MAINWINDOW_H
#define C__QT_MAINWINDOW_H

#include <QWidget>


QT_BEGIN_NAMESPACE
namespace Ui { class mainwindow; }
QT_END_NAMESPACE

class mainwindow : public QWidget {
Q_OBJECT

public:
    explicit mainwindow(QWidget *parent = nullptr);

    ~mainwindow() override;

private:
    Ui::mainwindow *ui;
};


#endif //C__QT_MAINWINDOW_H
