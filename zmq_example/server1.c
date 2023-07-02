#include <czmq.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "utils.h"

void* context;


void reply_example() {
    void* socket = zmq_socket(context, ZMQ_REP);
    zmq_bind(socket, "tcp://127.0.0.1:5555");

    char buff[100];
    while (1) {
        zmq_recv(socket, buff, sizeof(buff) - 1, 0);
        printf("receive request message: %s\n", buff);

        // send string
        char* reply_message = "world";
        zmq_send(socket, reply_message, strlen(reply_message), 0);
    }
    zmq_close(socket);
}


void publish_example() {
    void* socket = zmq_socket(context, ZMQ_PUB);
    zmq_bind(socket, "tcp://127.0.0.1:5555");

    Student alex;
    alex.age = 0;
    strcpy(alex.name, "alex");
    alex.school = "nuaa";
    alex.grade = 1200;

    while (1) {
        void* buffer = malloc(100);
        unsigned long n_bytes = serialize(alex, buffer);
        printf("send %lu bytes.\n", n_bytes);
        zmq_send(socket, "student", 7, ZMQ_SNDMORE);
        zmq_send(socket, buffer, n_bytes, 0);
        free(buffer);
        alex.age += 1;;
        sleep(1);
    }
    zmq_close(socket);
}


void push_job() {
    void* socket = zmq_socket(context, ZMQ_PUSH);
    zmq_bind(socket, "tcp://127.0.0.1:5555");

    int count = 0;
    while (1) {
        char* message = malloc(100);
        sprintf(message, "job%d", count);
        zmq_send(socket, message, strlen(message), 0);
        usleep(1000000);   // sleep 1s
        count += 1;
    }
    zmq_close(socket);
}


int main (void)
// sudo apt-get install libczmq-dev
// sudo apt-get install libzmq3-dev
{
    context = zmq_ctx_new();
    reply_example();
//    publish_example();
//    push_job();

    zmq_ctx_shutdown(context);
    zmq_ctx_term(context);

//    void* context = zmq_ctx_new();
//    void* socket = zmq_socket(context, ZMQ_REP);
//    zmq_bind(socket, "tcp://127.0.0.1:5555");
//
//    uint32_t count = 0;
//    char buff[100];
//
//
//    Person p = {0, "person1"};
//    while (1) {
//        zmq_recv(socket, buff, sizeof(buff) - 1, 0);
//        printf("%s\n", buff);
//
//        // send string
//        char* reply_message = "reply message";
//        zmq_send(socket, reply_message, strlen(reply_message) + 1, 0);
//
//        // send uint32
////        zmq_send(socket, &count, sizeof(uint32_t), 0);
//
//        // sent Person struct
////        zmq_send(socket, &p, sizeof(p), 0);
////        p.age += 1;
//
//        count += 1;
//    }
//    return 0;
}

