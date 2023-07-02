#include <czmq.h>
#include <stdio.h>

#include "zhelpers.h"
#include "utils.h"

void* context;


void subscribe_example() {
    void* socket = zmq_socket(context, ZMQ_SUB);
    zmq_connect(socket, "tcp://127.0.0.1:5555");
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "student", 0);

    while (1) {
        Student s;
        char* buffer = malloc(100);
        char* topic = malloc(10);
        zmq_recv(socket, topic, 10, 0);
        int recv_bytes = zmq_recv(socket, buffer, 100, 0);
        printf("receive %d bytes.\n", recv_bytes);
        deserialize(&s, buffer);
        free(buffer);
        free(topic);
        printf("student age: %d, name: %s, school: %s, grade: %d\n",
               s.age, s.name, s.school, s.grade);
        free(s.school);
    }
    zmq_close(socket);
}


int main (void)
{
    context = zmq_ctx_new();
    subscribe_example();

    zmq_ctx_shutdown(context);
    zmq_ctx_term(context);
    return 0;
}

