#include <czmq.h>
#include <stdio.h>
#include <signal.h>

#include "zhelpers.h"
#include "utils.h"

void* context;

static int s_interrupted = 0;
static void s_signal_handler(int signal_value) {
    s_interrupted = 1;
}


static void s_catch_signals (void) {
    struct sigaction action;
    action.sa_handler = s_signal_handler;
    action.sa_flags = 0;
    sigemptyset (&action.sa_mask);
    sigaction (SIGINT, &action, NULL);
    sigaction (SIGTERM, &action, NULL);
}


void subscribe_example() {
    void* socket = zmq_socket(context, ZMQ_SUB);
    zmq_connect(socket, "tcp://127.0.0.1:5555");
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "student", 0);

    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
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

