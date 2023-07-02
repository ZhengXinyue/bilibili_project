#include <czmq.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>

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


void reply_example() {
    void* socket = zmq_socket(context, ZMQ_REP);
    zmq_bind(socket, "tcp://127.0.0.1:5555");

    char buff[100];
    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        zmq_recv(socket, buff, sizeof(buff) - 1, 0);
        printf("receive request message: %s\n", buff);

        // send string
        char* reply_message = "world";
        zmq_send(socket, reply_message, strlen(reply_message), 0);
    }
    zmq_close(socket);
}


void* publish_example() {
    void* publisher = zmq_socket(context, ZMQ_PUB);
//    uint64_t hwm = 10;
//    zmq_setsockopt(publisher, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    zmq_bind(publisher, "tcp://127.0.0.1:8888");

    Student alex;
    alex.age = 0;
    strcpy(alex.name, "alex");
    alex.school = "nuaa";
    alex.grade = 1200;

    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        void* buffer = malloc(100);
        unsigned long n_bytes = serialize(alex, buffer);
        printf("send %lu bytes.\n", n_bytes);
        zmq_send(publisher, "/student", 8, ZMQ_SNDMORE);
        zmq_send(publisher, buffer, n_bytes, 0);
        free(buffer);
        alex.age += 1;;
        sleep(1);
    }
    zmq_close(publisher);
}


void* push_job() {
    void* socket = zmq_socket(context, ZMQ_PUSH);
    zmq_bind(socket, "tcp://127.0.0.1:5555");

    int count = 0;
    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        char* message = malloc(100);
        sprintf(message, "job %d", count);
        zmq_send(socket, message, strlen(message), 0);
        usleep(1000000);   // sleep 1s
        count += 1;
    }
    zmq_close(socket);
}


void push_publish_example() {
    pthread_t thread1;
    pthread_t thread2;
    pthread_create(&thread1, NULL, publish_example, NULL);
    pthread_create(&thread2, NULL, push_job, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
}


int main (void)
// sudo apt-get install libczmq-dev
// sudo apt-get install libzmq3-dev
{
    s_catch_signals();
    context = zmq_ctx_new();

//    reply_example();
    publish_example();
//    push_job();
//    push_publish_example();

    zmq_ctx_shutdown(context);
    zmq_ctx_term(context);
}

