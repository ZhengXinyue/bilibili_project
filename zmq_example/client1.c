#include <czmq.h>
#include <stdio.h>
#include <pthread.h>
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


void request_example() {
    void* socket = zmq_socket(context, ZMQ_REQ);
    zmq_connect(socket, "tcp://127.0.0.1:5555");

    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        char* request_message = "hello";
        zmq_send(socket, request_message, strlen(request_message) + 1, 0);

        // receive string
        char* s = calloc(100, sizeof(char));
        zmq_recv(socket, s, 100, 0);
        printf("receive reply message: %s\n", s);
        free(s);

        sleep(1);
    }
    zmq_close(socket);
}


void subscribe_example() {
    void* subscriber = zmq_socket(context, ZMQ_SUB);
//    zmq_setsockopt(subscriber, ZMQ_IDENTITY, "Hello", 5);
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "/student", 0);
    zmq_connect(subscriber, "tcp://127.0.0.1:8888");

    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        Student s;
        char* topic = calloc(10, sizeof(char));
        char* buffer = calloc(100, sizeof(char));
        zmq_recv(subscriber, topic, 10, 0);
        int recv_bytes = zmq_recv(subscriber, buffer, 100, 0);
        printf("receive %d bytes.\n", recv_bytes);
        deserialize(&s, buffer);
        free(buffer);
        free(topic);
        printf("student age: %d, name: %s, school: %s, grade: %d\n",
               s.age, s.name, s.school, s.grade);
        free(s.school);
    }
    zmq_close(subscriber);
}


void* pull_job() {
    //任务分发器和结果收集器是这个网络结构中较为稳定的部分, 因此应该由它们绑定至端点, 而非worker, 因为它们较为动态.
    void* worker_socket = zmq_socket(context, ZMQ_PULL);
    void* result_socket = zmq_socket(context, ZMQ_PUSH);
    zmq_connect(worker_socket, "tcp://127.0.0.1:5555");
    zmq_connect(result_socket, "tcp://127.0.0.1:5556");

    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        char* job_s = calloc(100, sizeof(char));
        zmq_recv(worker_socket, job_s, 100, 0);
        time_t current_time = time(NULL);
        printf("receive %s at time %ld\n", job_s, current_time);
        sleep(2);

        char* result = malloc(100);
        sprintf(result, "%s finished", job_s);
        zmq_send(result_socket, result, strlen(result), 0);

        free(job_s);
        free(result);
    }
    zmq_close(worker_socket);
}


void* pull_result() {
    void* result_socket = zmq_socket(context, ZMQ_PULL);
    zmq_bind(result_socket, "tcp://127.0.0.1:5556");
    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        char* result_s = calloc(100, sizeof(char));
        zmq_recv(result_socket, result_s, 100, 0);
        printf("%s\n", result_s);
        free(result_s);
    }
    zmq_close(result_socket);
}


void push_pull_example() {
    pthread_t thread1;
    pthread_t thread2;
    pthread_t thread3;
    pthread_t thread4;

    // three threads to work
    pthread_create(&thread1, NULL, pull_job, NULL);
    pthread_create(&thread2, NULL, pull_job, NULL);
    pthread_create(&thread3, NULL, pull_job, NULL);
    // one thread to collect result
    pthread_create(&thread4, NULL, pull_result, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);
}


void pull_subscribe_example() {
    void* subscriber = zmq_socket(context, ZMQ_SUB);
    zmq_connect(subscriber, "tcp://127.0.0.1:8888");
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "/student", 0);
    void* puller = zmq_socket(context, ZMQ_PULL);
    zmq_connect(puller, "tcp://127.0.0.1:5555");

    zmq_pollitem_t items[] = {
            {subscriber, 0, ZMQ_POLLIN, 0},
            {puller, 0, ZMQ_POLLIN, 0}
    };
    while (1) {
        if (s_interrupted) {
            printf ("terminating...");
            break;
        }
        zmq_poll(items, 2, -1);
        if (items[0].revents && ZMQ_POLLIN) {
            char* topic = calloc(100, sizeof(char));
            char* buffer = calloc(100, sizeof(char));
            zmq_recv(subscriber, topic, 100, 0);
            zmq_recv(subscriber, buffer, 100, 0);
            Student s;
            deserialize(&s, buffer);
            printf("student age: %d, name: %s, school: %s, grade: %d\n",
                   s.age, s.name, s.school, s.grade);
            free(s.school);
            free(topic);
            free(buffer);
        }
        if (items[1].revents && ZMQ_POLLIN) {
            char* job_s = calloc(100, sizeof(char));
            zmq_recv(puller, job_s, 100, ZMQ_NOBLOCK);
            time_t current_time = time(NULL);
            printf("receive %s at time %ld\n", job_s, current_time);
            free(job_s);
        }
    }
    zmq_close(subscriber);
    zmq_close(puller);
}


int main (void)
{
    context = zmq_ctx_new();

    subscribe_example();
//    request_example();
//    push_pull_example();
//    pull_subscribe_example();

    zmq_ctx_shutdown(context);
    zmq_ctx_term(context);
}

