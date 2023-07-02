#ifndef ZMQ_EXAMPLE_UTILS_H
#define ZMQ_EXAMPLE_UTILS_H
#include <stdint.h>


typedef struct {
    uint8_t age;
    char name[10];
    char* school;
    uint32_t grade;
} Student;


typedef struct {
    uint8_t age;
    char name[10];
} Person;


unsigned long serialize(const Student s, void* buffer) {
    // serialize the struct
    unsigned long n = 0;
    memcpy(buffer + n, &s.age, sizeof(s.age));
    n += sizeof(s.age);
    memcpy(buffer + n, s.name, strlen(s.name) + 1);
    n = n + strlen(s.name) + 1;
    memcpy(buffer + n, s.school, strlen(s.school) + 1);
    n = n + strlen(s.school) + 1;
    memcpy(buffer + n, &s.grade, sizeof(s.grade));
    n += sizeof(s.grade);
    return n;
}


void deserialize(Student* s, char* buffer) {
    // deserialize the struct
    unsigned long n = 0;
    memcpy(&s->age, buffer + n, sizeof(s->age));
    n += sizeof(s->age);

    strcpy(s->name, buffer + n);
    n = n + strlen(s->name) + 1;

    s->school = (char*)malloc(sizeof(char) * 100);
    strcpy(s->school, buffer + n);
    n = n + strlen(s->school) + 1;

    memcpy(&s->grade, buffer + n, sizeof(s->grade));
    n += sizeof(s->grade);
}

#endif //ZMQ_EXAMPLE_UTILS_H
