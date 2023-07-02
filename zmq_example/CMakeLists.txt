cmake_minimum_required(VERSION 3.10)
project(zmq_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)


include_directories(

)

add_executable(server1 server1.c)
add_executable(client1 client1.c)
add_executable(client2 client2.c)

target_link_libraries(
        server1
        czmq
        zmq
)

target_link_libraries(
        client1
        czmq
        zmq
        pthread
)

target_link_libraries(
        client2
        czmq
        zmq
        pthread
)

