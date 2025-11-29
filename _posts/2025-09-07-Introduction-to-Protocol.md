---
title: 无人机常见通信协议学习
date: 2025-09-07
categories: [无人机, 通信协议]
tags: [PX4, MAVLink, uORB, CAN, ROS]
math: false
---

## 一、学习过程简要总结

本次学习，系统梳理了 PX4 飞控、Gazebo 仿真、MAVLink 协议、uORB 消息中间件、CAN 总线、ROS topic 以及 UDP Socket 的原理、应用场景和实际开发方法。

## 二、各类通信方法具体介绍

### 1. uORB

- **定义**：PX4 飞控系统内部的轻量级发布-订阅消息机制，负责模块间的数据流转。
- **特点**：高效、实时、低资源消耗，仅用于飞控板内部。
- **应用场景**：传感器数据、控制器、导航等模块之间的数据交换。

### 2. MAVLink

- **定义**：无人机行业标准通信协议，定义了丰富的数据消息类型，适用于飞控与地面站、仿真器、伴飞电脑等外部系统之间的数据交换。
- **特点**：跨平台、跨设备、支持串口、UDP、TCP 等多种传输方式。
- **应用场景**：飞控与地面站、仿真器、伴飞电脑的数据同步与任务下发。

### 3. CAN 总线

- **定义**：高可靠性、抗干扰的物理总线通信协议，广泛用于汽车、工业、无人机等分布式硬件系统。
- **特点**：硬件级实时性强，广播型通信，支持多主机。
- **应用场景**：飞控与电调、舵机、传感器等外部硬件之间的数据交换。

### 4. ROS topic

- **定义**：ROS 框架下的发布-订阅消息机制，支持进程间、主机间的数据流转。
- **特点**：灵活可扩展，支持自定义消息类型，生态丰富。
- **应用场景**：机器人/无人机系统内部各模块间数据流转，如感知、导航、控制等。

### 5. UDP Socket

- **定义**：基础网络通信协议，支持进程间、主机间原始数据包传输。
- **特点**：无连接、轻量级、不保证可靠性，适合高效传输。
- **应用场景**：自定义协议、底层数据传输，或作为其他协议（如 MAVLink）的承载通道。

## 三、通信协议对比说明

| 通信机制 | 应用领域 | 通信机制类型 | 发布-订阅 | 话题支持 | 传输方式 | 适用场景 |
|---------|---------|-------------|----------|---------|---------|---------|
| uORB | PX4飞控内部 | 软件消息总线 | 是 | 有 | 内存共享 | 飞控模块间高效实时通信 |
| ROS topic | 机器人系统 | 软件消息总线 | 是 | 有 | TCP/UDP | 机器人软件模块间通信 |
| MAVLink | 无人机系统 | 消息协议 | 否 | 无 | 串口/UDP/TCP | 飞控与外部设备通信 |
| CAN总线 | 嵌入式系统 | 硬件总线 | 否 | 无 | 物理总线 | 分布式硬件设备通信 |
| UDP Socket | 通用网络 | 网络协议 | 否 | 无 | UDP | 自定义底层网络通信 |

## 四、代码举例

### 1. uORB 消息发布与订阅（PX4 C/C++）

**发布消息：**

```cpp
#include <uORB/uORB.h>
#include <uORB/topics/sensor_accel.h>

sensor_accel_s accel_data;
accel_data.timestamp = hrt_absolute_time();
accel_data.x = 0.1f;
accel_data.y = 0.2f;
accel_data.z = 9.8f;

orb_advert_t accel_pub = orb_advertise(ORB_ID(sensor_accel), &accel_data);
```

**订阅消息：**

```cpp
int accel_sub = orb_subscribe(ORB_ID(sensor_accel));
sensor_accel_s accel_data;
orb_copy(ORB_ID(sensor_accel), accel_sub, &accel_data);
printf("Accel X: %.2f\n", accel_data.x);
```

### 2. MAVLink 消息收发（Python pymavlink）

**发送心跳并接收心跳：**

```python
from pymavlink import mavutil

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
master.mav.heartbeat_send(
    mavutil.mavlink.MAV_TYPE_QUADROTOR,
    mavutil.mavlink.MAV_AUTOPILOT_PX4,
    0, 0, mavutil.mavlink.MAV_STATE_ACTIVE
)
msg = master.recv_match(type='HEARTBEAT', blocking=True)
print(f"Received heartbeat from system {msg.get_srcSystem()}")
```

### 3. CAN 总线消息发送（C，Linux SocketCAN）

```c
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
    int s = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    struct sockaddr_can addr;
    struct can_frame frame;

    addr.can_family = AF_CAN;
    addr.can_ifindex = if_nametoindex("can0");
    bind(s, (struct sockaddr *)&addr, sizeof(addr));

    frame.can_id = 0x101;
    frame.can_dlc = 2;
    frame.data[0] = 0x01;
    frame.data[1] = 100;

    write(s, &frame, sizeof(frame));
    close(s);
    return 0;
}
```

### 4. ROS topic 发布与订阅（Python）

**发布 IMU 数据：**

```python
import rospy
from sensor_msgs.msg import Imu

rospy.init_node('imu_publisher')
pub = rospy.Publisher('/imu/data', Imu, queue_size=10)
imu_msg = Imu()
imu_msg.linear_acceleration.x = 0.1
imu_msg.linear_acceleration.y = 0.2
imu_msg.linear_acceleration.z = 9.8

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    pub.publish(imu_msg)
    rate.sleep()
```

**订阅 IMU 数据：**

```python
import rospy
from sensor_msgs.msg import Imu

def callback(msg):
    print("Accel X:", msg.linear_acceleration.x)

rospy.init_node('imu_subscriber')
rospy.Subscriber('/imu/data', Imu, callback)
rospy.spin()
```

### 5. UDP Socket 通信（Python）

**发送端：**

```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b'Hello, UDP!', ('127.0.0.1', 8000))
```

**接收端：**

```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 8000))
data, addr = sock.recvfrom(1024)
print("Received:", data.decode())
```

## 五、学习总结

- uORB适合嵌入式飞控内部高效通信；
- MAVLink适合飞控与外部设备标准化数据交换；
- CAN总线适合分布式硬件系统高可靠实时通信；
- ROS topic适合机器人/无人机软件模块间灵活数据流转；
- UDP Socket则是最基础的网络数据传输方式。