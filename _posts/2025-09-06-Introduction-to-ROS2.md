---
title: ROS2 学习记录与实践
date: 2025-09-06
categories: [机器人, 操作系统]
tags: [ROS2, 机器人, 开发框架, 仿真]
math: false
---

> **Note**: 本文档系统梳理了我作为ROS初学者，从环境搭建、包开发、到仿真调试全过程。内容涵盖ROS 2基础架构、典型应用场景、开发流程、包结构、关键配置文件、代码示例及个人反思。

## 一、什么是ROS？

**ROS（Robot Operating System，机器人操作系统）**是一套专为机器人开发设计的中间件框架。它不是传统意义上的操作系统，而是一个支持模块化、分布式、复用性强的机器人软件开发平台。ROS核心特性包括：

- **节点（Node）**：独立功能模块或进程
- **话题（Topic）**：异步消息通信机制
- **服务（Service）**：同步请求-响应机制
- **参数（Parameter）**：动态配置机制
- **动作（Action）**：支持反馈和取消的异步长任务

## 二、ROS的典型应用场景

- **移动机器人**：自主导航、路径规划、避障
- **工业自动化**：机械臂控制、视觉识别
- **无人驾驶**：多传感器融合、环境感知、控制决策
- **科研与教学**：算法验证、仿真测试、系统集成
- **智能家居与服务机器人**：语音交互、目标跟踪

## 三、ROS的典型用法与开发流程

1. 环境搭建：安装ROS 2、配置依赖、准备仿真工具（如Gazebo）
2. 创建工作空间（workspace）：如`ros2_ws`
3. 开发功能包（package）：代码、配置、资源的基本单元
4. 编写节点（Node）：实现感知、规划、控制等算法逻辑
5. 配置包依赖与入口：`package.xml`、`setup.py`（Python）、`CMakeLists.txt`（C++）
6. 构建与安装：`colcon build`
7. 运行与调试：`ros2 run`、`ros2 topic echo`等
8. 仿真与可视化：用Gazebo、RViz等工具验证算法效果
9. 系统集成与协作：多节点间通过消息解耦协作

## 四、ROS 2 的典型框架与消息流

### 4.1 逻辑关系图

```
+-------------------+            +-------------------+            +-------------------+
|  感知节点         | --/scan--> |  控制/规划节点     | --/cmd_vel->|  机器人底层驱动   |
+-------------------+            +-------------------+            +-------------------+
```

- 感知节点负责采集/处理传感器数据并发布消息
- 控制/规划节点订阅感知数据，做决策后发布控制指令
- 机器人底层驱动节点接收控制指令，驱动实际运动

## 五、ROS 2 包的典型目录结构与文件说明

### 5.1 Python包结构示例

```
ros2_ws/
└── src/
    └── my_obstacle_avoidance/
        ├── my_obstacle_avoidance/
        │   └── obstacle_avoidance.py   # 算法与节点代码
        ├── package.xml                 # 包声明与依赖配置
        ├── setup.py                    # Python包安装与入口配置
        └── ...                         # 其他资源（如launch、test等）
```

### 5.2 C++包结构示例

```
ros2_ws/
└── src/
    └── my_obstacle_avoidance/
        ├── src/
        │   └── obstacle_avoidance.cpp  # 算法与节点代码
        ├── include/
        │   └── my_obstacle_avoidance/
        │       └── obstacle_avoidance.hpp # 头文件（可选）
        ├── package.xml                 # 包声明与依赖配置
        ├── CMakeLists.txt              # C++包编译与安装配置
        └── ...                         # 其他资源
```

### 5.3 关键文件作用

- **package.xml**  
  - 描述包的名称、版本、作者、依赖关系等元信息
  - 让ROS和colcon知道包之间的依赖关系和基本信息

- **setup.py**（Python包）
  - 定义Python包的安装方式、入口点（如节点主函数）、依赖等
  - 让colcon自动识别和安装Python节点

- **CMakeLists.txt**（C++包）
  - 定义C++包的编译规则、依赖、安装目标等
  - 让colcon自动调用CMake编译C++节点

- **src/**  
  - 存放你的主要算法代码（Python或C++）

- **include/**  
  - 存放C++头文件（如果有）

## 六、典型节点代码示例

### 6.1 Python节点示例

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.threshold = 0.4  # 距离阈值

    def listener_callback(self, msg):
        twist = Twist()
        center_index = len(msg.ranges) // 2
        window = 20
        forward_ranges = msg.ranges[center_index - window : center_index + window]
        valid_ranges = [r for r in forward_ranges if not math.isinf(r) and not math.isnan(r)]
        min_distance = min(valid_ranges) if valid_ranges else float('inf')
        self.get_logger().info(f"Min distance (front): {min_distance}")

        if min_distance < 0.15:
            twist.linear.x = -0.05  # 后退
            twist.angular.z = 0.5
        elif min_distance < self.threshold:
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # 左转
        else:
            twist.linear.x = 0.10  # 前进
            twist.angular.z = 0.0
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 6.2 C++节点示例

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"

class ObstacleAvoidanceNode : public rclcpp::Node
{
public:
    ObstacleAvoidanceNode() : Node("obstacle_avoidance")
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&ObstacleAvoidanceNode::scan_callback, this, std::placeholders::_1));
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        geometry_msgs::msg::Twist twist;
        float min_distance = *std::min_element(msg->ranges.begin(), msg->ranges.end());
        if (min_distance < 0.4) {
            twist.angular.z = 0.5;
            twist.linear.x = 0.0;
        } else {
            twist.linear.x = 0.2;
            twist.angular.z = 0.0;
        }
        publisher_->publish(twist);
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObstacleAvoidanceNode>());
    rclcpp::shutdown();
    return 0;
}
```

## 七、学习心得与关键反思

- **包结构和配置文件**是包管理和系统集成的基础，初学时需多实践和查阅官方文档。
- **角色分工明确**：感知、规划、控制各自专注节点开发，通过话题解耦协作。
- **ROS开发高度模块化**，只需关注节点的输入/输出消息和算法实现，极大提升了协作与复用效率。
- **仿真调试是必经环节**：Gazebo+RViz让算法验证更高效，遇到小车不动、倾倒等问题时，要善于用话题回显、日志输出等工具定位原因。
