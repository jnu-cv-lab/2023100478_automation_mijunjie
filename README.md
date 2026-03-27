# 📝 OpenCV 基础图像处理 Demo

这是一个基于 **C++ + OpenCV** 的入门级图像处理项目，实现了图像读取、信息打印、灰度转换、区域裁剪等基础操作，适合作为 OpenCV 入门练习。

## ✨ 功能实现

项目完成了以下 6 个核心任务：

1.  **读取测试图片**：使用 OpenCV 加载本地彩色图像
2.  **输出图像基本信息**：在终端打印图像尺寸（宽/高）、通道数、像素数据类型
3.  **显示原图**：通过 OpenCV 窗口展示原始图像
4.  **彩色转灰度图**：将彩色图像转换为单通道灰度图并展示
5.  **保存处理结果**：将生成的灰度图保存为新文件
6.  **NumPy 风格像素操作**：
    -   读取并打印指定像素的 BGR 值
    -   裁剪图像左上角区域并保存

## 🚀 快速开始

### 环境依赖
-   **操作系统**：Ubuntu / WSL Ubuntu
-   **编译器**：g++ (支持 C++17)
-   **OpenCV**：4.x 版本（需安装 `libopencv-dev`）

### 安装 OpenCV（Ubuntu/WSL）
```bash
sudo apt update
sudo apt install -y libopencv-dev
```

### 编译与运行
1.  克隆或下载本项目
2.  进入项目目录：
    ```bash
    cd /home/alexander/cv-course/202310/myproj2
    ```
3.  编译代码：
    ```bash
    g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
    ```
4.  运行程序：
    ```bash
    ./main
    ```

## 📂 项目结构


myproj2/
├── main.cpp          # 主程序代码
├── test.jpg          # 测试图片（需自行放置）
├── gray_test.jpg     # 生成的灰度图（运行后生成）
├── crop_test.jpg     # 裁剪后的区域图（运行后生成）
└── .vscode/          # VSCode 配置文件（可选）
    ├── c_cpp_properties.json
    ├── tasks.json
    └── launch.json


## 📊 输出示例

运行后终端会输出类似信息：

图像尺寸: 1920 x 1080
通道数: 3
像素数据类型: CV_8U
像素(100,100)的BGR值: (120, 135, 142)
灰度图已保存为 gray_test.jpg
左上角100x100区域已保存为 crop_test.jpg


## 🎯 适用场景

   OpenCV C++ 入门学习
   图像处理基础操作演示
   课程作业 / 实验报告参考

