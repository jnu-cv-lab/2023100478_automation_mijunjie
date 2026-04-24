#include <opencv2/opencv.hpp>  
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // ================== 任务1: 读取测试图片 ==================
    Mat img = imread("test.jpg", IMREAD_COLOR); // 读取彩色图
    if (img.empty()) {
        cout << "无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // ================== 任务2: 输出图像基本信息 ==================
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    string dtype = "";
    switch (img.depth()) {
        case CV_8U:  dtype = "CV_8U"; break;
        case CV_8S:  dtype = "CV_8S"; break;
        case CV_16U: dtype = "CV_16U"; break;
        case CV_16S: dtype = "CV_16S"; break;
        case CV_32S: dtype = "CV_32S"; break;
        case CV_32F: dtype = "CV_32F"; break;
        case CV_64F: dtype = "CV_64F"; break;
        default: dtype = "未知类型";
    }
    cout << "图像尺寸: " << width << " x " << height << endl;
    cout << "通道数: " << channels << endl;
    cout << "像素数据类型: " << dtype << endl;

    // ================== 任务3: 显示原图 ==================
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", img);

    // ================== 任务4: 转换为灰度图并显示 ==================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY); // BGR转灰度
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray_img);

    // ================== 任务5: 保存灰度图 ==================
    imwrite("gray_test.jpg", gray_img);
    cout << "灰度图已保存为 gray_test.jpg" << endl;

    // ================== 任务6: NumPy风格的简单操作（用OpenCV+Mat实现） ==================
    // 示例1: 输出某个像素值 (以(100,100)为例)
    if (channels == 3) {
        Vec3b pixel = img.at<Vec3b>(100, 100);
        cout << "像素(100,100)的BGR值: (" 
             << (int)pixel[0] << ", " 
             << (int)pixel[1] << ", " 
             << (int)pixel[2] << ")" << endl;
    } else {
        uchar pixel = gray_img.at<uchar>(100, 100);
        cout << "像素(100,100)的灰度值: " << (int)pixel << endl;
    }

    // 示例2: 裁剪左上角100x100区域并保存
    Rect roi(0, 0, 100, 100); // 起始点(x,y) + 宽高
    Mat crop_img = img(roi);
    imwrite("crop_test.jpg", crop_img);
    cout << "左上角100x100区域已裁剪保存为 crop_test.jpg" << endl;

    // 等待按键后关闭窗口
    waitKey(0);
    destroyAllWindows();

    return 0;
}