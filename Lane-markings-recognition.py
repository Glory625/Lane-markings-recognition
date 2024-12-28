import numpy as np
import cv2
from moviepy import editor


def hough_transform(image):
    # 以像素为单位的距离r的精度，一般情况下是使用1，值越大,考虑越多的线
    rho = 1
    # 表示搜索可能的角度，使用的精度是np.pi/180
    theta = np.pi / 180
    # 阈值，该值越小，判定的直线越多，相反则直线越少
    threshold = 20
    # 默认为0，控制接受直线的最小长度
    minLineLength = 20
    # 控制接受共线线段的最小间隔，如果两点间隔超过了参数，就认为两点不在同一直线上，默认为0
    maxLineGap = 500
    # 函数返回一个数组，其中包含输入图像中出现的直线尺寸
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def region_selection(image):
    """
    确定并切割输入图像中的兴趣区域。
    参数:
        image: 在这里传递来自canny的输出，其识别了帧中的边缘
    """
    # 创建一个与输入图像大小相同的数组
    mask = np.zeros_like(image)
    # 如果传递多通道的图像
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # 掩膜多边形的颜色（白色）
        ignore_mask_color = 255
    # 创建一个多边形，只关注图片中的道路
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # 用白色填充多边形并生成最终的掩膜
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # 对输入图像和掩膜执行按位与操作，只获取道路上的边缘
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def average_slope_intercept(lines):
    """
    找出每幅图像左右车道的斜率和截距。
    参数:
        lines: 霍夫变换的输出
    """
    left_lines = []  # (斜率, 截距)
    left_weights = []  # (长度,)
    right_lines = []  # (斜率, 截距)
    right_weights = []  # (长度,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # 计算一条线的斜率
            slope = (y2 - y1) / (x2 - x1)
            # 计算一条线的截距
            intercept = y1 - (slope * x1)
            # 计算线的长度
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # 左车道的斜率是负的，右车道的斜率是正的
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    #
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    将每条线的斜率和截距转换为像素点。
        参数:
            y1: 线的起始点的y值。
            y2: 线的结束点的y值。
            line: 线的斜率和截距。
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    将像素点转换为完整长度的线条。
        参数:
            image: 输入测试图像。
            lines: 霍夫变换的输出线条。
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 255, 0], thickness=10):
    """
    在输入图像上绘制线条。
        参数:
            image: 输入测试图像（视频帧）。
            lines: 霍夫变换的输出线条。
            color (默认为红色): 线条颜色。
            thickness (默认为12): 线条粗细。
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def frame_processor(image):
    """
    处理输入帧以检测车道线。
    参数:
        image: 车道线的道路图像（将视频的帧传递给这个函数）
    """
    # 将RGB图像转换为灰度图像
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊，去除噪声并聚焦于兴趣区域
    # 高斯核的大小
    kernel_size = 5
    # 应用高斯模糊去除噪声
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    # 应用Canny边缘检测并将边缘保存在一个变量中
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result


def process_video(test_video, output_video):
    """
    读取输入视频流并生成带有检测车道线的视频文件。
    参数:
        test_video: 输入视频文件的位置
        output_video: 输出视频文件要保存的位置
    """
    # 使用VideoFileClip读取视频文件
    input_video = editor.VideoFileClip(test_video, audio=False)
    # 将函数"frame_processor"应用于视频的每个帧
    # "processed"存储输出视频
    processed = input_video.fl_image(frame_processor)
    # 将输出视频流保存为mp4文件
    processed.write_videofile(output_video, audio=False)


process_video('testvideo.mp4', 'output.mp4')

