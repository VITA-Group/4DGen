import os
import cv2
from argparse import ArgumentParser



def extract_frames(input_folder):
    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        print(f"文件夹 '{input_folder}' 不存在.")
        return

    # 遍历输入文件夹及其子目录下的所有文件
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".mp4"):
                video_path = os.path.join(root, filename)

                # 使用OpenCV打开视频文件
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 创建与视频文件同名的文件夹
                video_folder = os.path.join(root, os.path.splitext(filename)[0])
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                # 逐帧保存图片
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 生成保存图片的文件路径
                    frame_filename = os.path.join(video_folder, f"{i}.png")
                    print(frame_filename)
                    # 保存图片
                    cv2.imwrite(frame_filename, frame)

                # 释放视频文件
                cap.release()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    args = parser.parse_args()
    extract_frames(args.input_folder)
