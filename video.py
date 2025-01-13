import imageio
import imageio_ffmpeg as ffmpeg
import os
import imageio.v2 as imageio
def get_sorted_images(folder_path):
    # 获取所有.jpg文件的路径
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".png")]
    
    # 按文件名排序（假设文件名格式为000001.jpg, 000002.jpg, ...）
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    return images
    
def images_to_mp4(image_paths, output_path, fps=30):
    # 使用imageio将图片序列写成mp4
    writer = imageio.get_writer(output_path, fps=fps)

    for image_path in image_paths:
        image = imageio.imread(image_path)
        writer.append_data(image)

    writer.close()
folder_path = 'result/output_12_20_wdepth/00000/novel_view/ours_50000/renders'  # 替换为你图片所在文件夹的路径
output_path = 'result/output_12_20_wdepth/00000//novel_view.mp4'  # 输出视频的路径和文件名

# 获取按顺序排序的图片路径
image_paths = get_sorted_images(folder_path)

# 将图片转换为mp4
images_to_mp4(image_paths, output_path, fps=20)
