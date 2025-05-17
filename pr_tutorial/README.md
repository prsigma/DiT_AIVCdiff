# 从原始图像数据获取merge后的RGB图像
参考preprocess.ipynb进行预处理(去除了smiles为空的drug，将图像分辨率变成256*256，每一个药增强到1000张图片，并按照8:2的比例划分训练集和测试集)
metadata中augmentation_id为0的代表数据没有被增强

输出：
/data/pr/cellpainting/BBBC021/raw_data/merged_rgb_images_test
/data/pr/cellpainting/BBBC021/raw_data/merged_rgb_images_train
/data/pr/cellpainting/BBBC021/raw_data/metadata/augmented_image_metadata.csv

# 检查数据增强是否正确
参考visual_augmentation.ipynb

# 获取药物的扰动embedding(可以直接使用morphodiff提供的)
/data/pr/DiT_AIVCdiff/input/perturbation_embedding_bbbc021.csv






