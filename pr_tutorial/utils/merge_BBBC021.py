import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import hashlib
import random
import shutil

# 定义处理单个图像的函数
def process_image(paths, root_dir, save_dir_train):
    channel_images = {}
    
    for channel, image_path in paths.items():
        if channel in ['idx', 'path_name', 'dapi_filename', 'split', 'smiles', 'moa']:
            continue
            
        try:
            # 读取tif文件
            img = Image.open(image_path)
            # 转换为numpy数组
            img = np.array(img, dtype=np.float32)
            
            # 使用OpenCV调整大小
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            # 归一化并转为8位
            img -= img.min()
            img = img / (img.max() - img.min()) 
            img = (img * 255).astype(np.uint8)
            
            channel_images[channel] = img
            
        except Exception as e:
            print(f"读取 {image_path} 时出错: {e}")
            channel_images[channel] = np.zeros((256, 256), dtype=np.uint8)
    
    # 将通道合并为RGB图像
    blue_channel = channel_images['dapi']    #DAPI 对核进行染色，使用蓝色通道
    green_channel = channel_images['tubulin']   #微管蛋白，使用绿色通道
    red_channel = channel_images['actin']   #使用红色通道
    
    # 创建RGB图像
    rgb_img = np.zeros((blue_channel.shape[0], blue_channel.shape[1], 3), dtype=np.uint8)
    rgb_img[:,:,0] = red_channel    # 红色通道
    rgb_img[:,:,1] = green_channel  # 绿色通道
    rgb_img[:,:,2] = blue_channel   # 蓝色通道
    
    # 生成保存文件名
    save_filename = f"image_{paths['idx']}.png"
    
    # 保存到训练目录
    save_path = os.path.join(save_dir_train, save_filename)
    plt.imsave(save_path, rgb_img)
    
    # 返回索引和保存路径
    return {
        'idx': paths['idx'],
        'merged_image': save_path,
        'smiles': paths.get('smiles', None),
        'moa': paths.get('moa', None)
    }

def augment_image(image_path, idx, augmentation_id, save_dir):
    """
    对单个图像进行增强
    
    参数:
    - image_path: 原始合并后的RGB图像路径
    - idx: 新的图像索引
    - augmentation_id: 增强ID，决定使用哪种增强方式
    - save_dir: 保存目录
    
    返回:
    - 增强后的图像路径
    """
    try:
        # 读取原始图像
        rgb_img = plt.imread(image_path)
        
        # 应用旋转
        rotation = (augmentation_id % 4) * 90
        if rotation > 0:
            rgb_img = np.rot90(rgb_img, k=rotation//90)
        
        # 应用水平翻转
        if (augmentation_id // 4) % 2 == 1:
            rgb_img = np.fliplr(rgb_img)
            
        # 应用垂直翻转
        if (augmentation_id // 8) % 2 == 1:
            rgb_img = np.flipud(rgb_img)
        
        # 保存增强后的图像
        save_filename = f"image_{idx}.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.imsave(save_path, rgb_img)
        
        return save_path
    except Exception as e:
        print(f"增强图像 {image_path} 时出错: {e}")
        return None

# 处理单个增强任务的函数
def process_augmentation(aug_params):
    """
    处理单个增强任务
    
    参数:
    - aug_params: 包含增强参数的元组
    
    返回:
    - 增强后的图像元数据或None
    """
    orig_path, aug_idx, aug_id, save_dir, compound, orig_idx, smiles, moa = aug_params
    
    # 设置实际的增强ID (1-15)
    actual_aug_id = aug_id % 15 + 1
    
    # 增强图像
    augmented_path = augment_image(
        orig_path, 
        aug_idx, 
        actual_aug_id, 
        save_dir
    )
    
    if augmented_path:
        return {
            'augmented_idx': aug_idx,
            'original_idx': orig_idx,
            'augmentation_id': actual_aug_id,
            'merged_image': augmented_path,
            'split': 'train',  # 初始设置为train，之后再划分
            'smiles': smiles,
            'moa': moa,
            'compound': compound
        }
    return None

def main():
    # 设置保存路径
    save_dir_train = "/data/pr/cellpainting/BBBC021/raw_data/merged_rgb_images_train"
    save_dir_test = "/data/pr/cellpainting/BBBC021/raw_data/merged_rgb_images_test"
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    # 设置随机种子确保可复现
    np.random.seed(42)
    random.seed(42)

    # 加载metadata
    image_info = pd.read_csv('/data/pr/cellpainting/BBBC021/BBBC021_v1_image.csv')
    
    # 加载化合物信息
    compound_info = pd.read_csv('/data/pr/cellpainting/BBBC021/BBBC021_v1_compound.csv')

    # 加载moa信息
    moa_info = pd.read_csv('/data/pr/cellpainting/BBBC021/BBBC021_v1_moa.csv')
    
    # 查看两个数据集的列名，确定合并键
    print(f"图像数据集列名: {image_info.columns.tolist()}")
    print(f"化合物数据集列名: {compound_info.columns.tolist()}")
    print(f"moa数据集列名: {moa_info.columns.tolist()}")
    
    # 假设合并键是'Image_Metadata_Compound'和'compound'
    # 根据实际列名调整
    merge_key_image = 'Image_Metadata_Compound'
    merge_key_compound = 'compound' if 'compound' in compound_info.columns else 'Compound'
    
    # 合并数据集
    merged_info = pd.merge(
        image_info, 
        compound_info, 
        left_on=merge_key_image, 
        right_on=merge_key_compound,
        how='left'
    )

    merged_info = pd.merge(
        merged_info,
        moa_info,
        left_on=merge_key_image,
        right_on='compound',
        how='left'
    )

    print(f"合并后数据集大小: {len(merged_info)}")
    
    # 检查SMILES列
    smiles_column = 'smiles' if 'smiles' in compound_info.columns else 'SMILES'
    if smiles_column not in merged_info.columns:
        print(f"警告: 合并后的数据集中没有找到SMILES列。可用的列: {merged_info.columns.tolist()}")
        # 如果找不到SMILES列，添加一个空列以避免错误
        merged_info[smiles_column] = None
    
    # 过滤掉SMILES空的样本
    print(f"过滤前样本数量: {len(merged_info)}")
    filtered_info = merged_info.dropna(subset=[smiles_column])
    print(f"过滤后样本数量: {len(filtered_info)}")
    print(f"过滤掉了 {len(merged_info) - len(filtered_info)} 个SMILES为空的样本")
    
    # 使用过滤后的数据集
    sample_images = filtered_info.copy()
    
    # 测试模式参数
    test_mode = False  # 设置为True时只处理少量样本且不更新metadata
    
    # 如果是测试模式，只取少量样本
    if test_mode:
        sample_images = sample_images.iloc[:20].copy()  # 测试模式只取20个样本
        print("测试模式：只处理前20个样本")
    else:
        print(f"完整模式：处理所有 {len(sample_images)} 个样本")

    # 1. 设置根目录
    root_dir = "/data/pr/cellpainting/BBBC021/raw_data/images"
    
    # 2. 获取所有药物
    compounds = sample_images[merge_key_image].unique()
    print(f"数据集中共有 {len(compounds)} 种不同药物")
    
    # 3. 准备原始图像路径并处理
    image_paths = []
    for idx, row in sample_images.iterrows():
        path_name = row['Image_PathName_DAPI']
        path_name = path_name.split('/')[-1]
        
        # 获取各通道文件名
        dapi_filename = row['Image_FileName_DAPI']
        tubulin_filename = row['Image_FileName_Tubulin']
        actin_filename = row['Image_FileName_Actin']
        
        # 构建完整路径
        dapi_path = os.path.join(root_dir, path_name, dapi_filename)
        tubulin_path = os.path.join(root_dir, path_name, tubulin_filename)
        actin_path = os.path.join(root_dir, path_name, actin_filename)
        
        image_paths.append({
            'idx': idx,
            'dapi': dapi_path,
            'tubulin': tubulin_path,
            'actin': actin_path,
            'path_name': path_name,
            'dapi_filename': dapi_filename,
            'smiles': row[smiles_column],
            'moa': row['moa']
        })
    
    # 4. 首先处理所有原始图像，将通道合并为RGB
    print("开始处理原始图像...")
    
    original_merged_images = []
    
    if test_mode:
        # 测试模式：单进程处理
        for i, paths in enumerate(tqdm(image_paths[:20], desc="处理原始图像")):
            result = process_image(paths, root_dir, save_dir_train)
            original_merged_images.append(result)
    else:
        # 非测试模式：多进程处理
        num_processes = 64
        print(f"使用 {num_processes} 个进程进行并行处理")
        
        # 创建进程池
        with mp.Pool(processes=num_processes) as pool:
            # 使用partial固定部分参数
            process_func = partial(process_image, root_dir=root_dir, save_dir_train=save_dir_train)
            
            # 使用imap处理并显示进度条
            original_merged_images = list(tqdm(
                pool.imap(process_func, image_paths),
                total=len(image_paths),
                desc="多进程处理原始图像"
            ))
    
    print(f"已完成 {len(original_merged_images)} 张原始图像的处理")
    
    # 5. 创建药物到图像的映射
    compound_to_images = {}
    for idx, row in sample_images.iterrows():
        compound = row[merge_key_image]
        if compound not in compound_to_images:
            compound_to_images[compound] = []
        
        # 找到对应的合并图像路径
        merged_image = next((img['merged_image'] for img in original_merged_images if img['idx'] == idx), None)
        if merged_image:
            compound_to_images[compound].append({
                'idx': idx,
                'merged_image': merged_image,
                'smiles': row[smiles_column],
                'moa': row['moa']
            })
    
    # 6. 对每种药物的图像进行数据增强
    print("开始数据增强...")
    
    augmented_metadata = []
    next_idx = max(sample_images.index) + 1
    
    # 将原始图像添加到元数据
    for compound, images in compound_to_images.items():
        # 如果原始图像数量已经达到或超过目标，则只保留部分
        if len(images) > 1000:
            images = images[:1000]
        
        # 处理每个原始图像，添加到元数据
        for img_info in images:
            # 添加原始图像元数据
            augmented_metadata.append({
                'augmented_idx': img_info['idx'],
                'original_idx': img_info['idx'],
                'augmentation_id': 0,  # 0表示原始图像
                'merged_image': img_info['merged_image'],
                'split': 'train',  # 初始设置为train，之后再划分
                'smiles': img_info['smiles'],
                'moa': img_info['moa'],
                'compound': compound
            })
    
    # 并行处理药物增强
    for compound, images in tqdm(compound_to_images.items(), desc="药物处理"):
        # 如果原始图像数量已经达到或超过目标，则跳过增强
        if len(images) >= 1000:
            continue
            
        # 计算需要增强的数量
        augment_needed = 1000 - len(images)
        
        # 准备增强任务参数
        aug_params_list = []
        
        # 循环所有原始图像，为每个图像创建增强任务
        for img_info in images:
            original_path = img_info['merged_image']
            
            # 每个进程需要处理的图像数量
            tasks_per_image = augment_needed // len(images) + 1
            
            # 为每个原始图像创建多个增强任务
            for aug_id in range(1, tasks_per_image + 1):
                aug_params_list.append((
                    original_path,
                    next_idx,
                    aug_id,
                    save_dir_train,
                    compound,
                    img_info['idx'],
                    img_info['smiles'],
                    img_info['moa']
                ))
                next_idx += 1
                
                # 如果已经创建了足够的任务，则退出循环
                if len(aug_params_list) >= augment_needed:
                    break
            
            # 如果已经创建了足够的任务，则退出循环
            if len(aug_params_list) >= augment_needed:
                break
        
        # 限制任务数量与需要增强的数量一致
        aug_params_list = aug_params_list[:augment_needed]
        
        # 使用多进程并行处理增强任务
        num_processes = min(64, len(aug_params_list))
        if num_processes > 0:
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_augmentation, aug_params_list),
                    total=len(aug_params_list),
                    desc=f"并行增强处理 {compound}"
                ))
                
                # 过滤掉None结果并添加到元数据
                valid_results = [r for r in results if r is not None]
                augmented_metadata.extend(valid_results)
    
    # 7. 划分训练集和测试集
    print("划分训练集和测试集...")
    
    # 按药物分组
    compound_groups = {}
    for item in augmented_metadata:
        compound = item['compound']
        if compound not in compound_groups:
            compound_groups[compound] = []
        compound_groups[compound].append(item)
    
    # 为每种药物划分训练集和测试集
    for compound, items in compound_groups.items():
        # 随机打乱
        random.shuffle(items)
        
        # 80%作为训练集，20%作为测试集
        train_count = int(len(items) * 0.8)
        
        # 标记分割
        for i, item in enumerate(items):
            item['split'] = 'train' if i < train_count else 'test'
    
    # 8. 移动测试集图像
    print("移动测试集图像到测试目录...")
    
    # 找出所有测试集图像
    test_images = [item for item in augmented_metadata if item['split'] == 'test']
    
    # 移动图像
    for item in tqdm(test_images, desc="移动测试集图像"):
        src_path = item['merged_image']
        if os.path.exists(src_path):
            # 创建目标路径
            dst_path = os.path.join(save_dir_test, os.path.basename(src_path))
            
            # 移动文件
            shutil.move(src_path, dst_path)
            
            # 更新路径
            item['merged_image'] = dst_path
    
    # 9. 保存增强后的元数据
    print("保存增强后的元数据...")
    augmented_df = pd.DataFrame(augmented_metadata)
    
    # 统计结果
    train_count = sum(1 for item in augmented_metadata if item['split'] == 'train')
    test_count = sum(1 for item in augmented_metadata if item['split'] == 'test')
    
    print(f"\n处理完成！")
    print(f"总图像数: {len(augmented_metadata)}")
    print(f"训练集图像: {train_count} 张 ({train_count/len(augmented_metadata):.2%})")
    print(f"测试集图像: {test_count} 张 ({test_count/len(augmented_metadata):.2%})")
    
    # 添加原始样本中的其他信息
    for idx in augmented_df['original_idx'].unique():
        if idx in sample_images.index:
            original_row = sample_images.loc[idx].to_dict()
            for key, value in original_row.items():
                if key not in augmented_df.columns and key != 'split':
                    augmented_df[key] = augmented_df['original_idx'].apply(
                        lambda x: original_row[key] if x == idx else None
                    )
    
    # 保存元数据
    metadata_path = '/data/pr/cellpainting/BBBC021/raw_data/metadata/augmented_image_metadata.csv'
    augmented_df.to_csv(metadata_path, index=False)
    print(f"已保存增强后的元数据至: {metadata_path}")

if __name__ == "__main__":
    # 确保多进程代码在主模块中运行
    main()