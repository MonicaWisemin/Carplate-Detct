import tensorflow as tf
import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from train import CHARS

# 设置GPU内存增长
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 加载模型
model = tf.keras.models.load_model('cnn_model_chi.h5')

# 图像预处理函数
def preprocess_image(image_path, target_size=(32, 32)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    return img

# 推理函数
def predict_plate(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_indices = np.argmax(predictions, axis=1)
    plate = ''.join([CHARS[i] for i in predicted_indices])
    return plate

# 批量推理
image_dir = '/Users/yangguang/Desktop/rev/CCPD2020/CCPD2020'
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    plate = predict_plate(image_path)
    print(f'Image: {image_file}, Predicted Plate: {plate}')

def preprocess_plate_image(img):
    """
    预处理车牌图片
    """
    # 调整大小
    img_resized = cv2.resize(img, (32, 32))
    
    # 转换为RGB
    if len(img_resized.shape) == 2:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 对比度增强
    img_contrast = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=30)
    
    # CLAHE处理
    lab = cv2.cvtColor(img_contrast, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_clahe = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
    
    # 锐化
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    img_sharp = cv2.filter2D(img_clahe, -1, kernel)
    
    # 归一化
    img_normalized = img_sharp.astype('float32') / 255.0
    
    return img_normalized

def extract_plate_region(image_path, save_debug=False):
    """
    从原始图片中提取车牌区域
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel边缘检测
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # 转换为8位无符号整数
    gradient = np.uint8(gradient)
    
    # 二值化
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # 根据面积和长宽比筛选可能的车牌区域
    plate_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        
        if 2.0 <= aspect_ratio <= 5.5 and area > 1000:
            plate_contours.append((x, y, w, h, area))
    
    if not plate_contours:
        return None
        
    # 选择面积最大的区域
    plate_contours.sort(key=lambda x: x[4], reverse=True)
    x, y, w, h = plate_contours[0][:4]
    
    # 提取车牌区域
    plate_region = img[y:y+h, x:x+w]
    
    if save_debug:
        debug_dir = "debug_plates"
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(f"{debug_dir}/{base_name}_plate.jpg", plate_region)
    
    return plate_region

def process_single_character(char_img, model):
    """
    处理单个字符
    """
    processed_img = preprocess_plate_image(char_img)
    prediction = model.predict(np.expand_dims(processed_img, axis=0), verbose=0)
    return np.argmax(prediction), np.max(prediction)

def batch_predict_plates(input_dir, save_results=True, save_debug=False):
    """
    批量处理车牌图片
    """
    # 加载模型
    print("加载模型中...")
    model_chinese = tf.keras.models.load_model('cnn_model_chi.h5')
    model_eng_num = tf.keras.models.load_model('cnn_model_eng_num.h5')
    
    # 字符映射
    provinces = ['川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', 
                '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '靑', '琼', '陕', 
                '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
                
    eng_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3',
                 '4', '5', '6', '7', '8', '9']
    
    results = []
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 创建进度条
    for img_file in tqdm(image_files, desc="处理车牌"):
        img_path = os.path.join(input_dir, img_file)
        
        try:
            # 提取车牌区域
            plate_region = extract_plate_region(img_path, save_debug)
            if plate_region is None:
                print(f"无法从 {img_file} 中提取车牌区域")
                continue
            
            # 分割字符
            h, w = plate_region.shape[:2]
            char_width = w // 7  # 假设车牌有7个字符
            
            # 处理第一个字符（省份）
            province_img = plate_region[:, :char_width]
            province_idx, province_conf = process_single_character(province_img, model_chinese)
            
            # 处理剩余字符
            result = provinces[province_idx]
            confidences = [province_conf]
            
            for i in range(1, 7):
                char_img = plate_region[:, i*char_width:(i+1)*char_width]
                char_idx, char_conf = process_single_character(char_img, model_eng_num)
                result += eng_chars[char_idx]
                confidences.append(char_conf)
            
            # 保存结果
            result_dict = {
                'image_file': img_file,
                'predicted_plate': result,
                'character_confidences': [float(c) for c in confidences],
                'average_confidence': float(np.mean(confidences))
            }
            results.append(result_dict)
            
            print(f"图片: {img_file}, 识别结果: {result}, 平均置信度: {np.mean(confidences):.4f}")
            
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")
            continue
    
    if save_results:
        # 保存结果到JSON文件
        with open('plate_recognition_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        total_plates = len(results)
        avg_confidence = np.mean([r['average_confidence'] for r in results])
        
        print("\n识别统计报告:")
        print(f"总处理图片数: {total_plates}")
        print(f"平均置信度: {avg_confidence:.4f}")
        
        # 保存统计报告
        with open('recognition_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"车牌识别统计报告\n")
            f.write(f"总处理图片数: {total_plates}\n")
            f.write(f"平均置信度: {avg_confidence:.4f}\n")
            f.write("\n详细识别结果:\n")
            for r in results:
                f.write(f"图片: {r['image_file']}\n")
                f.write(f"识别结果: {r['predicted_plate']}\n")
                f.write(f"置信度: {r['average_confidence']:.4f}\n")
                f.write("-" * 50 + "\n")
    
    return results

if __name__ == "__main__":
    input_directory = r"/Users/yangguang/Desktop/rev"
    batch_predict_plates(input_directory, save_results=True, save_debug=True) 