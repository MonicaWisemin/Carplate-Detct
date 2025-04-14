import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, LSTM, Bidirectional, Reshape
import os
import cv2
import numpy as np
from imutils import paths
import random

CHARS = ['鲁', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '京', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader:
    def __init__(self, img_dir, imgSize, lpr_max_len):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len

    def __len__(self):
        return len(self.img_paths)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.transform(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = []
        for c in imgname:
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            else:
                print(f"未识别字符: {c}")  # 输出未识别字符
                continue

        if len(label) == 8:
            if not self.check(label):
                print(imgname)
                assert 0, "Error label!"

        return Image, label, len(label)

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        return True

def create_lprnet(num_classes):
    input_tensor = Input(shape=(32, 32, 3))
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=x)
    return model

def load_data(data_dir, target_size=(32, 32)):
    images = []
    labels = []
    class_dirs = sorted(os.listdir(data_dir))
    
    for class_index, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        class_files = os.listdir(class_dir)
        for img_name in class_files:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(class_index)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def train_model(data_dir, model_name):
    images, labels = load_data(data_dir)
    num_classes = len(set(labels))
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    model = create_lprnet(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(images, labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    model.save(model_name)
    print(f"{model_name} 模型训练完成并保存")

def predict_and_validate():
    from predict import predict_plate
    
    test_image = "2.jpg"
    result = predict_plate(test_image)
    print(f"测试图片识别结果: {result}")
    
    expected_result = "粤BF12345"
    if result != expected_result:
        print("推理结果不正确，重新训练模型...")
        return False
    return True

def main():
    while True:
        train_model('/Users/yangguang/Desktop/rev/CCPD2020/train/charsChinese', 'cnn_model_chi.h5')
        train_model('/Users/yangguang/Desktop/rev/CCPD2020/train/chars2', 'cnn_model_eng_num.h5')
        
        if predict_and_validate():
            break

if __name__ == "__main__":
    main()