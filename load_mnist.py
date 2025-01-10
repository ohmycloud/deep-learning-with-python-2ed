from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape) # (60000, 28, 28)
    print(len(train_labels))  # 60000
    print(train_labels)

    print(test_images.shape)  # (10000, 28, 28)
    print(len(test_labels))   # 10000
    print(test_labels)

    # 准备图像数据, 图片缩放
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255.0

    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # compile
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)


if __name__ == "__main__":
    main()
