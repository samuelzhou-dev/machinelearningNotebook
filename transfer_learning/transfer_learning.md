模型保存

for instance using mlp:

```py
模型保存到本地:

from sklearn.externals import joblib

joblib.dump(model,"model1.m")

加载本地模型：

model2 = joblib.load("model1.m")
```

数据增强

```py
from keras.preprocessing.image import ImageDataGenerator
##图片加载的文件路径，其下应有文件夹，每个文件夹对应其类别(path文件下)
path = 'original_data'
dst_path = 'gen_data'
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.02, horizontal_flip=True, vertical_flip=True)
gen = datagen.flow_from_directory(path, target_size=(224，224),batch_size=2, save_to_dir=dst_path,save_prefix='gen',save_format='jpg')

for i in range(100):
	gen.next()
```

