# Önemli Kavramlar ve Kodlar

1. Öğrenme katsayısı azaltma **ReduceLROnPlateau()**
    
    1.1.  Kavramlar ve anlamları
    
    1.1.1. **Patience**
    
    Eğitimde ki "**val_loss**" değeri 3 kere aynı anda gelirse. Öğrenme katsayısını azalt.
    
    1.2.1. **Factor**
    
    Sürekli aynı değerleri (val_loss, accuracy) vermeye başladığında **Factor** değeri ile öğrenme katsayısı çarpılır çıkan sonuç ise yeni learning rating değeri olur.
    
    ```python
    lr = ReduceLROnPlateau(monitor='val_loss',
                           patience = 3,
                           verbose=1,
                           mode='auto',
                           factor=0.25,
                           min_lr=0.000001)
    ```
    
2. Kontrol noktası - Her **epochta** ağırlıkları kaydetme **`ModelCheckpoint()`**
3. **Glob** Nedir?
    
    3.1 **Glob** Klasör içinde gezinme
    
    3.1.1.  `Recursive = True` Nedir?
    
    Dosyaların içindeki dosyaların altındaki yani bir nevi alt kümelerin içindeki dosyaları taramamıza yarar.
    
    ```python
    files= glob.glob('../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/**/'+ '**/*flair.nii', recursive=True)
    ```
    
    ```python
    len(files)
    ```
    
4. [`skimage.io`](http://skimage.io) Nedir?
    
    Sıkıştırılmış ya da değişik uzantılarda olan tıbbi görüntüleri okumamıza yardımcı olur. Tıbbi görseller için bize pluginler sunar.
    
    4.1. `plugin = 'simpleitk'`  Tıbbi Görseller için
    
5. 3 boyutlu Resim Özellikleri
    
    Dosyalarımızı çıkarttık ve **"img"** adlı değişkene atadık.
    
    ```python
    img = io.imread(example, plugin='simpleitk')
    print(img.shape, img.dtype)
    ```
    
    **ÇIKTI:**
    
    (155, 240, 240) float32
    
    Resmimizin 3 boyutlu olduğunu anladık ve resim boyutumuz normalden çok değişik bir şekilde geldi. (240, 240, 3) gibi bir değere sahip olması gerekirken (155, 240, 240) değerinde.
    
    ```python
    plt.imshow(img)
    ```
    
    Bu kod satırı hatalı olarak gösterilecektir nedeni ise `imshow` fonksiyonunun sadece 2 boyutlu görselleri desteklemesidir.
    
    ```python
    plt.imshow(img[90])
    ```
    
    **ÇIKTI:**
    
    ![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled.png)
    
    Kodu bu şekilde değiştirdiğimizde 2 boyutlu bir görsele sahip oluyoruz.
    
    ```python
    plt.imshow(img[:,90,:])
    ```
    
    **ÇIKTI:**
    
    Arkadan Görüntüsü
    
    ![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%201.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%201.png)
    
    ```python
    plt.imshow(img[90])
    ```
    
    **ÇIKTI:**
    
    Yandan Görüntüsü
    
    ![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%202.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%202.png)
    
    ```python
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.title('Yandan Goruntu')
    plt.axis('off')
    plt.imshow(img[:,:, 90])
    
    plt.subplot(3, 4, 2)
    plt.title('Segmentation')
    plt.axis('off')
    plt.imshow(img_seg[:,:, 90])
    
    plt.subplot(3, 4, 3)
    plt.title('Alttan Goruntu')
    plt.axis('off')
    plt.imshow(img[:,90, :])
    plt.subplot(3, 4, 4)
    plt.title('Segmentation')
    plt.axis('off')
    plt.imshow(img_seg[:,90, :])
    
    plt.subplot(3, 4, 5)
    plt.title('Ustten Goruntu')
    plt.axis('off')
    plt.imshow(img[90,:, :])
    plt.subplot(3, 4, 6)
    plt.title('Segmentation')
    plt.axis('off')
    plt.imshow(img_seg[90,:,:])
    ```
    
    Not:  `plt.subplot(uzunluk, genişlik, sıra)`
    
    **ÇIKTI:**
    
    ![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%203.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%203.png)
    
6. Expand ve Dims (np.expand_dims)
    
    Boyut Eklememize Yarar
    
    ![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%204.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%204.png)
    

7. Nekroz, Ödem Genişleyen Tümor Ayrımı ( Segmentasyon )

1.1. Verinin kopyasını oluşturmak için `seg_tam = img_seg.copy()` 

1.2. Belirtmek istediğimiz bölgenin değerini 1 kalan bölgeleri ise 0'a çevirme işlemi

Nekroz = 1

Ödem = 2

Genişleyen Tümör = 4

Olacak şekilde piksel değerlerine bölünmüş.

1.2.1. Tüm Tümorü Segmentasyonu

```python
seg_tam[seg_tam != 0] = 1
plt.imshow(seg_tam[90,:,:])
```

**ÇIKTI:**

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%205.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%205.png)

1.2.2. Nekroz Segmentasyonu

```python
seg_nekroz = img_seg.copy()
seg_nekroz[seg_nekroz != 1] = 0
plt.imshow(seg_nekroz[90,:,:])
```

**ÇIKTI:**

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%206.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%206.png)

1.2.3. Ödem Segmentasyonu

```python
seg_odem = img_seg.copy()
seg_odem[seg_odem ==1] = 0
seg_odem[seg_odem == 4] = 0
seg_odem[seg_odem != 0] = 1
plt.imshow(seg_odem[90,:,:])
```

**ÇIKTI:**

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%207.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%207.png)

**Toplu Çıktı:**

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%208.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%208.png)

Ardından verilerimizi "**flair**, **t2**, **seg**" olarak 3 bölüme ayırdık bunun sebebi ise verilerimiz farklı şekillerde kaydedilmiş olması.

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%209.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%209.png)

Bu görselde **T2** ve **FLAIR** görüntülerinin birbirlerine çok yakın olup eksiklerini kapattıklarını görebiliyoruz. 3 ayrı sınıftaki görsellerin boyutlarını incelemek istediğimizde

```python
print(flair.shape, t2.shape, seg.shape)
```

**ÇIKTI:**

((13300, 1, 240, 240), (13300, 1, 240, 240), (13300, 1, 240, 240))

13300 Tane görsel hepsi 1 kanallı 240x240 olacak şekilde boyutlanmış. Peki bu kanallar ne anlama geliyor. RGB (**red**, **green**, **blue**) olarak 3 kanala ayrılmış bulunmakta örneğin:

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2010.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2010.png)

olarak düşünebiliriz. **FLAIR** ve **T2** birbirlerine çok uyumlu ve birbirlerinin eksiklerini kapattığını söyledik bu modelimiz için çok avantaj sağlayacaktır.

Bu yüzden **FLAIR** katmanı ile **T2** katmanını birleştirmemiz gerekiyor. Birleştirme işlemini `Numpy` kütüphanesinden `Concatenate` adlı fonksiyon ile gerçekleştireceğiz.

```python
x_train = np.concatenate((flair, t2), axis=1)
```

Birleştirme işlemini yaptıktan sonra boyutlarını tekrar kontrol edelim.

```python
x_train.shape
```

**ÇIKTI:**

(13300, 2, 240, 240)

1 Katmanlı görselimiz 2 katmana yükseltilmiş bulunmakta.

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2011.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2011.png)

Modelden sonra 

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2012.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2012.png)

Eğer böyle bir hata gelirse çözümü bu şekildedir.

8. Model Eğitimi ve Test Edilmesi

```python
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')

def dice_coef(y_true, y_pred):
    smooth = 0.005 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
def unet_model():
    
    inputs = Input((2, 240 , 240))
    
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (inputs)
    batch1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch1)
    batch1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D((2, 2)) (batch1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (pool1)
    batch2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch2)
    batch2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D((2, 2)) (batch2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (pool2)
    batch3 = BatchNormalization(axis=1)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch3)
    batch3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D((2, 2)) (batch3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (pool3)
    batch4 = BatchNormalization(axis=1)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch4)
    batch4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (batch4)
    
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (pool4)
    batch5 = BatchNormalization(axis=1)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same') (batch5)
    batch5 = BatchNormalization(axis=1)(conv5)
    
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (batch5)
    up6 = concatenate([up6, conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same') (up6)
    batch6 = BatchNormalization(axis=1)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same') (batch6)
    batch6 = BatchNormalization(axis=1)(conv6)
    
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (batch6)
    up7 = concatenate([up7, conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (up7)
    batch7 = BatchNormalization(axis=1)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same') (batch7)
    batch7 = BatchNormalization(axis=1)(conv7)
    
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (batch7)
    up8 = concatenate([up8, conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (up8)
    batch8 = BatchNormalization(axis=1)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same') (batch8)
    batch8 = BatchNormalization(axis=1)(conv8)
    
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (batch8)
    up9 = concatenate([up9, conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (up9)
    batch9 = BatchNormalization(axis=1)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same') (batch9)
    batch9 = BatchNormalization(axis=1)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(batch9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

model = unet_model()
```

Modelimiz klasik UNet modeli olmakta ve doğruluk parametreleri içinde dife_coef ve dife_coef_loss kullanıyoruz.

8.1. Sorenson-Dice Katsayısı (Dice Sorensen Coefficient) Nedir?

Bu yazının amacı, [dizgiler (String)](http://www.bilgisayarkavramlari.com/2008/08/02/dizgi-string/) arasındaki mesafenin ölçülmesi için kullanılan dizgi metriklerinden (string metrics) Sorensen-Dice katsayısını (Sorensen-dice coefficient) anlatmaktır.

Öncelikle bir özellik çıkarımı yöntemi ile iki metin üzerinden özellikler çıkarılır ve ardından aşağıdaki formüle göre benzerlik hesabı yapılır.

![https://bilgisayarkavramlari.com/wp-content/plugins/latex/cache/tex_684de0aae3615e8a07fa0c90c752257d.gif](https://bilgisayarkavramlari.com/wp-content/plugins/latex/cache/tex_684de0aae3615e8a07fa0c90c752257d.gif)

Yöntemin çalışmasını iki dizgi üzerinde gösterelim:

Dizgi 1 = “bilgi”

Dizgi 2 = “bilim”

Bu iki dizgi üzerinde, öncelikle [özellik çıkarımı (feature extraction)](http://www.bilgisayarkavramlari.com/2008/12/01/ozellik-cikarimi-feature-extraction/) yapıyoruz. Örneğin her harf bir özellik olabilir veya [bi-gram](http://www.bilgisayarkavramlari.com/2011/04/23/n-gram/) kullanabiliriz. Diyelim ki bi-gram kullanmak istedik bu durumda iki dizginin bi-gram değerleri aşağıdaki şekilde olacaktır:

Bi-Gram(Dizgi 1)= {bi,il,lg,gi}

Bi-Gram(Dzigi 2)= {bi,il,li,im}

Yöntemimizde iki kümenin kesişim sayısı ve iki kümenin ayrı ayrı eleman sayısına ihtiyacımız var. Buna göre kesişim kümesinin eleman sayısı 2 ve her iki kümenin eleman sayısı da 4. Formülde yerine koyacak olursak:

olarak bulunur. Bu değerin yüksek olması, benzerliğin fazla olduğu ve düşük olması da benzerliğin az olduğu anlamına gelir.

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2013.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2013.png)

8.2. Model Eğitimi

```python
model.fit(x_train,seg,validation_split=0.15,batch_size=20,epochs=15,shuffle=True,verbose=1)
```

**ÇIKTI:**

```python
Epoch 1/15
119/119 [==============================] - 151s 1s/step - loss: 0.1278 - dice_coef: 0.8722 - val_loss: 0.3596 - val_dice_coef: 0.6404
Epoch 2/15
119/119 [==============================] - 129s 1s/step - loss: 0.1107 - dice_coef: 0.8893 - val_loss: 0.3625 - val_dice_coef: 0.6375
Epoch 3/15
119/119 [==============================] - 133s 1s/step - loss: 0.1056 - dice_coef: 0.8944 - val_loss: 0.3521 - val_dice_coef: 0.6479
Epoch 4/15
119/119 [==============================] - 130s 1s/step - loss: 0.0962 - dice_coef: 0.9038 - val_loss: 0.3706 - val_dice_coef: 0.6294
Epoch 5/15
119/119 [==============================] - 130s 1s/step - loss: 0.0899 - dice_coef: 0.9101 - val_loss: 0.3504 - val_dice_coef: 0.6496
Epoch 6/15
119/119 [==============================] - 129s 1s/step - loss: 0.0877 - dice_coef: 0.9123 - val_loss: 0.3359 - val_dice_coef: 0.6641
Epoch 7/15
119/119 [==============================] - 132s 1s/step - loss: 0.1074 - dice_coef: 0.8926 - val_loss: 0.3392 - val_dice_coef: 0.6608
Epoch 8/15
119/119 [==============================] - 132s 1s/step - loss: 0.0776 - dice_coef: 0.9224 - val_loss: 0.3222 - val_dice_coef: 0.6778
Epoch 9/15
119/119 [==============================] - 132s 1s/step - loss: 0.0726 - dice_coef: 0.9274 - val_loss: 0.3327 - val_dice_coef: 0.6673
Epoch 10/15
119/119 [==============================] - 130s 1s/step - loss: 0.0698 - dice_coef: 0.9302 - val_loss: 0.3473 - val_dice_coef: 0.6527
Epoch 11/15
119/119 [==============================] - 129s 1s/step - loss: 0.0668 - dice_coef: 0.9332 - val_loss: 0.3359 - val_dice_coef: 0.6641
Epoch 12/15
119/119 [==============================] - 129s 1s/step - loss: 0.0679 - dice_coef: 0.9321 - val_loss: 0.3405 - val_dice_coef: 0.6595
Epoch 13/15
119/119 [==============================] - 129s 1s/step - loss: 0.0638 - dice_coef: 0.9362 - val_loss: 0.3346 - val_dice_coef: 0.6654
Epoch 14/15
119/119 [==============================] - 132s 1s/step - loss: 0.0633 - dice_coef: 0.9367 - val_loss: 0.3303 - val_dice_coef: 0.6697
Epoch 15/15
119/119 [==============================] - 129s 1s/step - loss: 0.0609 - dice_coef: 0.9391 - val_loss: 0.3242 - val_dice_coef: 0.6758
<keras.callbacks.History at 0x7fb1c5b6fd10>
```

Modelimizi %20'lik kısmını test için ayırıp. Batch_size değerini 20 epochs'u ise 15 olarak belirledik ve verilerimizin karışık şekilde egitime girmesi için shuffle değerimizi true yaptık.

```python
model.save_weights('/content/drive/MyDrive/brats-15epochs-batch20.h5')
```

Modelimizi tekrar tekrar kullanmak için h5 formatıyla kaydettik.

8.3. Model Test Aşaması

```python
pred = model.predict(x_train[460][0])
```

Normalde test etmek için bu şekilde bir yol izlerdik fakat bu yolu izlediğimizde hata alacağız çünkü biz modelimizi 4 boyutlu görseller için eğittik ve input olarak verdiğimiz görsel 3 boyutlu olmakta bunu boyut ekleyerek düzeltiyoruz.

```python
ornek = np.expand_dims(x_train[460], axis=0)
ornek.shape
```

**ÇIKTI:**

(1, 2, 240, 240)

Artık tahmin ettireceğimiz model 4 boyutlu tahmin işleminin sonucunu görmek için

```python
plt.imshow(pred[0][0])
```

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2014.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2014.png)

Sonucumuz başarıyla geldi şimdi bu sonucu gerçek değerleri ile karşılaştırmak için bir fonksiyon yazalım.

```python
def show_prediction(count_index, color_index):
    x = count_index

    color = {
      0:'magma',
      1:'viridis',
      2:'gray',
      3:'inferno',
      4:'cividis',
      5:'hot',
    }

    a = color_index

    exam = np.expand_dims(x_train[x], axis=0)
    pred = model.predict(exam)

    fig = plt.figure(figsize=(15,10))

    plt.subplot(141)
    plt.title('Input (Flair + T2)')
    plt.axis('off')
    plt.imshow(x_train[x][0], cmap=color[a])

    plt.subplot(142)
    plt.title('Radiologist (segmentation)')
    plt.axis('off')
    plt.imshow(seg[x][0], cmap=color[a])

    plt.subplot(143)
    plt.title('Tahmin (prediction)')
    plt.axis('off')
    plt.imshow(pred[0][0], cmap=color[a])
```

```python
show_prediction(1132, 0)
```

Evet artık doğru değerlerimizi ve sonuçları görebiliyoruz.

8.3. Model Eğitimi ( 30 Epochs * Batch 24 ) BONUS + Veri sayısı arttırılmış

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2015.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2015.png)

```python
model.fit(x_train,seg,validation_split=0.30,batch_size=24
          ,epochs=30,shuffle=True,verbose=1)
```

**ÇIKTI:**

```python

```

![O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2016.png](O%CC%88nemli%20Kavramlar%20ve%20Kodlar%20f6490a8634dd4f89ae807fc55d0e99ff/Untitled%2016.png)
