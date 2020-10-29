import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from keras.models import Sequential,Model
from keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Dropout,Dense,UpSampling2D,Flatten,Reshape,Input
from keras.initializers import RandomNormal
from keras.optimizers import Adam
PATH = './images/'
images = []
for image in os.listdir(PATH):
  img = Image.open(PATH+image)
  img = img.resize((120,120))
  img = np.asarray(img)
  images.append(img)
images = np.array(images)
images = (images.astype(np.float32) - 127.5) / 127.5
def call_generator():
  generator = Sequential()
  generator.add(Dense(128 * 15 * 15,kernel_initializer=RandomNormal(0,0.02),input_dim=100))
  generator.add(LeakyReLU(0.2))
  generator.add(Reshape((15, 15, 128)))
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Conv2DTranspose(128,(4,4),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Conv2D(3,(3,3),padding="same",activation="tanh",kernel_initializer=RandomNormal(0,0.02)))
  generator.compile(optimizer=Adam(0.0002,0.5),loss="binary_crossentropy")
  return generator
def call_discriminator():
  descriminator = Sequential()
  descriminator.add(Conv2D(64,(3,3),padding="same",kernel_initializer=RandomNormal(0,0.02),input_shape=(120,120,3)))
  descriminator.add(LeakyReLU(0.2))
  descriminator.add(Conv2D(128,(3,3),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  descriminator.add(LeakyReLU(0.2))
  descriminator.add(Conv2D(128,(3,3),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  descriminator.add(LeakyReLU(0.2))
  descriminator.add(Conv2D(256,(3,3),strides=2,padding="same",kernel_initializer=RandomNormal(0,0.02)))
  descriminator.add(LeakyReLU(0.2))
  descriminator.add(Flatten())
  descriminator.add(Dropout(0.2))
  descriminator.add(Dense(1,activation="sigmoid"))
  descriminator.compile(loss="binary_crossentropy",optimizer=Adam(0.0002,0.5))
  return descriminator
def show_images(noise, epoch=None):
    generated_images = gen.predict(noise)
    plt.figure(figsize=(10, 10))
    
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(image.reshape((120, 120, 3)))
        plt.axis('off')
    plt.tight_layout()
    if epoch != None:
        plt.savefig(f'./gan-images_epoch-{epoch}.png')
    plt.show()
gen = call_generator()
desc = call_discriminator()
desc.trainable = False
gan_input = Input(shape=(100,))
fake_img = gen(gan_input)
gan_output = desc(fake_img)
gan = Model(gan_input,gan_output)
gan.compile(loss="binary_crossentropy",optimizer=Adam(0.0002,0.5))
batch_size = 16
step_per_epoch = 491
s_noise = np.random.normal(0,1,size=(100,100))
for epoch in range(800):
  for batch in range(step_per_epoch):
   noise = np.random.normal(0,1,size=(batch_size,100))
   fake = gen.predict(noise)
   real = images[np.random.randint(0,images.shape[0],size=batch_size)]
   x = np.concatenate((real,fake))
   label_real = np.ones(2*batch_size)
   label_real[:batch_size] = 0.9
   desc_loss = desc.train_on_batch(x,label_real)
   label_fake = np.zeros(batch_size)
   gen_loss = gan.train_on_batch(noise,label_fake)
  
  print(f"Epoch : {epoch} / Descriminator Loss : {desc_loss} / Generator Loss : {gen_loss}")
  if epoch % 10 == 0:
   show_images(s_noise, epoch)
