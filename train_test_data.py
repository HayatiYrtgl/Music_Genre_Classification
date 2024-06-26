import tensorflow as tf

data = tf.data.Dataset.load("created")
data = data.cache()
data = data.shuffle(1000)
data = data.batch(16)
data = data.prefetch(8)


train_size = int(len(data) * 0.7)
train_dataset = data.take(train_size)
test_dataset = data.skip(train_size).take(len(data) - train_size)

samples, labels = train_dataset.as_numpy_iterator().next()
print(labels.shape, samples.shape)



