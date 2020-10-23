import pickle
from ConvolutionalNeuralNetworks import conv_model
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam   
'''
Adam = algorithm developed in 2014 designed for training neural networks

It looks like this:

for t in range(num_iterations):
    g = compute_gradient(x, y)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)

'''



training_file = './input/train.p'
testing_file = './input/test.p'
validating_file = './input/valid.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validating_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Number of training examples =", X_train.shape[0])
print("Number of validating examples =", X_valid.shape[0])
print("Number of testing examples =", X_test.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", len(np.unique(y_train)))

batch_size = 128
epochs = 150
model = conv_model()

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="./output/weights.hdf5", verbose=1, save_best_only=True, save_weights_only=True)
try:
    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_valid, y_valid),
                shuffle=True,
                callbacks=[checkpointer])
except KeyboardInterrupt:
    print("training interrupted")

model.load_weights("./output/weights.hdf5")
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))