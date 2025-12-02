import numpy as np 
import tensorflow as tf
from nn_models import Classifier
from epoch_metrics import PrintEpochMetrics
import functions as fn
from tqdm import tqdm
import time
import os
import random

SEED = 42

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Defining the Epochs and Batches 
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-4

if __name__ == '__main__':

    # Load Dataset 
    dataset_root = '../Dataset'
    auto = tf.data.AUTOTUNE
    base_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_root,
        label_mode = 'int',
        batch_size = None,
        image_size = (96, 96),
        color_mode = 'grayscale', 
        shuffle = False
    )
    
    total_examples = base_ds.cardinality().numpy()
    
    all_ds = base_ds.shuffle(
        buffer_size = total_examples,
        seed = SEED,
        reshuffle_each_iteration=False,
    )
    
    train_count = int(0.7 * total_examples)
    val_count = int(0.15 * total_examples)
    test_count = total_examples - train_count - val_count

    # Printing Training, Validation and Test set 
    print(f'Number of training examples: {train_count}')
    print(f'Number of validation examples: {val_count}')
    print(f'Number of testing examples: {test_count}')

    
    # mean/std on raw data -> [0,255] -> [0,1] -> bad for quantized 
    # normalized_ds = all_ds.map(lambda x,y : (tf.cast(x, tf.float32)/255.0, y))
    
    #split 
    # train_raw = normalized_ds.take(train_count)
    # val_raw = normalized_ds.skip(train_count).take(val_count)
    # test_raw = normalized_ds.skip(train_count + val_count)
    
    #  mean, std = fn.get_mean_std(train_raw, batch_size=BATCH_SIZE)

    
    train_count = int(0.7 * total_examples)
    val_count = int(0.15 * total_examples)
    test_count = total_examples - train_count - val_count
    
    train_raw = all_ds.take(train_count)
    val_raw = all_ds.skip(train_count).take(val_count)
    test_raw = all_ds.skip(train_count + val_count)
    
    
    rotator = tf.keras.layers.RandomRotation(
        factor = 0.087,
        fill_mode = 'constant',
        fill_value = 0.0,
        seed=SEED
    )
    
    # Pre-Processing 
    def preprocess(img, label, training=False):
        if training: 
            img = tf.image.random_flip_left_right(img)
            img = rotator(img)
        return img, label
    
    # train -> agumented
    train_ds = (
        train_raw
        .shuffle(buffer_size=1000, seed=SEED)
        .map(lambda x,y: preprocess(x,y,True), num_parallel_calls=1, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(1)
    )
    
    # val -> only normalized
    val_ds = (
        val_raw.map(lambda x,y: preprocess(x,y,False), num_parallel_calls=1, deterministic=True)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )
    
    # test -> only normalized
    test_ds = (
        test_raw.map(lambda x,y: preprocess(x,y,False),
            num_parallel_calls=1, deterministic=True)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )
    
    # Model definition
    
    
    # input_layer = tf.keras.layers.Input(shape=(64, 64, 1))
    
    # Model 
    model = Classifier(
        input_channels=1, conv1_out=2, conv2_out=4, 
        fc1_units=8, num_classes=2
        )
    
    #outputs = base_model(input_layer)
    
    #model = tf.keras.Model(inputs=input_layer, outputs=outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(LR),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'tut1-model', save_best_only=True, monitor='val_loss'
    )    
    print_cb = PrintEpochMetrics()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks = [checkpoint, print_cb,reduce_lr, early_stop ]

    
    # Training loop 
    train_losses , val_losses = [], []
    train_accs, val_accs = [], []
    
    start = time.time()

    for _ in tqdm(range(EPOCHS), desc='Epochs'):
        history = model.fit(
            train_ds, 
            validation_data = val_ds,
            epochs = 1,
            verbose=1, 
            callbacks = callbacks
            )
        # record metrics 
        train_losses.append(history.history['loss'][0])
        val_losses.append(history.history['val_loss'][0])
        train_accs.append(history.history['accuracy'][0])
        val_accs.append(history.history['val_accuracy'][0])
        
    end = time.time()
    m, s = fn.epoch_time(start,end)
    print(f"Total Training time: {m}m {s}s")
    
    # Evaluate on Test
    test_loss , test_acc = model.evaluate(test_ds, verbose = 0)
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")
    
    # Plot metrics 
    fn.plot_train_valid_loss_accuracy(
        train_losses,
        val_losses,
        test_loss,
        train_accs,
        val_accs,
        EPOCHS
    )
    
    #confusion matrix
    imgs, labels, probs = fn.get_prediction(model, test_ds)
    preds = np.argmax(probs, axis=1)
    fn.plot_confusion_matrix(labels, preds)
    
    # save model 
    model.save('classifier_full_model', save_format='tf')


