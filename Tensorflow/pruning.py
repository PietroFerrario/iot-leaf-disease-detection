import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
#from tensorflow.keras.callbacks import ModelCheckpoin
import functions as fn

def print_compression_stats(original_model, final_model):
    total = sum(w.size for w in original_model.get_weights())
    nonzero = sum(np.count_nonzero(w) for w in final_model.get_weights())
    
    pruned = sum(w.size for w in final_model.get_weights())
    zeroed = pruned - nonzero
    
    print(f"Original model params: {total:,}")
    print(f"Final (pruned) params: {pruned:,}")
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Zeroed weights: {zeroed:,} ({zeroed/pruned*100:.2f}% sparsity)")
    print(f"Remaining (non-zero): {nonzero:,}")
    print(f"Compression ratio: {total/nonzero:.2f}x smaller")
SEED=42


# Defining the Epochs and Batches 
EPOCHS = 45
BATCH_SIZE = 32
LR = 1e-4
PRUNE_EPOCHS = 10  # How many epochs to fine-tune and prune
FINAL_SAVED_MODEL = 'pruned_and_finetuned_model'


# Load model 
base_model = tf.keras.models.load_model('classifier_full_model')



dataset_root = '../Dataset'

raw_ds =  tf.keras.preprocessing.image_dataset_from_directory(
        dataset_root,
        label_mode = 'int',
        batch_size = None,
        image_size = (96, 96),
        color_mode = 'grayscale',
        shuffle = True
    )
total_examples = raw_ds.cardinality().numpy()

    
all_ds = raw_ds.shuffle(
    buffer_size = total_examples,
    seed = SEED,
    reshuffle_each_iteration=False,
)

train_count = int(0.7 * total_examples)
val_count = int(0.15 * total_examples)
test_count = total_examples - train_count - val_count

steps_per_epoch = math.ceil(train_count/ BATCH_SIZE)
end_step = steps_per_epoch * PRUNE_EPOCHS

pruning_params = {
    'pruning_schedule' : tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.87,
        begin_step=0,
        end_step=end_step
    )
}


# mean/std on raw data -> [0,255] -> [0,1]
normalized_ds = all_ds.map(lambda x,y : (tf.cast(x, tf.float32)/255.0, y))

#split 
train_raw = normalized_ds.take(train_count)
val_raw = normalized_ds.skip(train_count).take(val_count)
test_raw = normalized_ds.skip(train_count + val_count)

mean, std = fn.get_mean_std(train_raw, batch_size=BATCH_SIZE)

rotator = tf.keras.layers.RandomRotation(
    factor = 0.087,
    fill_mode = 'constant',
    fill_value = 0.0,
    seed=SEED
)

# Pre-Processing 
def preprocess(img, label, training=False):
    img = (img - mean) / std
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

inputs = tf.keras.layers.Input(shape=(96,96,1), name='model.input')
x = tfmot.sparsity.keras.prune_low_magnitude(
    tf.keras.layers.Conv2D(2,3, padding='same', activation='relu'),
    **pruning_params
    )(inputs)
x=tf.keras.layers.MaxPool2D(2)(x)

x = tfmot.sparsity.keras.prune_low_magnitude(
    tf.keras.layers.Conv2D(4,3, padding='same', activation='relu'),
    **pruning_params
    )(x)
x=tf.keras.layers.MaxPool2D(2)(x)

x = tf.keras.layers.Dropout(0.15)(x)
x = tf.keras.layers.Flatten()(x)

x = tfmot.sparsity.keras.prune_low_magnitude(
    tf.keras.layers.Dense(8, activation='relu'),
    **pruning_params
    )(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tfmot.sparsity.keras.prune_low_magnitude(
    tf.keras.layers.Dense(2),
    ** pruning_params
    )(x)


pruned_model = tf.keras.models.Model(inputs, outputs, name="pruned_functional")

for layer in pruned_model.layers:
    if layer.__class__.__name__ == 'PruneLowMagnitude':
        orig = base_model.get_layer(layer.layer.name)
        layer.layer.set_weights(orig.get_weights())
    elif layer.weights:
        try:
            base_layer = base_model.get_layer(layer.name)
            layer.set_weights(base_layer.get_weights())
        except ValueError:
            pass
    
# compile
pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
pruned_loss, pruned_acc = pruned_model.evaluate(val_ds, verbose =0)
print(f"Pruned Model Accuracy = {pruned_acc*100:.2f}")
        

callbacks = [
    # update pruning masks at each step
    tfmot.sparsity.keras.UpdatePruningStep(),
    tf.keras.callbacks.ModelCheckpoint('best_pruned_model', save_best_only=True, monitor='val_loss')
]

# Fine tune
pruned_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PRUNE_EPOCHS,
    callbacks=callbacks
)

# Strip pruning wrappers -> lean and unmasked model
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# compile
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
finetuned_loss, finetuned_acc = final_model.evaluate(test_ds, verbose =0)
print(f"FineTuned Accuracy = {finetuned_acc*100:.2f}%")
# Save model 
final_model.save(FINAL_SAVED_MODEL, save_format='tf')

# Inspect sparsity
loss, acc = final_model.evaluate(test_ds, verbose=0)
print(f'Pruned and Fine-Tuned test accuracy = {acc*100:.2f}%')
for layer in final_model.layers:
    if hasattr(layer, 'kernel'):
        w = layer.kernel.numpy().flatten()
        print(f"{layer.name:20s} sparsity = {100*np.mean(w==0):.2f}%")
        
print_compression_stats(base_model, final_model)

# Load and specify the input layer 
final_model_layer_spec = tf.keras.models.load_model(FINAL_SAVED_MODEL)
input_layer = tf.keras.layers.Input(batch_shape=(1,96,96,1))
fixed_input_model = tf.keras.Model(inputs=input_layer, outputs=final_model_layer_spec(input_layer))

# Convert to tf lite 
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_input_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_sparse_model = True
    
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.uint8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

raw_train = train_raw.batch(BATCH_SIZE)

mean, std = fn.get_mean_std(train_raw, batch_size=BATCH_SIZE)

def representative_dataset():
    for images, _ in raw_train.take(100):
        # images: [batch, 96, 96, 1] 
        # rescale to [0,1]
        images = tf.cast(images, tf.float32) / 255.0 
        images = (images - mean) / std
        for i in range(images.shape[0]):
            yield [images[i:i+1]] # -> shape (1, 96, 96, 1)
        
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()
with open("model_pruned.tflite", "wb") as f: 
    f.write(tflite_model)