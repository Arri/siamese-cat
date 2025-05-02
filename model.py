import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model


@tf.keras.utils.register_keras_serializable()
def triplet_loss(margin):
    ''' Compute squared Euclidean distances.
    This function is decorated with @register_keras_serializable()
    so it can be saved/loaded with the model.'''
    def loss_fn(inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss_fn


def identity_loss(y_true, y_pred):
    ''' Used when the model already computes the loss as its output.
    Ignores y_true, returns the mean of the precomputed loss (y_pred).
    This is a workaround for Keras, which expects y_true and y_pred. '''
    return tf.reduce_mean(y_pred)


def get_base_model(b_model, input_shape):
    ''' This returns a pre-trained base model, without the top
    classification layer. Used as a feature extractor in the embedding
    model. '''
    model_dict = {
        'vgg': tf.keras.applications.vgg16.VGG16,
        'efficient': tf.keras.applications.EfficientNetB1,
        'mobile': tf.keras.applications.MobileNet,
        'mobileV2': tf.keras.applications.MobileNetV2,
        'MobileNetV3Small': tf.keras.applications.MobileNetV3Small,
        'MobileNetV3Large': tf.keras.applications.MobileNetV3Large
    }
    if b_model not in model_dict:
        raise ValueError(f"Unsupported model: {b_model}")
    return model_dict[b_model](include_top=False, input_shape=input_shape,
                               weights='imagenet')


def embedding_model_tl(inputShape, b_model, embeddingDim=128, unfreeze_layers=20):
    ''' Creates the embedding model using transfer learning.
    Freezes the vase mode.
    Adds Global average pooling, Dense and ReLU and Dense without
    activation
    Outputs a compact embedding for an input image. '''
    base_model = get_base_model(b_model, inputShape)
    # Unfreeze the last `unfreeze_layers` layers for fine-tuning
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=inputShape)

    x = base_model(inputs, training=False)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(embeddingDim)(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def complete_model_tl(base_model, imgsize, LR, margin=0.5):
    ''' Builds the full triplet network.
    - Takes 3 inputs: anchor, positive and negative images.
    - Feeds all through the same base embedding model.
    - Computes the triplet loss uring a Lambda layer.
    - Compiles the model using identity_loss and Adam optimizer. '''
    input_1 = tf.keras.Input((imgsize, imgsize, 3))
    input_2 = tf.keras.Input((imgsize, imgsize, 3))
    input_3 = tf.keras.Input((imgsize, imgsize, 3))

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    loss = layers.Lambda(triplet_loss(margin), output_shape=(1,))([A, P, N])
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model
