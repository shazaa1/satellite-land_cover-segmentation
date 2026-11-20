# Minimal CNN - Baseline Architecture

minimal_cnn = keras.Sequential(
    [
        layers.Input(shape=INPUT_SHAPE),

        # First Convolutional Layer
        layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # Second Convolutional Layer
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        # Output Layer: 5 classes
        layers.Conv2D(
            filters=NUM_CLASSES,
            kernel_size=(1, 1),
            padding="same"
        )
    ],
    name="Minimal_CNN_Baseline"
)


# 3. COMPILE MODEL

minimal_cnn.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
)


# 4. MODEL SUMMARY
minimal_cnn.summary()
