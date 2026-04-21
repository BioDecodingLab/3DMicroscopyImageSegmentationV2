from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Concatenate,
    Conv3D,
    Conv3DTranspose,
    Dense,
    GlobalAveragePooling3D,
    GroupNormalization,
    Input,
    MaxPool3D,
    Multiply,
    ReLU,
    Reshape,
    UpSampling3D,
)


class UNet3D:
    def encoder(self, x, out_channels, is_maxpool):
        if is_maxpool:
            x = MaxPool3D(2, padding="same")(x)

        x = Conv3D(out_channels, 3, padding="same")(x)
        x = ReLU()(x)
        x = Conv3D(out_channels, 3, padding="same")(x)
        x = ReLU()(x)
        return x

    def decoder(self, x, encoder_input, out_channels):
        x = UpSampling3D(2)(x)
        x = Concatenate(axis=4)([x, encoder_input])
        x = self.encoder(x, out_channels, False)
        return x

    def build_model(self, input_size=(64, 64, 64, 1)):
        x = Input(input_size)

        encoder1 = self.encoder(x, 64, False)
        encoder2 = self.encoder(encoder1, 128, True)
        encoder3 = self.encoder(encoder2, 256, True)
        encoder4 = self.encoder(encoder3, 512, True)
        encoder5 = self.encoder(encoder4, 1024, True)

        decoder4 = self.decoder(encoder5, encoder4, 512)
        decoder3 = self.decoder(decoder4, encoder3, 256)
        decoder2 = self.decoder(decoder3, encoder2, 128)
        decoder1 = self.decoder(decoder2, encoder1, 64)

        output = Conv3D(1, 1, activation="sigmoid")(decoder1)

        return Model(x, output)


class AttentionUNet3D:
    def double_conv(self, x, out_channels):
        x = Conv3D(out_channels, kernel_size=3, padding="same")(x)
        x = ReLU()(x)
        x = GroupNormalization(8)(x)
        x = Conv3D(out_channels, kernel_size=3, padding="same")(x)
        x = ReLU()(x)
        x = GroupNormalization(8)(x)
        return x

    def attention(self, x, channels):
        # Channel Attention
        avg_pool = GlobalAveragePooling3D()(x)
        fc1 = Dense(1024)(avg_pool)
        fc2 = Dense(channels)(fc1)
        channel_weights = Reshape((1, 1, 1, channels))(fc2)
        channel_weights = Activation("sigmoid")(channel_weights)
        channel_attention = Multiply()([x, channel_weights])

        # Spatial Attention
        spatial_weights = Conv3D(1, 1)(x)
        spatial_weights = Activation("sigmoid")(spatial_weights)
        spatial_attention = Multiply()([x, spatial_weights])

        # Combine attentions
        return Add()([x, channel_attention, spatial_attention])

    def encoder(self, x, out_channels, is_maxpool):
        if is_maxpool:
            x = MaxPool3D(2, padding="same")(x)
        return self.double_conv(x, out_channels)

    def decoder(self, x, encoder_input, out_channels):
        # Upsample
        x = Conv3DTranspose(out_channels, 2, 2)(x)

        # Add skip connection
        x = Add()([encoder_input, x])

        # Apply attention
        x = self.attention(x, out_channels)

        # Double convolution
        return self.double_conv(x, out_channels)

    def build_model(self, input_shape=(64, 64, 64, 1)):
        inputs = Input(input_shape)

        # Encoder path
        encoder1 = self.encoder(inputs, 64, False)
        encoder2 = self.encoder(encoder1, 128, True)
        encoder3 = self.encoder(encoder2, 256, True)
        encoder4 = self.encoder(encoder3, 512, True)
        encoder5 = self.encoder(encoder4, 1024, True)

        # Decoder path
        decoder4 = self.decoder(encoder5, encoder4, 512)
        decoder3 = self.decoder(decoder4, encoder3, 256)
        decoder2 = self.decoder(decoder3, encoder2, 128)
        decoder1 = self.decoder(decoder2, encoder1, 64)

        # Output
        outputs = Conv3D(1, 1, activation="sigmoid")(decoder1)

        return Model(inputs, outputs)
