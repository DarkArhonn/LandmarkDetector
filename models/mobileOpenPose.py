import tensorflow as tf


class RefineBlock(tf.keras.layers.Layer):
    def __init__(self, filter=128, kernel_size=3,  trainable=True):
        super().__init__(trainable)
        self.conv1 = tf.keras.layers.Conv2D(filters=filter, kernel_size=1, activation="relu", name="preconv")
        self.conv2 = tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, activation="relu", name="internal_conv")
        self.conv3 = tf.keras.layers.Conv2D(filters=filter, kernel_size=kernel_size, dilation_rate=2, activation="relu", name='dilated_conv')

    def call(self, inputs, **kwargs):
        out_one = self.conv1(inputs)
        out_two = self.conv2(out_one)
        out_three = self.conv3(out_two)
        out = out_three + out_one
        return out

class RefinementNet(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 n_heatmapts,
                 trainable=True):
        super().__init__(trainable)
        block_kwargs = {"filters": filters, "kernel_size": kernel_size, "trainable": True}
        self.refinement_blocks = [RefineBlock(**block_kwargs) for i in range(4)]
        self.post_conv = tf.keras.layers.Conv2D(filters=filters,kernel_size=1,activation="relu")
        self.heatmap = tf.keras.layers.Conv2D(filters=n_heatmapts,kernel_size=1,activation=None)

    def call(self, inputs, **kwargs):
        out = inputs
        for block in self.refinement_blocks:
            out = block(out)
        out = self.post_conv(out)
        return self.heatmap(out)



class MOpenPose(tf.keras.Model):
    def __init__(self,
                 backbone,
                 filters,
                 kernel_size,
                 n_heatmapts,
                 n_ref_blocks,
                 trainable=True):

        super().__init__(trainable)
        self.backbone = backbone
        build_kwargs = {"filters":filters,"kernel_size":kernel_size}
        self.conv_layers = [tf.keras.layers.Conv2D(**build_kwargs,activation="relu") for i in range(3)]
        self.conv_final = tf.keras.layers.Conv2D(filters=512,kernel_size=1,activation="relu")
        self.landmarks = tf.keras.layers.Conv2D(filters=n_heatmapts,kernel_size=1,activation=None)
        self.refinement_nets = [RefinementNet(**build_kwargs,n_heatmapts=n_heatmapts,trainable=trainable) for _ in range(n_ref_blocks)]

    def call(self, inputs, **kwargs):
        outputs = []
        backbone_out = self.backbone(inputs)
        backbone_out = backbone_out
        for conv in self.conv_layers:
            backbone_out = conv(backbone_out)
        backbone_out = self.conv_final(backbone_out)
        outputs.append(self.landmarks(backbone_out))
        for net in self.refinement_nets:
            outputs.append(net(outputs[-1]))
        return outputs
