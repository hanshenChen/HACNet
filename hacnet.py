#keras with tensorflow backend
from keras.models import Model
from keras.layers import Input, Layer, Activation, Conv2D, BatchNormalization, concatenate,MaxPooling2D,Dropout,Lambda
#tensorflow 
#from keras.models import Model
#from keras.layers import Input, Layer, Activation, Conv2D, BatchNormalization, concatenate,MaxPooling2D,Dropout,Lambda

def dilated_conv_block(inputs, filters, kernel_size=3, rate=1, name=None):
    if name==None:
        x = Conv2D(filters, kernel_size, padding='same',dilation_rate=rate,kernel_initializer='he_normal')(inputs)
    else:
        x = Conv2D(filters, kernel_size, padding='same',dilation_rate=rate,kernel_initializer='he_normal',name=name)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def hybird_aspp(x,channel,k,rates,sname=""):
    d1 = dilated_conv_block(x, int(channel / k), 3, rate=rates[0], name="aspps" +sname+ str(channel) + str(k) + "_b1conv2")
    d1 = Dropout(0.2)(d1)

    d2 = dilated_conv_block(d1, int(channel / k), 3, rate=rates[1], name="aspps" +sname+ str(channel) + str(k) + "_b2conv2")
    d2 = Dropout(0.2)(d2)

    d3 = dilated_conv_block(d2, int(channel / k), 3, rate=rates[2], name="aspps" +sname+ str(channel) + str(k) + "_b3conv2")
    d3 = Dropout(0.2)(d3)

    d4 = dilated_conv_block(d3, int(channel / k), 3, rate=rates[3], name="aspps" +sname+ str(channel) + str(k) + "_b4conv2")
    d4 = Dropout(0.2)(d4)

    d5 = concatenate([x, d1, d2, d3, d4])
    output_ch = channel - 8
    d5 = dilated_conv_block(d5, int(output_ch), (1, 3), name="aspps" +sname+ str(channel) + str(k) + "_b5conv1")
    d5 = dilated_conv_block(d5, int(output_ch), (3, 1), name="aspps" +sname+ str(channel) + str(k) + "_b5conv2")
    return d5

def hacnet(input_tensor=None, input_shape=None):

    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(tensor=input_tensor, shape=input_shape)

    conv1 = dilated_conv_block(inputs, int(48), 3, rate=1)

    p1=hybird_aspp(conv1,64,2,[1,3,9,27])
    p2=hybird_aspp(p1,80,2,[1,3,9,27])

    conv2d_p1 = Conv2D(1, 1, activation=None,name="conv2d_p1", kernel_initializer='he_normal')(p1)
    out_p1 = Activation('sigmoid')(conv2d_p1)

    conv2d_p2 = Conv2D(1, 1, activation=None,name="conv2d_p2", kernel_initializer='he_normal')(p2)
    out_p2 = Activation('sigmoid')(conv2d_p2)

    conv3 = Conv2D(1, 1, name="c1", activation='sigmoid', kernel_initializer='he_normal')(concatenate([conv2d_p1,conv2d_p2]))
    outputmerge = concatenate([out_p1,out_p2], axis=-1)
    model = Model(inputs, [outputmerge,conv3])
    model.summary()

    return model

def hacnet_d(input_tensor=None, input_shape=None):

    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(tensor=input_tensor, shape=input_shape)

    conv1 = dilated_conv_block(inputs, int(48), 3, rate=1)

    p1=hybird_aspp(conv1,64,2,[1,3,9,27])
    p2=hybird_aspp(p1,80,2,[1,3,9,27])
    p3=hybird_aspp(p2,80,2,[1,3,9,27],"s")

    conv2d_p1 = Conv2D(1, 1, activation=None,name="conv2d_p1", kernel_initializer='he_normal')(p1)
    out_p1 = Activation('sigmoid')(conv2d_p1)

    conv2d_p2 = Conv2D(1, 1, activation=None,name="conv2d_p2", kernel_initializer='he_normal')(p2)
    out_p2 = Activation('sigmoid')(conv2d_p2)

    conv2d_p3 = Conv2D(1, 1, activation=None,name="conv2d_p3", kernel_initializer='he_normal')(p3)
    out_p3 = Activation('sigmoid')(conv2d_p3)

    conv3 = Conv2D(1, 1, name="c1", activation='sigmoid', kernel_initializer='he_normal')(concatenate([conv2d_p1,conv2d_p2,conv2d_p3]))
    outputmerge = concatenate([out_p1,out_p2,out_p3], axis=-1)
    model = Model(inputs, [outputmerge,conv3])
    model.summary()

    return model

