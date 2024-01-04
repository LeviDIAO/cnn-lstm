import pickle
import csv
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tqdm import tqdm  # 引入tqdm


def load_pickle(p, csv_file):
    # 从预处理过的数据文件中加载数据
    print("loading data...: "),
    x = pickle.load(open(p, "rb"), encoding='latin1')
    revs, U, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
    # print("从预处理过的数据文件中加载数据")
    # 从情感词典文件中读取情感词，构建情感词列表
    charged_words = []
    emof = open(csv_file, encoding='utf-8')
    csvf = csv.reader(emof, delimiter=',', quotechar='"')
    first_line = True
    for line in csvf:
        if first_line:
            first_line = False
            continue
        if line[11] == "1":
            charged_words.append(line[0])
    emof.close()
    charged_words = set(charged_words)
    # print("从情感词典文件中读取情感词，构建情感词列表")
    return revs, word_idx_map, mairesse, charged_words, U


# 将句子转换为索引列表，用零进行填充
def get_idx_from_sent(status, word_idx_map, charged_words, max_l=51, max_s=200, k=300, filter_h=5):
    # 存储索引表示的句子
    x = []
    pad = filter_h - 1
    length = len(status)
    pass_one = True
    while len(x) == 0:
        for i in range(length):
            words = status[i].split()
            if pass_one:
                words_set = set(words)
                if len(charged_words.intersection(words_set)) == 0:
                    continue
            else:
                if np.random.randint(0, 2) == 0:
                    continue
            y = []
            for i in range(pad):
                y.append(0)
            for word in words:
                if word in word_idx_map:
                    y.append(word_idx_map[word])
            while len(y) < max_l + 2 * pad:
                y.append(0)
            x.append(y)
        pass_one = False
    if len(x) < max_s:
        x.extend([[0] * (max_l + 2 * pad)] * (max_s - len(x)))
    return x


# 将句子转换为二维矩阵
def make_idx_data_cv(revs, word_idx_map, mairesse, charged_words, cv, per_attr=0, max_l=51, max_s=200, k=300,
                     filter_h=5):
    X, Y, M = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map,
                                 charged_words,
                                 max_l, max_s, k, filter_h)

        X.append(sent)
        Y.append(rev['y' + str(per_attr)])
        M.append(mairesse[rev["user"]])

    X = np.array(X)
    Y = np.array(Y)
    M = np.array(M)

    return X, Y, M


p_file = r"D:\JetBrains\PycharmProjects\cnn-bi-lstm-python27\essays_mairesse_small_batch.p"
csv_file = r"C:\Users\LeviDIAO\Desktop\Dataset\Emotion_Lexicon.csv"
revs, word_idx_map, mairesse, charged_words, U = load_pickle(p_file, csv_file)
attr = 2
X, Y, M = make_idx_data_cv(revs, word_idx_map,
                                mairesse, charged_words, attr, max_l=149, max_s=312,
                                k=300, filter_h=3)

feature_maps = 200
filter_hs = [1, 2, 3]
img_w = 300
filter_w = img_w
img_h = len(X[0][0])
filter_shapes = []
pool_sizes = []
for filter_h in filter_hs:
    filter_shapes.append((feature_maps, 1, filter_h, filter_w))
    pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))


# 定义共享变量 Words，包含所有词向量。初始化为预训练的词向量矩阵 U
# Words = paddle.create_parameter(shape=U.shape, dtype='float32',
#                                 default_initializer=paddle.nn.initializer.Assign(U), is_bias=False)

class ConvolutionalNetwork(nn.Layer):
    def __init__(self, num_filters, num_input_feature_maps, filter_height, filter_width, pool_size, activation='relu'):
        super(ConvolutionalNetwork, self).__init__()

        self.num_filters = num_filters
        self.num_input_feature_maps = num_input_feature_maps
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.image_height = img_h
        self.image_width = img_w
        self.pool_size = pool_size
        self.activation = activation

        # Initialize weights and biases
        fan_in = num_input_feature_maps * filter_height * filter_width
        fan_out = (num_filters * filter_height * filter_width) / (pool_size[0] * pool_size[1])
        W_bound = np.sqrt(6.0 / (fan_in + fan_out))

        if activation.lower() in ['none', 'relu']:
            self.W = self.create_parameter(shape=[num_filters, num_input_feature_maps, filter_height, filter_width],
                                           dtype='float32',
                                           default_initializer=nn.initializer.Uniform(low=-0.01, high=0.01))
        else:
            self.W = self.create_parameter(shape=[num_filters, num_input_feature_maps, filter_height, filter_width],
                                           dtype='float32',
                                           default_initializer=nn.initializer.Uniform(low=-W_bound, high=W_bound))

        self.b = self.create_parameter(shape=[num_filters], dtype='float32', is_bias=True,
                                       default_initializer=nn.initializer.Constant(value=0.0))

        # Convolution
        self.conv_out = nn.Conv2D(in_channels=self.num_input_feature_maps,
                                  out_channels=self.num_filters,
                                  kernel_size=(self.filter_height, self.filter_width),
                                  padding="VALID",
                                  stride=1,
                                  weight_attr=paddle.ParamAttr(
                                      initializer=nn.initializer.NumpyArrayInitializer(self.W.numpy())),
                                  bias_attr=paddle.ParamAttr(
                                      initializer=nn.initializer.NumpyArrayInitializer(self.b.numpy())))

        # Pooling
        self.max_pool = nn.MaxPool2D(kernel_size=self.pool_size, padding=0)

    def forward(self, x):
        conv_out = self.conv_out(x)
        # print('卷积输出conv_out:', conv_out.shape)

        # 非线性激活函数
        if self.activation == "tanh":
            conv_out_tanh = F.tanh(conv_out)
        elif self.activation == "relu":
            conv_out_tanh = F.relu(conv_out)
        else:
            conv_out_tanh = conv_out

        out = self.max_pool(conv_out_tanh)

        return out


# Create multiple convolutional layers
conv_layers = []
for i in range(len(filter_hs)):
    filter_shape = filter_shapes[i]
    pool_size = pool_sizes[i]
    conv_layer = ConvolutionalNetwork(num_filters=filter_shape[0],
                                      num_input_feature_maps=filter_shape[1],
                                      filter_height=filter_shape[2],
                                      filter_width=filter_shape[3],
                                      pool_size=pool_size)
    conv_layers.append(conv_layer)

# 构建了输入层，将文本数据映射为对应的词向量表示
paddle.enable_static()
x = paddle.static.data(name="x", shape=X.shape, dtype='int64')
embedding_output_train = paddle.static.nn.embedding(x, U.shape, is_sparse=True,
                                                        param_attr=paddle.nn.initializer.Assign(U))
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())
result_train = exe.run(paddle.static.default_main_program(), feed={'x': X.astype('int64')},
                           fetch_list=[embedding_output_train])
X = result_train[0]
# print('词嵌入向量layer0_input:', layer0_input.shape)

paddle.disable_static()


def cnn(layer0_input):
    index = layer0_input.sum(axis=(2, 3)) != 0
    conv_feats = []
    for i in range(layer0_input.shape[0]):
        layer1_inputs = []
        relv_input = layer0_input[i][:sum(index[i]), np.newaxis, :, :]
        # print('非零句子向量relv_input:', relv_input.shape)
        input = paddle.fluid.dygraph.to_variable(relv_input)

        for conv_layer in conv_layers:
            conv_out = conv_layer(paddle.to_tensor(input))
            # print('池化输出output:', conv_out.shape)
            layer1_inputs.append(paddle.flatten(conv_out, start_axis=1, stop_axis=-1))
            # print('展平layer1_inputs[0]:', layer1_inputs[0].shape)

        user_features = paddle.concat(layer1_inputs, axis=1)
        # print('n-gram特征的句子向量user_features:', user_features.shape)

        avg_feat = paddle.max(user_features, axis=0)

        conv_feats.append(avg_feat.numpy())

    conv_feats = np.array(conv_feats)
    layer1_input = np.concatenate((conv_feats, M), axis=1)
    print('layer1_input', layer1_input.shape)
    return layer1_input
layer1_input = cnn(X)
# Output CSV file path
csv_output_file = "document_features.csv"

# Writing feature vectors to CSV file
with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['Feature_' + str(i) for i in range(layer1_input.shape[1])]
    csv_writer.writerow(header)

    # Write feature vectors
    for feature_vector in layer1_input:
        csv_writer.writerow(feature_vector)

    print(f"Feature vectors have been written to {csv_output_file}.")

'''
input_dim = feature_maps * len(filter_hs) + M.shape[1]
fc_output_dim = 200
num_classes = 2


class Model(paddle.nn.Layer):
    def __init__(self, input_dim, fc_output_dim, num_classes):
        super(Model, self).__init__()

        # Fully connected layer
        self.fc_layer = paddle.nn.Linear(input_dim, fc_output_dim)

        # Sigmoid activation
        self.sigmoid = paddle.nn.Sigmoid()

        # Fully connected layer for softmax output
        self.fc_softmax = paddle.nn.Linear(fc_output_dim, num_classes)

    def forward(self, x):
        # Fully connected layer with sigmoid activation
        fc_output = self.fc_layer(x)
        sigmoid_output = self.sigmoid(fc_output)

        # Fully connected layer for softmax output
        softmax_output = self.fc_softmax(sigmoid_output)

        return softmax_output


# Instantiate the model
model_with_softmax = Model(input_dim, fc_output_dim, num_classes)

# Define the loss function (negative log-likelihood loss)
loss_fn = paddle.nn.CrossEntropyLoss()

# Define the optimizer (Adadelta optimizer)
optimizer = paddle.optimizer.Adadelta(parameters=model_with_softmax.parameters(), learning_rate=0.001)

Y = paddle.to_tensor(Y, dtype='int64')
X = paddle.to_tensor(cnn(trainX), dtype='float32')
print(X.shape)
print(Y.shape)
epochs = 100
for epoch in range(epochs):
    # Forward pass
    predictions = model_with_softmax(X)
    # Compute loss
    loss = loss_fn(predictions, Y)
    # Backward pass
    loss.backward()
    # Update parameters
    optimizer.step()
    optimizer.clear_grad()
    # Print the loss for every few epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

'''
# testY = paddle.to_tensor(testY, dtype='int64')
# print(testY.shape)
# testX = paddle.to_tensor(cnn(testX), dtype='int64')
# print(testX.shape)
# # Set the model to evaluation mode
# model_with_softmax.eval()
#
# # Forward pass to get predictions
# predictions = model_with_softmax(testX)
#
# # Calculate accuracy
# accuracy_metric = paddle.metric.accuracy(predictions, testY)
# accuracy = accuracy_metric.numpy()
# print(f'Accuracy on training data: {accuracy}')
