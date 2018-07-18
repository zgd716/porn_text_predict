import tensorflow as tf
from tensorflow.python.framework import graph_util
from preprocessing import  *
import  pickle
import json



def predict(rnn_graph,texts,vocab_to_int):

    # textcnn  预测
    x = rnn_graph.get_tensor_by_name('prefix/Inputs/batch_ph:0')
    seq_length= rnn_graph.get_tensor_by_name('prefix/Inputs/seq_len_ph:0')
    keep_prob = rnn_graph.get_tensor_by_name('prefix/Inputs/keep_prob_ph:0')
    logits = rnn_graph.get_tensor_by_name('prefix/Fully_connected_layer/y_hat:0')
    # prediction data


        # the cutted sentence

    #sentence = list(cut('一起奸杀案之后, 她成为头条新闻的标题'))
    #sentence = list(cut('相比同时代手机产品，华为P20 Pro最为闪耀的特性即是搭载史无前例的后置三摄像头模组'))
    # sentence=list(cut('女排最大难题已在酝酿！郎平或被迫放弃一接班人'))
    # sentence = list(cut('趁父母不在，带男友回房间打炮自拍！'))
    # sentence = list(cut('传说是新疆夫妻自拍但有狼友说不是'))
    sentences=[cut(text) for text in texts]

    sentences_to_int = get_sentence2int(sentences, vocab_to_int, 30)#[0][np.newaxis, :]
    #sequence_len=list(x_to_int[0]).index(0)
    sequences_len=[]
    for sentence_to_int in sentences_to_int:
        if 0 in sentence_to_int:
            sequences_len.append(list(sentence_to_int).index(0))
        else:
            sequences_len.append(30)

    feed = {
        x: sentences_to_int,
        seq_length:sequences_len,
        keep_prob: 1.0
    }

    with tf.Session(graph=rnn_graph) as sess:
        log = sess.run([logits], feed)
        #print('logits:{}'.format(log))
        prob=sess.run(tf.nn.softmax(log[0]))
        if len(prob.shape)>1:
            prob = prob[:, 1]
        else:
            prob =[prob[1]]
        return [round(p, 2) for p in prob], [int(10 * p) for p in prob]




#加载固化模型
def load_pb(frozen_graph_filename):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(frozen_graph_filename, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")


            # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            output_graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph
if __name__=='__main__':
    with open('vocab_to_int.pkl', 'rb') as pickle_file:
        vocab_to_int = pickle.load(pickle_file)
    rnn_graph = load_pb('freeze_model/textrnn.pb')

    prob2=predict(rnn_graph,['后入河北衡水熟女'],vocab_to_int)
    prob3= predict(rnn_graph,['娇小的宝贝口交坚挺阴茎酷坏坏汇聚全球经典潮吹成人视频在线诱惑成人视频大全最新在线色宅色情视频排行榜免费在线点播高清视频视频'],vocab_to_int)
    #prob3= predict(rnn_graph,'令人印象深刻的怪物乳房亚洲跳舞')
    prob4= predict(rnn_graph,['令人印象深刻的怪物乳房亚洲跳舞'],vocab_to_int)
    prob5= predict(rnn_graph,['全部'],vocab_to_int)
    #print(prob1)
    print(prob2)
    print(prob3)
    print(prob4)
    print(prob5)


