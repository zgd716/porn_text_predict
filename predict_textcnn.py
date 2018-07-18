import tensorflow as tf
from tensorflow.python.framework import graph_util
from preprocessing import  *
import  pickle
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




def predict(text):
    cnn_graph = load_pb('freeze_model/textcnn.pb')
    # textcnn  预测
    x = cnn_graph.get_tensor_by_name('prefix/input_x:0')
    keep_prob = cnn_graph.get_tensor_by_name('prefix/keep_prob:0')
    logits = cnn_graph.get_tensor_by_name('prefix/output/logits:0')
    predictions = cnn_graph.get_tensor_by_name('prefix/output/predictions:0')
    # prediction data
    with open('vocab_to_int.pkl', 'rb') as pickle_file:
        vocab_to_int = pickle.load(pickle_file)

        # the cutted sentence

    #sentence = list(cut('一起奸杀案之后, 她成为头条新闻的标题'))
    #sentence = list(cut('相比同时代手机产品，华为P20 Pro最为闪耀的特性即是搭载史无前例的后置三摄像头模组'))
    # sentence=list(cut('女排最大难题已在酝酿！郎平或被迫放弃一接班人'))
    # sentence = list(cut('趁父母不在，带男友回房间打炮自拍！'))
    # sentence = list(cut('传说是新疆夫妻自拍但有狼友说不是'))
    sentence=list(cut(text))

    x_to_int = get_sentence2int([sentence], vocab_to_int, 30)[0][np.newaxis, :]
    feed = {
        x: x_to_int,
        keep_prob: 1.0
    }

    with tf.Session(graph=cnn_graph) as sess:
        log = sess.run([logits], feed)
        #print('logits:{}'.format(log))
        prob=sess.run(tf.nn.softmax(log[0]))[0][1]
        return  round(prob.item(),2),int(10*prob)



if __name__=='__main__':
    prob1=predict('后入河北衡水熟女')
    prob2= predict('娇小的宝贝口交坚挺阴茎酷坏坏汇聚全球经典潮吹成人视频在线诱惑成人视频大全最新在线色宅色情视频排行榜免费在线点播高清视频视频')
    prob3= predict('令人印象深刻的怪物乳房亚洲跳舞')
    print(prob1)
    print(prob2)
    print(prob3)


