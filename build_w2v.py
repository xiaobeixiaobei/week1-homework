import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:

        ret.append(line.split())

    return ret


def save_sentence(lines, sentence_path):

    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line)

    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)

    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    """
    通过gensim工具完成word2vec的训练，输入格式采用sentences，使用skip-gram，embedding维度256
    your code
    w2v = （one line）
    """

    # 训练skip-gram模型
    # min_count,频数阈值，大于等于1的保留
    # size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
    # workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。




    w2v = Word2Vec(sentences, size=256, window=5, min_count=1, workers=4, sg=1)

    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok.__________------" % w2v_bin_path)

    # test
    sim = w2v.wv.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)


    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    print(type(model))
    word_dict = {}
    for word in model.vocab:

        word_dict[word] = model[word]

    dump_pkl(word_dict, out_path, overwrite=True)


if __name__ == '__main__':
    build('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
          '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
          '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))

