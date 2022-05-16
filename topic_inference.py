import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_text as text

from proto import page_topics_override_list_pb2


class PageTopicModelMetadata:

    def __init__(self, max_categories, min_category_weight, min_normalized_weight_within_top_n, min_none_weight):
        self.max_categories = max_categories
        self.min_category_weight = min_category_weight
        self.min_normalized_weight_within_top_n = min_normalized_weight_within_top_n
        self.min_none_weight = min_none_weight


class TopicsModel:

    def __load_override_list(override_file_path):
        ptol = page_topics_override_list_pb2.PageTopicsOverrideList()

        with open('../topic_model/override_list.pb', 'rb') as f:
            data = f.read()
        ptol.ParseFromString(data)

        res = {}
        for entry in ptol.entries:
            res[entry.domain] = list(entry.topics.topic_ids)

        return res

    def __load_labelmap(labelmap_file_path):
        labelmap = {}
        with open(labelmap_file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                topic_id = int(line)
                labelmap[i] = topic_id

        perm = np.zeros(len(labelmap), dtype=np.int32)

        for idx, topic_id in labelmap.items():
            if topic_id == -2:
                topic_id = 0
            perm[topic_id] = idx

        return perm

    def __load_taxonomy(taxonomy_file_path):
        taxonomy = {}
        with open(taxonomy_file_path) as f:
            for line in f:
                line = line.strip()
                topic_id, topic = line.split(';')
                topic_id = int(topic_id)
                taxonomy[topic_id] = topic
        return taxonomy

    # reimplementation from Chromium source code:
    # https://github.com/chromium/chromium/blob/081f9b510bde51fffeeb68f7022a9764444d9803/components/optimization_guide/content/browser/page_content_annotations_service.cc#L266
    def __preprocess(hostname):
        if hostname.startswith('www.'):
            hostname = hostname[4:]

        to_replace = ['-', '_', '.', '+']
        for c in to_replace:
            hostname = hostname.replace(c, ' ')

        return hostname

    def __init__(self, model_path, vocab_file_path, labelmap_file_path,
                 taxonomy_file_path, model_metadata, override_file_path=None):
        self.model_path = model_path
        self.vocab_file_path = vocab_file_path
        self.labelmap_file_path = labelmap_file_path
        self.taxonomy_file_path = taxonomy_file_path

        self.labelmap = TopicsModel.__load_labelmap(self.labelmap_file_path)
        self.taxonomy = TopicsModel.__load_taxonomy(self.taxonomy_file_path)

        self.metadata = model_metadata

        self.override_list = None
        if override_file_path:
            self.override_list = TopicsModel.__load_override_list(override_file_path)

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)

        self.tokenizer = text.BertTokenizer(self.vocab_file_path, token_out_type=tf.int32)

    def __vectorize(self, sequence):
        input_ids = np.zeros((1, 128), dtype='int32')
        mask = np.zeros((1, 128), dtype='int32')

        # Not very fancy, could break if vocab.txt changes
        CLS_TOKEN_ID = 2
        SEP_TOKEN_ID = 3

        tokens = self.tokenizer.tokenize(sequence)
        tokens = tokens.merge_dims(-2, -1)[0]

        input_ids[0, 0] = CLS_TOKEN_ID
        mask[0, 0] = 1
        for i, t in enumerate(tokens):
            input_ids[0, i + 1] = t
            mask[0, i + 1] = 1
        input_ids[0, len(tokens) + 1] = SEP_TOKEN_ID
        mask[0, len(tokens) + 1] = 1

        return input_ids, mask

    def __call_model(self, processed_hostname):
        input_ids, mask = self.__vectorize(processed_hostname)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(input_details[0]['index'], input_ids)
        self.interpreter.set_tensor(input_details[1]['index'], mask)
        self.interpreter.set_tensor(input_details[2]['index'], np.zeros((1, 128), dtype='int32'))
        self.interpreter.invoke()
        output_weights = self.interpreter.get_tensor(output_details[0]['index'])

        return output_weights[0][self.labelmap]

    # This follows the algorithm in:
    # https://github.com/chromium/chromium/blob/b97843e911ce8c2238f0b4145cf96a51ad81003b/components/optimization_guide/core/page_topics_model_executor.cc#L189
    def predict(self, hostname):

        NoneCategoryId = 0

        processed_hostname = TopicsModel.__preprocess(hostname)

        if self.override_list:
            if processed_hostname in self.override_list:
                categories = []
                for topic_id in self.override_list[processed_hostname]:
                    categories.append((topic_id, self.get_label(topic_id), 1.0))
                return categories

        weights = self.__call_model(processed_hostname)

        category_candidates = [(topic_id, weights[topic_id]) for topic_id in range(len(weights))]
        category_candidates = sorted(category_candidates, key=lambda item: item[1], reverse=True)

        total_weight = 0.0
        sum_positive_scores = 0.0
        none_idx_and_weight = None

        categories = []

        for i in range(self.metadata.max_categories):
            candidate = category_candidates[i]
            categories.append(candidate)

            total_weight += candidate[1]

            if candidate[1] > 0:
                sum_positive_scores += candidate[1]

            if candidate[0] == NoneCategoryId:
                none_idx_and_weight = (i, candidate[1])

        if self.metadata.min_category_weight > 0:
            categories = [category for category in categories if category[1] >= self.metadata.min_category_weight]

        if total_weight == 0:
            return []

        if none_idx_and_weight:
            if none_idx_and_weight[1] / total_weight > self.metadata.min_none_weight:
                return []

            categories = [category for category in categories if category[0] != NoneCategoryId]

        normalization_factor = sum_positive_scores if sum_positive_scores > 0 else 1.0

        categories = [category for category in categories
                      if category[1] / normalization_factor >= self.metadata.min_normalized_weight_within_top_n]

        final_categories = [category for category in categories if category[1] >= 0.0 and category[1] <= 1.0]

        final_categories = [(category[0], self.get_label(category[0]), category[1]) for category in final_categories]

        return final_categories

    def __call__(self, hostname):
        return self.predict(hostname)

    def get_label(self, label_id):
        return self.taxonomy.get(label_id, 'NA')


if __name__ == '__main__':
    import sys
    import argparse
    from os.path import exists

    # Parsing CLI arguments
    parser = argparse.ArgumentParser(description='Infers Privacy Sandbox''s Topics for given hostnames.')
    parser.add_argument('filename', type=str, help='A file containing one hostname per line')

    args = parser.parse_args()

    if not exists(args.filename):
        print(f'ERROR: File ''{args.filename}'' does not exist', file=sys.stderr)
        print()
        parser.print_help()
        exit(1)

    # Loading the model
    model_path = 'resources/topics_model/model.tflite'
    vocab_file_path = 'resources//vocab.txt'
    labelmap_file_path = 'resources/final_chrome_labelmap.txt'
    taxonomy_file_path = 'resources/taxonomy_v1.csv'
    override_file_path = 'resources/topic_model/override_list.pb'

    # parameters that appear to produce the same results as Chrome Canary
    model_metadata = PageTopicModelMetadata(max_categories=5,
                                            min_category_weight=0.1,
                                            min_normalized_weight_within_top_n=0.25,
                                            min_none_weight=0.8)

    model = TopicsModel(model_path, vocab_file_path, labelmap_file_path,
                        taxonomy_file_path, model_metadata, override_file_path)

    # Running inference for hostnames in provided file
    results = []
    with open(args.filename) as f:
        for line in f:
            hostname = line.strip()
            topics = model(hostname)
            results.append((hostname, topics))

    # Pretty printing the results
    col_length = [50, 50, 8]
    row_format = '{hostname:{col_length[0]}} {topic:{col_length[1]}} {weight:.2f}'
    header_format = '{hostname:{col_length[0]}} {topic:{col_length[1]}} {weight:{col_length[2]}}'
    print(header_format.format(hostname='Hostname', topic='Topics', weight='Weight', col_length=col_length))
    print('-' * (np.sum(col_length) + len(col_length) - 1))
    for hostname, topics in results:
        first_row = True
        for topic in topics:
            hostname_str = ''
            if first_row:
                hostname_str = hostname

            topic_str = '{} - {}'.format(topic[0], topic[1].split('/')[-1])
            print(row_format.format(hostname=hostname_str, topic=topic_str, weight=topic[2], col_length=col_length))
            first_row = False
        print()
