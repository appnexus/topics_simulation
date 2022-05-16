import unittest
from topic_inference import TopicsModel, PageTopicModelMetadata


class TestTopicsModel(unittest.TestCase):

    # All the hardcoded values come from Chromium unit tests:
    # https://github.com/chromium/chromium/blob/main/chrome/browser/optimization_guide/page_content_annotations_service_browsertest.cc#L258

    def setUp(self):
        model_path = 'resources/test_model/model.tflite'
        vocab_file_path = 'resources/test_model/vocab.txt'
        labelmap_file_path = 'resources/test_model/final_chrome_labelmap.txt'
        taxonomy_file_path = 'resources/taxonomy_v1.csv'

        model_metadata = PageTopicModelMetadata(max_categories=5,
                                                min_category_weight=0.1,
                                                min_normalized_weight_within_top_n=0.1,
                                                min_none_weight=0.8)

        self.model = TopicsModel(model_path, vocab_file_path, labelmap_file_path, taxonomy_file_path, model_metadata)

        self.expected_values = {
            'youtube.com': [(250, '/Online Communities', 0.601997),
                            (43, '/Arts & Entertainment/Online Video', 0.915914)],

            'chrome.com': [(223, '/Internet & Telecom/Web Apps & Online Tools', 0.209933),
                           (43, '/Arts & Entertainment/Online Video', 0.474946),
                           (148, '/Computers & Electronics/Software/Web Browsers', 0.881723)],

            'music.youtube.com': [(250, '/Online Communities', 0.450154),
                                  (1, '/Arts & Entertainment', 0.518014),
                                  (43, '/Arts & Entertainment/Online Video', 0.596481),
                                  (23, '/Arts & Entertainment/Music & Audio', 0.827426)]
        }

    def generic_test_hostname_prediction(self, hostname, decimal_place):
        expected_values = self.expected_values[hostname]
        actual_values = self.model(hostname)

        expected_values = sorted(expected_values)
        actual_values = sorted(actual_values)

        self.assertTrue(len(expected_values) == len(actual_values))

        for i in range(len(expected_values)):
            expected_topic_id, expected_topic, expected_weight = expected_values[i]
            actual_topic_id, actual_topic, actual_weight = actual_values[i]

            self.assertEqual(expected_topic_id, actual_topic_id)
            self.assertEqual(expected_topic, actual_topic)
            self.assertAlmostEqual(expected_weight, actual_weight, decimal_place)

    def test_youtube_predictions(self):
        hostname = 'youtube.com'
        decimal_place = 6
        self.generic_test_hostname_prediction(hostname, decimal_place)

    def test_chrome_predictions(self):
        hostname = 'chrome.com'
        decimal_place = 6
        self.generic_test_hostname_prediction(hostname, decimal_place)

    def test_music_youtube_predictions(self):
        hostname = 'music.youtube.com'
        decimal_place = 6
        self.generic_test_hostname_prediction(hostname, decimal_place)


if __name__ == '__main__':
    unittest.main()
