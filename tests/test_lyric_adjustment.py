import unittest

from soramimic_score import LyricLine, adjust_known_lyrics


def transcript(*texts):
    return tuple(LyricLine(text, index * 2, index * 2 + 2) for index, text in enumerate(texts))


class LyricAdjustmentTests(unittest.TestCase):
    def test_unrecognized_whole_lines_remain_in_original_order(self):
        original = ("赤い花", "青い空", "白い雲")
        result = adjust_known_lyrics(original, transcript("青い空"))
        self.assertEqual([line.text for line in result.lines], list(original))
        self.assertEqual(result.detail["supplied_lines"], list(original))
        self.assertEqual([row["supplied_line_indices"] for row in result.detail["decisions"]
                          if row["operation"] == "remove"], [])

    def test_repeats_whole_input_lines_and_adds_new_heard_line(self):
        result = adjust_known_lyrics(("青い空",), transcript("青い空", "青い空", "ラララ"))
        self.assertEqual([line.text for line in result.lines], ["青い空", "青い空", "ラララ"])
        self.assertCountEqual([row["operation"] for row in result.detail["decisions"]],
                              ["keep", "repeat", "add"])

    def test_matching_pronunciation_preserves_input_spelling(self):
        reading = {"青い空": "アオイソラ", "あおいそら": "アオイソラ"}.__getitem__
        result = adjust_known_lyrics(("青い空",), transcript("あおいそら"), reading=reading)
        self.assertEqual(result.lines[0].text, "青い空")

    def test_identical_input_lines_are_consumed_before_counting_repetitions(self):
        result = adjust_known_lyrics(("青い空", "青い空"), transcript("青い空", "青い空", "青い空"))
        self.assertCountEqual([row["operation"] for row in result.detail["decisions"]],
                              ["keep", "keep", "repeat"])
        self.assertEqual([i for row in result.detail["decisions"]
                          for i in row["supplied_line_indices"]], [0, 1])

    def test_split_recognition_does_not_split_input_line(self):
        result = adjust_known_lyrics(("青い空白い雲",), transcript("青い空", "白い雲"))
        self.assertEqual([line.text for line in result.lines], ["青い空白い雲"])
        self.assertEqual((result.lines[0].start_sec, result.lines[0].end_sec), (0, 4))

    def test_merged_recognition_keeps_input_lines_without_inventing_times(self):
        result = adjust_known_lyrics(("青い空", "白い雲"), transcript("青い空白い雲"))
        self.assertEqual([line.text for line in result.lines], ["青い空", "白い雲"])
        self.assertTrue(all(line.start_sec is None and line.end_sec is None for line in result.lines))

    def test_matching_line_is_not_spliced_at_an_internal_omission(self):
        result = adjust_known_lyrics(("あおいそら",), transcript("あおそら"))
        self.assertEqual(result.lines[0].text, "あおいそら")

    def test_unrecognized_short_input_is_kept_without_fabricated_times(self):
        result = adjust_known_lyrics(("あいうえおかきくけこ", "ん"), transcript("あいうえおかきくけこ"))
        self.assertEqual([line.text for line in result.lines], ["あいうえおかきくけこ", "ん"])
        self.assertIsNone(result.lines[-1].start_sec)

    def test_unrelated_audio_does_not_replace_all_input(self):
        result = adjust_known_lyrics(("青い空",), transcript("ラララ"))
        self.assertEqual([line.text for line in result.lines], ["青い空"])

    def test_invalid_transcript_fails_but_empty_recognition_preserves_input(self):
        with self.assertRaises(ValueError):
            adjust_known_lyrics("青い空", transcript("青い空"))
        result = adjust_known_lyrics(("青い空",), ())
        self.assertEqual([line.text for line in result.lines], ["青い空"])
        with self.assertRaises(RuntimeError):
            adjust_known_lyrics(("青い空",), (LyricLine("青い空"),))


if __name__ == "__main__":
    unittest.main()
