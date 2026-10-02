import unittest

import numpy as np

from server.person_segment import decode_rgba_request, encode_mask_response, union_person_masks


class PersonMaskTests(unittest.TestCase):
    def test_binary_stream_preserves_request_id_dimensions_and_rgb_pixels(self):
        import struct
        message = struct.pack('<IHH', 7, 2, 1) + bytes([10, 20, 30, 255, 40, 50, 60, 255])
        request_id, image = decode_rgba_request(message)
        self.assertEqual(request_id, 7)
        self.assertEqual(image.shape, (1, 2, 3))
        self.assertEqual(image[0, 1].tolist(), [40, 50, 60])
        response = encode_mask_response(request_id, 2, 1, 4.25, bytes([255, 0]))
        self.assertEqual(struct.unpack_from('<IHHf', response), (7, 2, 1, 4.25))
        self.assertEqual(response[12:], bytes([255, 0]))

    def test_unions_multiple_people_without_losing_hands(self):
        masks = np.zeros((2, 3, 4), dtype=np.float32)
        masks[0, 0, 0] = 1.0
        masks[0, 1, 1] = 1.0
        masks[1, 2, 3] = 1.0
        output = union_person_masks(masks, width=4, height=3)
        self.assertEqual(len(output), 12)
        self.assertEqual(output[0], 255)
        self.assertEqual(output[5], 255)
        self.assertEqual(output[11], 255)
        self.assertEqual(output[6], 0)

    def test_empty_detection_returns_a_full_size_zero_mask(self):
        self.assertEqual(union_person_masks(None, width=4, height=3), bytes(12))

    def test_resizes_model_mask_to_source_frame(self):
        masks = np.array([[[0, 1], [0, 0]]], dtype=np.float32)
        output = union_person_masks(masks, width=4, height=4)
        self.assertEqual(output[2], 255)
        self.assertEqual(output[3], 255)
        self.assertEqual(output[10], 0)
