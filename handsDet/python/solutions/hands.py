import enum
from typing import NamedTuple

import numpy as np

from handsDet.calculators.core import constant_side_packet_calculator_pb2
from handsDet.calculators.core import gate_calculator_pb2
from handsDet.calculators.core import split_vector_calculator_pb2
from handsDet.calculators.tensor import image_to_tensor_calculator_pb2
from handsDet.calculators.tensor import inference_calculator_pb2
from handsDet.calculators.tensor import tensors_to_classification_calculator_pb2
from handsDet.calculators.tensor import tensors_to_detections_calculator_pb2
from handsDet.calculators.tensor import tensors_to_landmarks_calculator_pb2
from handsDet.calculators.tflite import ssd_anchors_calculator_pb2
from handsDet.calculators.util import association_calculator_pb2
from handsDet.calculators.util import detections_to_rects_calculator_pb2
from handsDet.calculators.util import logic_calculator_pb2
from handsDet.calculators.util import non_max_suppression_calculator_pb2
from handsDet.calculators.util import rect_transformation_calculator_pb2
from handsDet.python.solution_base import SolutionBase
from handsDet.python.solutions.hands_connections import HAND_CONNECTIONS


class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks."""
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20


_BINARYPB_FILE_PATH = 'handsDet/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb'


class Hands(SolutionBase):


  def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            'model_complexity': model_complexity,
            'num_hands': max_num_hands,
            'use_prev_landmarks': not static_image_mode,
        },
        calculator_params={
            'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'handlandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=[
            'multi_hand_landmarks', 'multi_hand_world_landmarks',
            'multi_handedness'
        ])

  def process(self, image: np.ndarray) -> NamedTuple:

    return super().process(input_data={'image': image})
