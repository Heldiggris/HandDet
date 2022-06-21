import collections
import enum
import os
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import numpy as np

from google.protobuf import descriptor
from google.protobuf import message
from handsDet.framework import calculator_pb2
from handsDet.framework.formats import detection_pb2
from handsDet.calculators.core import constant_side_packet_calculator_pb2
from handsDet.calculators.image import image_transformation_calculator_pb2
from handsDet.calculators.tensor import tensors_to_detections_calculator_pb2
from handsDet.calculators.util import landmarks_smoothing_calculator_pb2
from handsDet.calculators.util import logic_calculator_pb2
from handsDet.calculators.util import thresholding_calculator_pb2
from handsDet.framework.formats import classification_pb2
from handsDet.framework.formats import landmark_pb2
from handsDet.python._framework_bindings import calculator_graph
from handsDet.python._framework_bindings import image_frame
from handsDet.python._framework_bindings import packet
from handsDet.python._framework_bindings import resource_util
from handsDet.python._framework_bindings import validated_graph_config
import handsDet.python.packet_creator as packet_creator
import handsDet.python.packet_getter as packet_getter

RGB_CHANNELS = 3
CALCULATOR_TO_OPTIONS = {
    'ConstantSidePacketCalculator':
        constant_side_packet_calculator_pb2.ConstantSidePacketCalculatorOptions,
    'ImageTransformationCalculator':
        image_transformation_calculator_pb2
        .ImageTransformationCalculatorOptions,
    'LandmarksSmoothingCalculator':
        landmarks_smoothing_calculator_pb2.LandmarksSmoothingCalculatorOptions,
    'LogicCalculator':
        logic_calculator_pb2.LogicCalculatorOptions,
    'ThresholdingCalculator':
        thresholding_calculator_pb2.ThresholdingCalculatorOptions,
    'TensorsToDetectionsCalculator':
        tensors_to_detections_calculator_pb2
        .TensorsToDetectionsCalculatorOptions,
}


def type_names_from_oneof(oneof_type_name: str) -> Optional[List[str]]:
  if oneof_type_name.startswith('OneOf<') and oneof_type_name.endswith('>'):
    comma_separated_types = oneof_type_name[len('OneOf<'):-len('>')]
    return [n.strip() for n in comma_separated_types.split(',')]
  return None


@enum.unique
class PacketDataType(enum.Enum):
  STRING = 'string'
  BOOL = 'bool'
  BOOL_LIST = 'bool_list'
  INT = 'int'
  FLOAT = 'float'
  FLOAT_LIST = 'float_list'
  AUDIO = 'matrix'
  IMAGE = 'image'
  IMAGE_FRAME = 'image_frame'
  PROTO = 'proto'
  PROTO_LIST = 'proto_list'

  @staticmethod
  def from_registered_name(registered_name: str) -> 'PacketDataType':
    try:
      return NAME_TO_TYPE[registered_name]
    except KeyError as e:
      names = type_names_from_oneof(registered_name)
      if names:
        for n in names:
          if n in NAME_TO_TYPE.keys():
            return NAME_TO_TYPE[n]
      raise e

NAME_TO_TYPE: Mapping[str, 'PacketDataType'] = {
    'string':
        PacketDataType.STRING,
    'bool':
        PacketDataType.BOOL,
    '::std::vector<bool>':
        PacketDataType.BOOL_LIST,
    'int':
        PacketDataType.INT,
    'float':
        PacketDataType.FLOAT,
    '::std::vector<float>':
        PacketDataType.FLOAT_LIST,
    '::handsDet::Matrix':
        PacketDataType.AUDIO,
    '::handsDet::ImageFrame':
        PacketDataType.IMAGE_FRAME,
    '::handsDet::Classification':
        PacketDataType.PROTO,
    '::handsDet::ClassificationList':
        PacketDataType.PROTO,
    '::handsDet::ClassificationListCollection':
        PacketDataType.PROTO,
    '::handsDet::Detection':
        PacketDataType.PROTO,
    '::handsDet::DetectionList':
        PacketDataType.PROTO,
    '::handsDet::Landmark':
        PacketDataType.PROTO,
    '::handsDet::LandmarkList':
        PacketDataType.PROTO,
    '::handsDet::LandmarkListCollection':
        PacketDataType.PROTO,
    '::handsDet::NormalizedLandmark':
        PacketDataType.PROTO,
    '::handsDet::FrameAnnotation':
        PacketDataType.PROTO,
    '::handsDet::Trigger':
        PacketDataType.PROTO,
    '::handsDet::Rect':
        PacketDataType.PROTO,
    '::handsDet::NormalizedRect':
        PacketDataType.PROTO,
    '::handsDet::NormalizedLandmarkList':
        PacketDataType.PROTO,
    '::handsDet::NormalizedLandmarkListCollection':
        PacketDataType.PROTO,
    '::handsDet::Image':
        PacketDataType.IMAGE,
    '::std::vector<::handsDet::Classification>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::ClassificationList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::Detection>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::DetectionList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::Landmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::LandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::NormalizedLandmark>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::NormalizedLandmarkList>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::Rect>':
        PacketDataType.PROTO_LIST,
    '::std::vector<::handsDet::NormalizedRect>':
        PacketDataType.PROTO_LIST,
}


class SolutionBase:

  def __init__(
      self,
      binary_graph_path: Optional[str] = None,
      calculator_params: Optional[Mapping[str, Any]] = None,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):

    root_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    resource_util.set_resource_dir(root_path)
    validated_graph = validated_graph_config.ValidatedGraphConfig()
    validated_graph.initialize(
        binary_graph_path=os.path.join(root_path, binary_graph_path))

    canonical_graph_config_proto = self._initialize_graph_interface(
        validated_graph, side_inputs, outputs, stream_type_hints)
    if calculator_params:
      self._modify_calculator_options(canonical_graph_config_proto,
                                      calculator_params)
    self._graph = calculator_graph.CalculatorGraph(
        graph_config=canonical_graph_config_proto)
    self._simulated_timestamp = 0
    self._graph_outputs = {}

    def callback(stream_name: str, output_packet: packet.Packet) -> None:
      self._graph_outputs[stream_name] = output_packet

    for stream_name in self._output_stream_type_info.keys():
      self._graph.observe_output_stream(stream_name, callback, True)

    self._input_side_packets = {
        name: self._make_packet(self._side_input_type_info[name], data)
        for name, data in (side_inputs or {}).items()
    }
    self._graph.start_run(self._input_side_packets)

  def process(
      self, input_data: Union[np.ndarray, Mapping[str, Union[np.ndarray,
                                                             message.Message]]]
  ) -> NamedTuple:
    self._graph_outputs.clear()

    if isinstance(input_data, np.ndarray):
      if len(self._input_stream_type_info.keys()) != 1:
        raise ValueError(
            "Can't process single image input since the graph has more than one input streams."
        )
      input_dict = {next(iter(self._input_stream_type_info)): input_data}
    else:
      input_dict = input_data
    self._simulated_timestamp += 33333
    for stream_name, data in input_dict.items():
      input_stream_type = self._input_stream_type_info[stream_name]
      if (input_stream_type == PacketDataType.PROTO_LIST or
          input_stream_type == PacketDataType.AUDIO):
        raise NotImplementedError(
            f'SolutionBase can only process non-audio and non-proto-list data. '
            f'{self._input_stream_type_info[stream_name].name} '
            f'type is not supported yet.')
      elif (input_stream_type == PacketDataType.IMAGE_FRAME or
            input_stream_type == PacketDataType.IMAGE):
        if data.shape[2] != RGB_CHANNELS:
          raise ValueError('Input image must contain three channel rgb data.')
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))
      else:
        self._graph.add_packet_to_input_stream(
            stream=stream_name,
            packet=self._make_packet(input_stream_type,
                                     data).at(self._simulated_timestamp))

    self._graph.wait_until_idle()
    solution_outputs = collections.namedtuple(
        'SolutionOutputs', self._output_stream_type_info.keys())
    for stream_name in self._output_stream_type_info.keys():
      if stream_name in self._graph_outputs:
        setattr(
            solution_outputs, stream_name,
            self._get_packet_content(self._output_stream_type_info[stream_name],
                                     self._graph_outputs[stream_name]))
      else:
        setattr(solution_outputs, stream_name, None)

    return solution_outputs


  def _initialize_graph_interface(
      self,
      validated_graph: validated_graph_config.ValidatedGraphConfig,
      side_inputs: Optional[Mapping[str, Any]] = None,
      outputs: Optional[List[str]] = None,
      stream_type_hints: Optional[Mapping[str, PacketDataType]] = None):
    canonical_graph_config_proto = calculator_pb2.CalculatorGraphConfig()
    canonical_graph_config_proto.ParseFromString(validated_graph.binary_config)

    def get_name(tag_index_name):
      return tag_index_name.split(':')[-1]

    def get_stream_packet_type(packet_tag_index_name):
      stream_name = get_name(packet_tag_index_name)
      if stream_type_hints and stream_name in stream_type_hints.keys():
        return stream_type_hints[stream_name]
      return PacketDataType.from_registered_name(
          validated_graph.registered_stream_type_name(stream_name))

    self._input_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in canonical_graph_config_proto.input_stream
    }

    if not outputs:
      output_streams = canonical_graph_config_proto.output_stream
    else:
      output_streams = outputs
    self._output_stream_type_info = {
        get_name(tag_index_name): get_stream_packet_type(tag_index_name)
        for tag_index_name in output_streams
    }

    def get_side_packet_type(packet_tag_index_name):
      return PacketDataType.from_registered_name(
          validated_graph.registered_side_packet_type_name(
              get_name(packet_tag_index_name)))

    self._side_input_type_info = {
        get_name(tag_index_name): get_side_packet_type(tag_index_name)
        for tag_index_name, _ in (side_inputs or {}).items()
    }
    return canonical_graph_config_proto

  def _modify_calculator_options(
      self, calculator_graph_config: calculator_pb2.CalculatorGraphConfig,
      calculator_params: Mapping[str, Any]) -> None:
    def generate_nested_calculator_params(flat_map):
      nested_map = {}
      for compound_name, field_value in flat_map.items():
        calculator_and_field_name = compound_name.split('.')
        if len(calculator_and_field_name) != 2:
          raise ValueError(
              f'The key "{compound_name}" in the calculator_params is invalid.')
        calculator_name = calculator_and_field_name[0]
        field_name = calculator_and_field_name[1]
        if calculator_name in nested_map:
          nested_map[calculator_name].append((field_name, field_value))
        else:
          nested_map[calculator_name] = [(field_name, field_value)]
      return nested_map

    def modify_options_fields(calculator_options, options_field_list):
      for field_name, field_value in options_field_list:
        if field_value is None:
          calculator_options.ClearField(field_name)
        else:
          field_label = calculator_options.DESCRIPTOR.fields_by_name[
              field_name].label
          if field_label == descriptor.FieldDescriptor.LABEL_REPEATED:
            if not isinstance(field_value, Iterable):
              raise ValueError(
                  f'{field_name} is a repeated proto field but the value '
                  f'to be set is {type(field_value)}, which is not iterable.')
            calculator_options.ClearField(field_name)
            for elem in field_value:
              getattr(calculator_options, field_name).append(elem)
          else:
            setattr(calculator_options, field_name, field_value)

    nested_calculator_params = generate_nested_calculator_params(
        calculator_params)

    num_modified = 0
    for node in calculator_graph_config.node:
      if node.name not in nested_calculator_params:
        continue
      options_type = CALCULATOR_TO_OPTIONS.get(node.calculator)
      if options_type is None:
        raise ValueError(
            f'Modifying the calculator options of {node.name} is not supported.'
        )
      options_field_list = nested_calculator_params[node.name]
      if node.HasField('options') and node.node_options:
        raise ValueError(
            f'Cannot modify the calculator options of {node.name} because it '
            f'has both options and node_options fields.')
      if node.node_options:
        node_options_modified = False
        for elem in node.node_options:
          type_name = elem.type_url.split('/')[-1]
          if type_name == options_type.DESCRIPTOR.full_name:
            calculator_options = options_type.FromString(elem.value)
            modify_options_fields(calculator_options, options_field_list)
            elem.value = calculator_options.SerializeToString()
            node_options_modified = True
            break
        if not node_options_modified:
          calculator_options = options_type()
          modify_options_fields(calculator_options, options_field_list)
          node.node_options.add().Pack(calculator_options)
      else:
        modify_options_fields(node.options.Extensions[options_type.ext],
                              options_field_list)

      num_modified += 1
      if num_modified == len(nested_calculator_params):
        break
    if num_modified < len(nested_calculator_params):
      raise ValueError('Not all calculator params are valid.')

  def _make_packet(self, packet_data_type: PacketDataType,
                   data: Any) -> packet.Packet:
    if (packet_data_type == PacketDataType.IMAGE_FRAME or
        packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_creator, 'create_' + packet_data_type.value)(
          data, image_format=image_frame.ImageFormat.SRGB)
    else:
      return getattr(packet_creator, 'create_' + packet_data_type.value)(data)

  def _get_packet_content(self, packet_data_type: PacketDataType,
                          output_packet: packet.Packet) -> Any:


    if output_packet.is_empty():
      return None
    if packet_data_type == PacketDataType.STRING:
      return packet_getter.get_str(output_packet)
    elif (packet_data_type == PacketDataType.IMAGE_FRAME or
          packet_data_type == PacketDataType.IMAGE):
      return getattr(packet_getter, 'get_' +
                     packet_data_type.value)(output_packet).numpy_view()
    else:
      return getattr(packet_getter, 'get_' + packet_data_type.value)(
          output_packet)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
