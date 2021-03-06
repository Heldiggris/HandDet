# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/image/segmentation_smoothing_calculator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from handsDet.framework import calculator_pb2 as handsDet_dot_framework_dot_calculator__pb2
try:
  handsDet_dot_framework_dot_calculator__options__pb2 = handsDet_dot_framework_dot_calculator__pb2.handsDet_dot_framework_dot_calculator__options__pb2
except AttributeError:
  handsDet_dot_framework_dot_calculator__options__pb2 = handsDet_dot_framework_dot_calculator__pb2.handsDet.framework.calculator_options_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/calculators/image/segmentation_smoothing_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nChandsDet/calculators/image/segmentation_smoothing_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\xb2\x01\n&SegmentationSmoothingCalculatorOptions\x12&\n\x1b\x63ombine_with_previous_ratio\x18\x01 \x01(\x02:\x01\x30\x32`\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\xe8\x99\xfc\xb3\x01 \x01(\x0b\x32\x31.handsDet.SegmentationSmoothingCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_SEGMENTATIONSMOOTHINGCALCULATOROPTIONS = _descriptor.Descriptor(
  name='SegmentationSmoothingCalculatorOptions',
  full_name='handsDet.SegmentationSmoothingCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='combine_with_previous_ratio', full_name='handsDet.SegmentationSmoothingCalculatorOptions.combine_with_previous_ratio', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.SegmentationSmoothingCalculatorOptions.ext', index=0,
      number=377425128, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=121,
  serialized_end=299,
)

DESCRIPTOR.message_types_by_name['SegmentationSmoothingCalculatorOptions'] = _SEGMENTATIONSMOOTHINGCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SegmentationSmoothingCalculatorOptions = _reflection.GeneratedProtocolMessageType('SegmentationSmoothingCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _SEGMENTATIONSMOOTHINGCALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.image.segmentation_smoothing_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.SegmentationSmoothingCalculatorOptions)
  ))
_sym_db.RegisterMessage(SegmentationSmoothingCalculatorOptions)

_SEGMENTATIONSMOOTHINGCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _SEGMENTATIONSMOOTHINGCALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_SEGMENTATIONSMOOTHINGCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
