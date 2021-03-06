# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/util/top_k_scores_calculator.proto

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
  name='handsDet/calculators/util/top_k_scores_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n8handsDet/calculators/util/top_k_scores_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\xae\x01\n\x1bTopKScoresCalculatorOptions\x12\r\n\x05top_k\x18\x01 \x01(\x05\x12\x11\n\tthreshold\x18\x02 \x01(\x02\x12\x16\n\x0elabel_map_path\x18\x03 \x01(\t2U\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x8c\xba\xa9\x81\x01 \x01(\x0b\x32&.handsDet.TopKScoresCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_TOPKSCORESCALCULATOROPTIONS = _descriptor.Descriptor(
  name='TopKScoresCalculatorOptions',
  full_name='handsDet.TopKScoresCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='top_k', full_name='handsDet.TopKScoresCalculatorOptions.top_k', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='threshold', full_name='handsDet.TopKScoresCalculatorOptions.threshold', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_map_path', full_name='handsDet.TopKScoresCalculatorOptions.label_map_path', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.TopKScoresCalculatorOptions.ext', index=0,
      number=271211788, type=11, cpp_type=10, label=1,
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
  serialized_start=110,
  serialized_end=284,
)

DESCRIPTOR.message_types_by_name['TopKScoresCalculatorOptions'] = _TOPKSCORESCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TopKScoresCalculatorOptions = _reflection.GeneratedProtocolMessageType('TopKScoresCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _TOPKSCORESCALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.util.top_k_scores_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.TopKScoresCalculatorOptions)
  ))
_sym_db.RegisterMessage(TopKScoresCalculatorOptions)

_TOPKSCORESCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _TOPKSCORESCALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TOPKSCORESCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
