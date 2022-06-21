# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/util/rect_transformation_calculator.proto

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
  name='handsDet/calculators/util/rect_transformation_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n?handsDet/calculators/util/rect_transformation_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\xa4\x02\n#RectTransformationCalculatorOptions\x12\x12\n\x07scale_x\x18\x01 \x01(\x02:\x01\x31\x12\x12\n\x07scale_y\x18\x02 \x01(\x02:\x01\x31\x12\x10\n\x08rotation\x18\x03 \x01(\x02\x12\x18\n\x10rotation_degrees\x18\x04 \x01(\x05\x12\x0f\n\x07shift_x\x18\x05 \x01(\x02\x12\x0f\n\x07shift_y\x18\x06 \x01(\x02\x12\x13\n\x0bsquare_long\x18\x07 \x01(\x08\x12\x14\n\x0csquare_short\x18\x08 \x01(\x08\x32\\\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x88\x83\x85} \x01(\x0b\x32..handsDet.RectTransformationCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_RECTTRANSFORMATIONCALCULATOROPTIONS = _descriptor.Descriptor(
  name='RectTransformationCalculatorOptions',
  full_name='handsDet.RectTransformationCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale_x', full_name='handsDet.RectTransformationCalculatorOptions.scale_x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_y', full_name='handsDet.RectTransformationCalculatorOptions.scale_y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='handsDet.RectTransformationCalculatorOptions.rotation', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotation_degrees', full_name='handsDet.RectTransformationCalculatorOptions.rotation_degrees', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shift_x', full_name='handsDet.RectTransformationCalculatorOptions.shift_x', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shift_y', full_name='handsDet.RectTransformationCalculatorOptions.shift_y', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='square_long', full_name='handsDet.RectTransformationCalculatorOptions.square_long', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='square_short', full_name='handsDet.RectTransformationCalculatorOptions.square_short', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.RectTransformationCalculatorOptions.ext', index=0,
      number=262226312, type=11, cpp_type=10, label=1,
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
  serialized_start=117,
  serialized_end=409,
)

DESCRIPTOR.message_types_by_name['RectTransformationCalculatorOptions'] = _RECTTRANSFORMATIONCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RectTransformationCalculatorOptions = _reflection.GeneratedProtocolMessageType('RectTransformationCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _RECTTRANSFORMATIONCALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.util.rect_transformation_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.RectTransformationCalculatorOptions)
  ))
_sym_db.RegisterMessage(RectTransformationCalculatorOptions)

_RECTTRANSFORMATIONCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _RECTTRANSFORMATIONCALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECTTRANSFORMATIONCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)