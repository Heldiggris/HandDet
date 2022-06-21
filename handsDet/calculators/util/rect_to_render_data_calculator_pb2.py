# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/util/rect_to_render_data_calculator.proto

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
from handsDet.util import color_pb2 as handsDet_dot_util_dot_color__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/calculators/util/rect_to_render_data_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n?handsDet/calculators/util/rect_to_render_data_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\x1a\x1ahandsDet/util/color.proto\"\xf7\x01\n!RectToRenderDataCalculatorOptions\x12\x0e\n\x06\x66illed\x18\x01 \x01(\x08\x12\x1f\n\x05\x63olor\x18\x02 \x01(\x0b\x32\x10.handsDet.Color\x12\x14\n\tthickness\x18\x03 \x01(\x01:\x01\x31\x12\x13\n\x04oval\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x12top_left_thickness\x18\x05 \x01(\x01\x32Z\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\xac\xdb\x87} \x01(\x0b\x32,.handsDet.RectToRenderDataCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,handsDet_dot_util_dot_color__pb2.DESCRIPTOR,])




_RECTTORENDERDATACALCULATOROPTIONS = _descriptor.Descriptor(
  name='RectToRenderDataCalculatorOptions',
  full_name='handsDet.RectToRenderDataCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filled', full_name='handsDet.RectToRenderDataCalculatorOptions.filled', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='color', full_name='handsDet.RectToRenderDataCalculatorOptions.color', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='thickness', full_name='handsDet.RectToRenderDataCalculatorOptions.thickness', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='oval', full_name='handsDet.RectToRenderDataCalculatorOptions.oval', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='top_left_thickness', full_name='handsDet.RectToRenderDataCalculatorOptions.top_left_thickness', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.RectToRenderDataCalculatorOptions.ext', index=0,
      number=262270380, type=11, cpp_type=10, label=1,
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
  serialized_start=145,
  serialized_end=392,
)

_RECTTORENDERDATACALCULATOROPTIONS.fields_by_name['color'].message_type = handsDet_dot_util_dot_color__pb2._COLOR
DESCRIPTOR.message_types_by_name['RectToRenderDataCalculatorOptions'] = _RECTTORENDERDATACALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RectToRenderDataCalculatorOptions = _reflection.GeneratedProtocolMessageType('RectToRenderDataCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _RECTTORENDERDATACALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.util.rect_to_render_data_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.RectToRenderDataCalculatorOptions)
  ))
_sym_db.RegisterMessage(RectToRenderDataCalculatorOptions)

_RECTTORENDERDATACALCULATOROPTIONS.extensions_by_name['ext'].message_type = _RECTTORENDERDATACALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECTTORENDERDATACALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)