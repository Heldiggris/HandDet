# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/gpu/gl_scaler_calculator.proto

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
from handsDet.gpu import scale_mode_pb2 as handsDet_dot_gpu_dot_scale__mode__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/gpu/gl_scaler_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n(handsDet/gpu/gl_scaler_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\x1a\x1ehandsDet/gpu/scale_mode.proto\"\xa6\x02\n\x19GlScalerCalculatorOptions\x12\x14\n\x0coutput_width\x18\x01 \x01(\x05\x12\x15\n\routput_height\x18\x02 \x01(\x05\x12\x17\n\x0coutput_scale\x18\x07 \x01(\x02:\x01\x31\x12\x10\n\x08rotation\x18\x03 \x01(\x05\x12\x15\n\rflip_vertical\x18\x04 \x01(\x08\x12\x17\n\x0f\x66lip_horizontal\x18\x05 \x01(\x08\x12-\n\nscale_mode\x18\x06 \x01(\x0e\x32\x19.handsDet.ScaleMode.Mode2R\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x96\xcd\xaaO \x01(\x0b\x32$.handsDet.GlScalerCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,handsDet_dot_gpu_dot_scale__mode__pb2.DESCRIPTOR,])




_GLSCALERCALCULATOROPTIONS = _descriptor.Descriptor(
  name='GlScalerCalculatorOptions',
  full_name='handsDet.GlScalerCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_width', full_name='handsDet.GlScalerCalculatorOptions.output_width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_height', full_name='handsDet.GlScalerCalculatorOptions.output_height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_scale', full_name='handsDet.GlScalerCalculatorOptions.output_scale', index=2,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='handsDet.GlScalerCalculatorOptions.rotation', index=3,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip_vertical', full_name='handsDet.GlScalerCalculatorOptions.flip_vertical', index=4,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip_horizontal', full_name='handsDet.GlScalerCalculatorOptions.flip_horizontal', index=5,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scale_mode', full_name='handsDet.GlScalerCalculatorOptions.scale_mode', index=6,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.GlScalerCalculatorOptions.ext', index=0,
      number=166373014, type=11, cpp_type=10, label=1,
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
  serialized_start=126,
  serialized_end=420,
)

_GLSCALERCALCULATOROPTIONS.fields_by_name['scale_mode'].enum_type = handsDet_dot_gpu_dot_scale__mode__pb2._SCALEMODE_MODE
DESCRIPTOR.message_types_by_name['GlScalerCalculatorOptions'] = _GLSCALERCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GlScalerCalculatorOptions = _reflection.GeneratedProtocolMessageType('GlScalerCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _GLSCALERCALCULATOROPTIONS,
  __module__ = 'handsDet.gpu.gl_scaler_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.GlScalerCalculatorOptions)
  ))
_sym_db.RegisterMessage(GlScalerCalculatorOptions)

_GLSCALERCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _GLSCALERCALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GLSCALERCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
