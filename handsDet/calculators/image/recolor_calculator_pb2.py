# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/image/recolor_calculator.proto

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
  name='handsDet/calculators/image/recolor_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n4handsDet/calculators/image/recolor_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\x1a\x1ahandsDet/util/color.proto\"\xcb\x02\n\x18RecolorCalculatorOptions\x12J\n\x0cmask_channel\x18\x01 \x01(\x0e\x32/.handsDet.RecolorCalculatorOptions.MaskChannel:\x03RED\x12\x1f\n\x05\x63olor\x18\x02 \x01(\x0b\x32\x10.handsDet.Color\x12\x1a\n\x0binvert_mask\x18\x03 \x01(\x08:\x05\x66\x61lse\x12#\n\x15\x61\x64just_with_luminance\x18\x04 \x01(\x08:\x04true\".\n\x0bMaskChannel\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x07\n\x03RED\x10\x01\x12\t\n\x05\x41LPHA\x10\x02\x32Q\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x8d\x84\xb5x \x01(\x0b\x32#.handsDet.RecolorCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,handsDet_dot_util_dot_color__pb2.DESCRIPTOR,])



_RECOLORCALCULATOROPTIONS_MASKCHANNEL = _descriptor.EnumDescriptor(
  name='MaskChannel',
  full_name='handsDet.RecolorCalculatorOptions.MaskChannel',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RED', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALPHA', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=336,
  serialized_end=382,
)
_sym_db.RegisterEnumDescriptor(_RECOLORCALCULATOROPTIONS_MASKCHANNEL)


_RECOLORCALCULATOROPTIONS = _descriptor.Descriptor(
  name='RecolorCalculatorOptions',
  full_name='handsDet.RecolorCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mask_channel', full_name='handsDet.RecolorCalculatorOptions.mask_channel', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='color', full_name='handsDet.RecolorCalculatorOptions.color', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='invert_mask', full_name='handsDet.RecolorCalculatorOptions.invert_mask', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='adjust_with_luminance', full_name='handsDet.RecolorCalculatorOptions.adjust_with_luminance', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.RecolorCalculatorOptions.ext', index=0,
      number=252527117, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[],
  enum_types=[
    _RECOLORCALCULATOROPTIONS_MASKCHANNEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=465,
)

_RECOLORCALCULATOROPTIONS.fields_by_name['mask_channel'].enum_type = _RECOLORCALCULATOROPTIONS_MASKCHANNEL
_RECOLORCALCULATOROPTIONS.fields_by_name['color'].message_type = handsDet_dot_util_dot_color__pb2._COLOR
_RECOLORCALCULATOROPTIONS_MASKCHANNEL.containing_type = _RECOLORCALCULATOROPTIONS
DESCRIPTOR.message_types_by_name['RecolorCalculatorOptions'] = _RECOLORCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RecolorCalculatorOptions = _reflection.GeneratedProtocolMessageType('RecolorCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _RECOLORCALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.image.recolor_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.RecolorCalculatorOptions)
  ))
_sym_db.RegisterMessage(RecolorCalculatorOptions)

_RECOLORCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _RECOLORCALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECOLORCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
