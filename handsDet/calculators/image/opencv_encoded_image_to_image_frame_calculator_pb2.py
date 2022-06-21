# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/calculators/image/opencv_encoded_image_to_image_frame_calculator.proto

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
  name='handsDet/calculators/image/opencv_encoded_image_to_image_frame_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nPhandsDet/calculators/image/opencv_encoded_image_to_image_frame_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\xcd\x01\n/OpenCvEncodedImageToImageFrameCalculatorOptions\x12/\n apply_orientation_from_exif_data\x18\x01 \x01(\x08:\x05\x66\x61lse2i\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x8c\xfa\xd8\x90\x01 \x01(\x0b\x32:.handsDet.OpenCvEncodedImageToImageFrameCalculatorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS = _descriptor.Descriptor(
  name='OpenCvEncodedImageToImageFrameCalculatorOptions',
  full_name='handsDet.OpenCvEncodedImageToImageFrameCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='apply_orientation_from_exif_data', full_name='handsDet.OpenCvEncodedImageToImageFrameCalculatorOptions.apply_orientation_from_exif_data', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.OpenCvEncodedImageToImageFrameCalculatorOptions.ext', index=0,
      number=303447308, type=11, cpp_type=10, label=1,
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
  serialized_start=134,
  serialized_end=339,
)

DESCRIPTOR.message_types_by_name['OpenCvEncodedImageToImageFrameCalculatorOptions'] = _OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OpenCvEncodedImageToImageFrameCalculatorOptions = _reflection.GeneratedProtocolMessageType('OpenCvEncodedImageToImageFrameCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.image.opencv_encoded_image_to_image_frame_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.OpenCvEncodedImageToImageFrameCalculatorOptions)
  ))
_sym_db.RegisterMessage(OpenCvEncodedImageToImageFrameCalculatorOptions)

_OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS.extensions_by_name['ext'].message_type = _OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_OPENCVENCODEDIMAGETOIMAGEFRAMECALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
