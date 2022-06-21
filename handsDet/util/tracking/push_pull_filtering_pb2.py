# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/util/tracking/push_pull_filtering.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/util/tracking/push_pull_filtering.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n1handsDet/util/tracking/push_pull_filtering.proto\x12\thandsDet\"\xc0\x01\n\x0fPushPullOptions\x12\x1b\n\x0f\x62ilateral_sigma\x18\x01 \x01(\x02:\x02\x32\x30\x12!\n\x16pull_propagation_scale\x18\x03 \x01(\x02:\x01\x38\x12!\n\x16push_propagation_scale\x18\x04 \x01(\x02:\x01\x38\x12!\n\x14pull_bilateral_scale\x18\x05 \x01(\x02:\x03\x30.7\x12!\n\x14push_bilateral_scale\x18\x06 \x01(\x02:\x03\x30.9*\x04\x08\x02\x10\x03')
)




_PUSHPULLOPTIONS = _descriptor.Descriptor(
  name='PushPullOptions',
  full_name='handsDet.PushPullOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bilateral_sigma', full_name='handsDet.PushPullOptions.bilateral_sigma', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(20),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pull_propagation_scale', full_name='handsDet.PushPullOptions.pull_propagation_scale', index=1,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(8),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='push_propagation_scale', full_name='handsDet.PushPullOptions.push_propagation_scale', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(8),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pull_bilateral_scale', full_name='handsDet.PushPullOptions.pull_bilateral_scale', index=3,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.7),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='push_bilateral_scale', full_name='handsDet.PushPullOptions.push_bilateral_scale', index=4,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.9),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(2, 3), ],
  oneofs=[
  ],
  serialized_start=65,
  serialized_end=257,
)

DESCRIPTOR.message_types_by_name['PushPullOptions'] = _PUSHPULLOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PushPullOptions = _reflection.GeneratedProtocolMessageType('PushPullOptions', (_message.Message,), dict(
  DESCRIPTOR = _PUSHPULLOPTIONS,
  __module__ = 'handsDet.util.tracking.push_pull_filtering_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.PushPullOptions)
  ))
_sym_db.RegisterMessage(PushPullOptions)


# @@protoc_insertion_point(module_scope)