# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/framework/status_handler.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from handsDet.framework import handsDet_options_pb2 as handsDet_dot_framework_dot_handsDet__options__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/framework/status_handler.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n(handsDet/framework/status_handler.proto\x12\thandsDet\x1a+handsDet/framework/handsDet_options.proto\"\x8f\x01\n\x13StatusHandlerConfig\x12\x16\n\x0estatus_handler\x18\x01 \x01(\t\x12\x19\n\x11input_side_packet\x18\x02 \x03(\t\x12\x17\n\x0e\x65xternal_input\x18\xea\x07 \x03(\t\x12,\n\x07options\x18\x03 \x01(\x0b\x32\x1b.handsDet.MediaPipeOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_handsDet__options__pb2.DESCRIPTOR,])




_STATUSHANDLERCONFIG = _descriptor.Descriptor(
  name='StatusHandlerConfig',
  full_name='handsDet.StatusHandlerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status_handler', full_name='handsDet.StatusHandlerConfig.status_handler', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_side_packet', full_name='handsDet.StatusHandlerConfig.input_side_packet', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='external_input', full_name='handsDet.StatusHandlerConfig.external_input', index=2,
      number=1002, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='options', full_name='handsDet.StatusHandlerConfig.options', index=3,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=101,
  serialized_end=244,
)

_STATUSHANDLERCONFIG.fields_by_name['options'].message_type = handsDet_dot_framework_dot_handsDet__options__pb2._MEDIAPIPEOPTIONS
DESCRIPTOR.message_types_by_name['StatusHandlerConfig'] = _STATUSHANDLERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StatusHandlerConfig = _reflection.GeneratedProtocolMessageType('StatusHandlerConfig', (_message.Message,), dict(
  DESCRIPTOR = _STATUSHANDLERCONFIG,
  __module__ = 'handsDet.framework.status_handler_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.StatusHandlerConfig)
  ))
_sym_db.RegisterMessage(StatusHandlerConfig)


# @@protoc_insertion_point(module_scope)
