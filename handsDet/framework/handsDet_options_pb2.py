# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/framework/handsDet_options.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/framework/handsDet_options.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=_b('\n\032com.google.handsDet.protoB\025MediaPipeOptionsProto'),
  serialized_pb=_b('\n+handsDet/framework/handsDet_options.proto\x12\thandsDet\"\x1e\n\x10MediaPipeOptions*\n\x08\xa0\x9c\x01\x10\x80\x80\x80\x80\x02\x42\x33\n\x1a\x63om.google.handsDet.protoB\x15MediaPipeOptionsProto')
)




_MEDIAPIPEOPTIONS = _descriptor.Descriptor(
  name='MediaPipeOptions',
  full_name='handsDet.MediaPipeOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(20000, 536870912), ],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=88,
)

DESCRIPTOR.message_types_by_name['MediaPipeOptions'] = _MEDIAPIPEOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MediaPipeOptions = _reflection.GeneratedProtocolMessageType('MediaPipeOptions', (_message.Message,), dict(
  DESCRIPTOR = _MEDIAPIPEOPTIONS,
  __module__ = 'handsDet.framework.handsDet_options_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.MediaPipeOptions)
  ))
_sym_db.RegisterMessage(MediaPipeOptions)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
