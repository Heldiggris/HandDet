# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/framework/thread_pool_executor.proto

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
  name='handsDet/framework/thread_pool_executor.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n.handsDet/framework/thread_pool_executor.proto\x12\thandsDet\x1a+handsDet/framework/handsDet_options.proto\"\xe9\x02\n\x19ThreadPoolExecutorOptions\x12\x13\n\x0bnum_threads\x18\x01 \x01(\x05\x12\x12\n\nstack_size\x18\x02 \x01(\x05\x12\x1b\n\x13nice_priority_level\x18\x03 \x01(\x05\x12`\n\x1drequire_processor_performance\x18\x04 \x01(\x0e\x32\x39.handsDet.ThreadPoolExecutorOptions.ProcessorPerformance\x12\x1a\n\x12thread_name_prefix\x18\x05 \x01(\t\"5\n\x14ProcessorPerformance\x12\n\n\x06NORMAL\x10\x00\x12\x07\n\x03LOW\x10\x01\x12\x08\n\x04HIGH\x10\x02\x32Q\n\x03\x65xt\x12\x1b.handsDet.MediaPipeOptions\x18\x93\xd3\xf5J \x01(\x0b\x32$.handsDet.ThreadPoolExecutorOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_handsDet__options__pb2.DESCRIPTOR,])



_THREADPOOLEXECUTOROPTIONS_PROCESSORPERFORMANCE = _descriptor.EnumDescriptor(
  name='ProcessorPerformance',
  full_name='handsDet.ThreadPoolExecutorOptions.ProcessorPerformance',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NORMAL', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOW', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HIGH', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=332,
  serialized_end=385,
)
_sym_db.RegisterEnumDescriptor(_THREADPOOLEXECUTOROPTIONS_PROCESSORPERFORMANCE)


_THREADPOOLEXECUTOROPTIONS = _descriptor.Descriptor(
  name='ThreadPoolExecutorOptions',
  full_name='handsDet.ThreadPoolExecutorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_threads', full_name='handsDet.ThreadPoolExecutorOptions.num_threads', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stack_size', full_name='handsDet.ThreadPoolExecutorOptions.stack_size', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nice_priority_level', full_name='handsDet.ThreadPoolExecutorOptions.nice_priority_level', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='require_processor_performance', full_name='handsDet.ThreadPoolExecutorOptions.require_processor_performance', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='thread_name_prefix', full_name='handsDet.ThreadPoolExecutorOptions.thread_name_prefix', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.ThreadPoolExecutorOptions.ext', index=0,
      number=157116819, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[],
  enum_types=[
    _THREADPOOLEXECUTOROPTIONS_PROCESSORPERFORMANCE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=107,
  serialized_end=468,
)

_THREADPOOLEXECUTOROPTIONS.fields_by_name['require_processor_performance'].enum_type = _THREADPOOLEXECUTOROPTIONS_PROCESSORPERFORMANCE
_THREADPOOLEXECUTOROPTIONS_PROCESSORPERFORMANCE.containing_type = _THREADPOOLEXECUTOROPTIONS
DESCRIPTOR.message_types_by_name['ThreadPoolExecutorOptions'] = _THREADPOOLEXECUTOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ThreadPoolExecutorOptions = _reflection.GeneratedProtocolMessageType('ThreadPoolExecutorOptions', (_message.Message,), dict(
  DESCRIPTOR = _THREADPOOLEXECUTOROPTIONS,
  __module__ = 'handsDet.framework.thread_pool_executor_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ThreadPoolExecutorOptions)
  ))
_sym_db.RegisterMessage(ThreadPoolExecutorOptions)

_THREADPOOLEXECUTOROPTIONS.extensions_by_name['ext'].message_type = _THREADPOOLEXECUTOROPTIONS
handsDet_dot_framework_dot_handsDet__options__pb2.MediaPipeOptions.RegisterExtension(_THREADPOOLEXECUTOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)