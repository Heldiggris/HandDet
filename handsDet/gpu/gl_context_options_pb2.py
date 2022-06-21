
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
  name='handsDet/gpu/gl_context_options.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n&handsDet/gpu/gl_context_options.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"v\n\x10GlContextOptions\x12\x17\n\x0fgl_context_name\x18\x01 \x01(\t2I\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\x82\x89\x82j \x01(\x0b\x32\x1b.handsDet.GlContextOptions')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_GLCONTEXTOPTIONS = _descriptor.Descriptor(
  name='GlContextOptions',
  full_name='handsDet.GlContextOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gl_context_name', full_name='handsDet.GlContextOptions.gl_context_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.GlContextOptions.ext', index=0,
      number=222332034, type=11, cpp_type=10, label=1,
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
  serialized_start=91,
  serialized_end=209,
)

DESCRIPTOR.message_types_by_name['GlContextOptions'] = _GLCONTEXTOPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GlContextOptions = _reflection.GeneratedProtocolMessageType('GlContextOptions', (_message.Message,), dict(
  DESCRIPTOR = _GLCONTEXTOPTIONS,
  __module__ = 'handsDet.gpu.gl_context_options_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.GlContextOptions)
  ))
_sym_db.RegisterMessage(GlContextOptions)

_GLCONTEXTOPTIONS.extensions_by_name['ext'].message_type = _GLCONTEXTOPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GLCONTEXTOPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
