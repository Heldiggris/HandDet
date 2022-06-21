import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()


from handsDet.framework import calculator_pb2 as handsDet_dot_framework_dot_calculator__pb2
try:
  handsDet_dot_framework_dot_calculator__options__pb2 = handsDet_dot_framework_dot_calculator__pb2.handsDet_dot_framework_dot_calculator__options__pb2
except AttributeError:
  handsDet_dot_framework_dot_calculator__options__pb2 = handsDet_dot_framework_dot_calculator__pb2.handsDet.framework.calculator_options_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/calculators/core/graph_profile_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=_b('\242\002\thandsDet'),
  serialized_pb=_b('\n9handsDet/calculators/core/graph_profile_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\x9b\x01\n\x1dGraphProfileCalculatorOptions\x12!\n\x10profile_interval\x18\x01 \x01(\x03:\x07\x31\x30\x30\x30\x30\x30\x30\x32W\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\xd7\xa7\x9d\xaf\x01 \x01(\x0b\x32(.handsDet.GraphProfileCalculatorOptionsB\x0c\xa2\x02\thandsDet')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_GRAPHPROFILECALCULATOROPTIONS = _descriptor.Descriptor(
  name='GraphProfileCalculatorOptions',
  full_name='handsDet.GraphProfileCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='profile_interval', full_name='handsDet.GraphProfileCalculatorOptions.profile_interval', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=1000000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.GraphProfileCalculatorOptions.ext', index=0,
      number=367481815, type=11, cpp_type=10, label=1,
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
  serialized_start=111,
  serialized_end=266,
)

DESCRIPTOR.message_types_by_name['GraphProfileCalculatorOptions'] = _GRAPHPROFILECALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphProfileCalculatorOptions = _reflection.GeneratedProtocolMessageType('GraphProfileCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _GRAPHPROFILECALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.core.graph_profile_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.GraphProfileCalculatorOptions)
  ))
_sym_db.RegisterMessage(GraphProfileCalculatorOptions)

_GRAPHPROFILECALCULATOROPTIONS.extensions_by_name['ext'].message_type = _GRAPHPROFILECALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GRAPHPROFILECALCULATOROPTIONS.extensions_by_name['ext'])

DESCRIPTOR._options = None