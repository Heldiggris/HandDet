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
  name='handsDet/calculators/core/gate_calculator.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=_b('\242\002\tMediaPipe'),
  serialized_pb=_b('\n0handsDet/calculators/core/gate_calculator.proto\x12\thandsDet\x1a$handsDet/framework/calculator.proto\"\x9d\x01\n\x15GateCalculatorOptions\x12\x1e\n\x16\x65mpty_packets_as_allow\x18\x01 \x01(\x08\x12\x14\n\x05\x61llow\x18\x02 \x01(\x08:\x05\x66\x61lse2N\n\x03\x65xt\x12\x1c.handsDet.CalculatorOptions\x18\xdf\x9f\xe8| \x01(\x0b\x32 .handsDet.GateCalculatorOptionsB\x0c\xa2\x02\tMediaPipe')
  ,
  dependencies=[handsDet_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_GATECALCULATOROPTIONS = _descriptor.Descriptor(
  name='GateCalculatorOptions',
  full_name='handsDet.GateCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='empty_packets_as_allow', full_name='handsDet.GateCalculatorOptions.empty_packets_as_allow', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow', full_name='handsDet.GateCalculatorOptions.allow', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='handsDet.GateCalculatorOptions.ext', index=0,
      number=261754847, type=11, cpp_type=10, label=1,
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
  serialized_start=102,
  serialized_end=259,
)

DESCRIPTOR.message_types_by_name['GateCalculatorOptions'] = _GATECALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GateCalculatorOptions = _reflection.GeneratedProtocolMessageType('GateCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _GATECALCULATOROPTIONS,
  __module__ = 'handsDet.calculators.core.gate_calculator_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.GateCalculatorOptions)
  ))
_sym_db.RegisterMessage(GateCalculatorOptions)

_GATECALCULATOROPTIONS.extensions_by_name['ext'].message_type = _GATECALCULATOROPTIONS
handsDet_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GATECALCULATOROPTIONS.extensions_by_name['ext'])

DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
