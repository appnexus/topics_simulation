# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/page_topics_override_list.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/page_topics_override_list.proto',
  package='optimization_guide.proto',
  syntax='proto2',
  serialized_options=b'\n0org.chromium.components.optimization_guide.protoB\033PageTopicsOverrideListProtoH\003',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n%proto/page_topics_override_list.proto\x12\x18optimization_guide.proto\"\\\n\x16PageTopicsOverrideList\x12\x42\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x31.optimization_guide.proto.PageTopicsOverrideEntry\"h\n\x17PageTopicsOverrideEntry\x12\x0e\n\x06\x64omain\x18\x01 \x01(\t\x12=\n\x06topics\x18\x02 \x01(\x0b\x32-.optimization_guide.proto.AnnotatedPageTopics\"(\n\x13\x41nnotatedPageTopics\x12\x11\n\ttopic_ids\x18\x01 \x03(\x05\x42Q\n0org.chromium.components.optimization_guide.protoB\x1bPageTopicsOverrideListProtoH\x03'
)




_PAGETOPICSOVERRIDELIST = _descriptor.Descriptor(
  name='PageTopicsOverrideList',
  full_name='optimization_guide.proto.PageTopicsOverrideList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='entries', full_name='optimization_guide.proto.PageTopicsOverrideList.entries', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=67,
  serialized_end=159,
)


_PAGETOPICSOVERRIDEENTRY = _descriptor.Descriptor(
  name='PageTopicsOverrideEntry',
  full_name='optimization_guide.proto.PageTopicsOverrideEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='domain', full_name='optimization_guide.proto.PageTopicsOverrideEntry.domain', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='topics', full_name='optimization_guide.proto.PageTopicsOverrideEntry.topics', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=161,
  serialized_end=265,
)


_ANNOTATEDPAGETOPICS = _descriptor.Descriptor(
  name='AnnotatedPageTopics',
  full_name='optimization_guide.proto.AnnotatedPageTopics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='topic_ids', full_name='optimization_guide.proto.AnnotatedPageTopics.topic_ids', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=267,
  serialized_end=307,
)

_PAGETOPICSOVERRIDELIST.fields_by_name['entries'].message_type = _PAGETOPICSOVERRIDEENTRY
_PAGETOPICSOVERRIDEENTRY.fields_by_name['topics'].message_type = _ANNOTATEDPAGETOPICS
DESCRIPTOR.message_types_by_name['PageTopicsOverrideList'] = _PAGETOPICSOVERRIDELIST
DESCRIPTOR.message_types_by_name['PageTopicsOverrideEntry'] = _PAGETOPICSOVERRIDEENTRY
DESCRIPTOR.message_types_by_name['AnnotatedPageTopics'] = _ANNOTATEDPAGETOPICS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PageTopicsOverrideList = _reflection.GeneratedProtocolMessageType('PageTopicsOverrideList', (_message.Message,), {
  'DESCRIPTOR' : _PAGETOPICSOVERRIDELIST,
  '__module__' : 'proto.page_topics_override_list_pb2'
  # @@protoc_insertion_point(class_scope:optimization_guide.proto.PageTopicsOverrideList)
  })
_sym_db.RegisterMessage(PageTopicsOverrideList)

PageTopicsOverrideEntry = _reflection.GeneratedProtocolMessageType('PageTopicsOverrideEntry', (_message.Message,), {
  'DESCRIPTOR' : _PAGETOPICSOVERRIDEENTRY,
  '__module__' : 'proto.page_topics_override_list_pb2'
  # @@protoc_insertion_point(class_scope:optimization_guide.proto.PageTopicsOverrideEntry)
  })
_sym_db.RegisterMessage(PageTopicsOverrideEntry)

AnnotatedPageTopics = _reflection.GeneratedProtocolMessageType('AnnotatedPageTopics', (_message.Message,), {
  'DESCRIPTOR' : _ANNOTATEDPAGETOPICS,
  '__module__' : 'proto.page_topics_override_list_pb2'
  # @@protoc_insertion_point(class_scope:optimization_guide.proto.AnnotatedPageTopics)
  })
_sym_db.RegisterMessage(AnnotatedPageTopics)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
