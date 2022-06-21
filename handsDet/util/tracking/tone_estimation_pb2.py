# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: handsDet/util/tracking/tone_estimation.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from handsDet.util.tracking import tone_models_pb2 as handsDet_dot_util_dot_tracking_dot_tone__models__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='handsDet/util/tracking/tone_estimation.proto',
  package='handsDet',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n-handsDet/util/tracking/tone_estimation.proto\x12\thandsDet\x1a)handsDet/util/tracking/tone_models.proto\"\x97\x05\n\nToneChange\x12+\n\tgain_bias\x18\x01 \x01(\x0b\x32\x18.handsDet.GainBiasModel\x12*\n\x06\x61\x66\x66ine\x18\x02 \x01(\x0b\x32\x1a.handsDet.AffineToneModel\x12:\n\x11mixture_gain_bias\x18\x03 \x01(\x0b\x32\x1f.handsDet.MixtureGainBiasModel\x12\x39\n\x0emixture_affine\x18\x04 \x01(\x0b\x32!.handsDet.MixtureAffineToneModel\x12\x1c\n\x14mixture_domain_sigma\x18\x05 \x01(\x02\x12\x17\n\x0c\x66rac_clipped\x18\x06 \x01(\x02:\x01\x30\x12\x16\n\x0elow_percentile\x18\x08 \x01(\x02\x12\x1a\n\x12low_mid_percentile\x18\t \x01(\x02\x12\x16\n\x0emid_percentile\x18\n \x01(\x02\x12\x1b\n\x13high_mid_percentile\x18\x0b \x01(\x02\x12\x17\n\x0fhigh_percentile\x18\x0c \x01(\x02\x12\x19\n\nlog_domain\x18\r \x01(\x08:\x05\x66\x61lse\x12/\n\x04type\x18\x0e \x01(\x0e\x32\x1a.handsDet.ToneChange.Type:\x05VALID\x12=\n\x0fstability_stats\x18\x0f \x01(\x0b\x32$.handsDet.ToneChange.StabilityStats\x1aU\n\x0eStabilityStats\x12\x13\n\x0bnum_inliers\x18\x01 \x01(\x05\x12\x17\n\x0finlier_fraction\x18\x02 \x01(\x02\x12\x15\n\rinlier_weight\x18\x03 \x01(\x01\"\x1e\n\x04Type\x12\t\n\x05VALID\x10\x00\x12\x0b\n\x07INVALID\x10\n\"\xd2\x01\n\x10ToneMatchOptions\x12\"\n\x14min_match_percentile\x18\x01 \x01(\x02:\x04\x30.01\x12\"\n\x14max_match_percentile\x18\x02 \x01(\x02:\x04\x30.99\x12\"\n\x16match_percentile_steps\x18\x03 \x01(\x05:\x02\x31\x30\x12\x18\n\x0cpatch_radius\x18\x04 \x01(\x05:\x02\x31\x38\x12\x1d\n\x10max_frac_clipped\x18\x05 \x01(\x02:\x03\x30.4\x12\x19\n\nlog_domain\x18\x08 \x01(\x08:\x05\x66\x61lse\"\x89\x01\n\x0f\x43lipMaskOptions\x12\x1a\n\x0cmin_exposure\x18\x01 \x01(\x02:\x04\x30.02\x12\x1a\n\x0cmax_exposure\x18\x02 \x01(\x02:\x04\x30.98\x12\x1f\n\x14max_clipped_channels\x18\x04 \x01(\x05:\x01\x31\x12\x1d\n\x12\x63lip_mask_diameter\x18\x05 \x01(\x05:\x01\x35\"\x81\x07\n\x15ToneEstimationOptions\x12\x37\n\x12tone_match_options\x18\x01 \x01(\x0b\x32\x1b.handsDet.ToneMatchOptions\x12\x35\n\x11\x63lip_mask_options\x18\x02 \x01(\x0b\x32\x1a.handsDet.ClipMaskOptions\x12\"\n\x14stats_low_percentile\x18\x03 \x01(\x02:\x04\x30.05\x12%\n\x18stats_low_mid_percentile\x18\x04 \x01(\x02:\x03\x30.2\x12!\n\x14stats_mid_percentile\x18\x05 \x01(\x02:\x03\x30.5\x12&\n\x19stats_high_mid_percentile\x18\x06 \x01(\x02:\x03\x30.8\x12#\n\x15stats_high_percentile\x18\x07 \x01(\x02:\x04\x30.95\x12\x1b\n\x0firls_iterations\x18\x08 \x01(\x05:\x02\x31\x30\x12P\n\x17stable_gain_bias_bounds\x18\t \x01(\x0b\x32/.handsDet.ToneEstimationOptions.GainBiasBounds\x12Y\n\x0f\x64ownsample_mode\x18\n \x01(\x0e\x32/.handsDet.ToneEstimationOptions.DownsampleMode:\x0f\x44OWNSAMPLE_NONE\x12\x1e\n\x11\x64ownsampling_size\x18\x0b \x01(\x05:\x03\x32\x35\x36\x12\x1c\n\x11\x64ownsample_factor\x18\x0c \x01(\x02:\x01\x32\x1a\xbb\x01\n\x0eGainBiasBounds\x12!\n\x13min_inlier_fraction\x18\x01 \x01(\x02:\x04\x30.75\x12\x1e\n\x11min_inlier_weight\x18\x02 \x01(\x02:\x03\x30.5\x12\x18\n\nlower_gain\x18\x03 \x01(\x02:\x04\x30.75\x12\x19\n\nupper_gain\x18\x04 \x01(\x02:\x05\x31.334\x12\x18\n\nlower_bias\x18\x05 \x01(\x02:\x04-0.2\x12\x17\n\nupper_bias\x18\x06 \x01(\x02:\x03\x30.2\"w\n\x0e\x44ownsampleMode\x12\x13\n\x0f\x44OWNSAMPLE_NONE\x10\x01\x12\x1a\n\x16\x44OWNSAMPLE_TO_MAX_SIZE\x10\x02\x12\x18\n\x14\x44OWNSAMPLE_BY_FACTOR\x10\x03\x12\x1a\n\x16\x44OWNSAMPLE_TO_MIN_SIZE\x10\x04\"/\n\tToneMatch\x12\x10\n\x08\x63urr_val\x18\x01 \x01(\x02\x12\x10\n\x08prev_val\x18\x02 \x01(\x02\"R\n\x0ePatchToneMatch\x12(\n\ntone_match\x18\x01 \x03(\x0b\x32\x14.handsDet.ToneMatch\x12\x16\n\x0birls_weight\x18\x02 \x01(\x02:\x01\x31')
  ,
  dependencies=[handsDet_dot_util_dot_tracking_dot_tone__models__pb2.DESCRIPTOR,])



_TONECHANGE_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='handsDet.ToneChange.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='VALID', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INVALID', index=1, number=10,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=737,
  serialized_end=767,
)
_sym_db.RegisterEnumDescriptor(_TONECHANGE_TYPE)

_TONEESTIMATIONOPTIONS_DOWNSAMPLEMODE = _descriptor.EnumDescriptor(
  name='DownsampleMode',
  full_name='handsDet.ToneEstimationOptions.DownsampleMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DOWNSAMPLE_NONE', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOWNSAMPLE_TO_MAX_SIZE', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOWNSAMPLE_BY_FACTOR', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DOWNSAMPLE_TO_MIN_SIZE', index=3, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1901,
  serialized_end=2020,
)
_sym_db.RegisterEnumDescriptor(_TONEESTIMATIONOPTIONS_DOWNSAMPLEMODE)


_TONECHANGE_STABILITYSTATS = _descriptor.Descriptor(
  name='StabilityStats',
  full_name='handsDet.ToneChange.StabilityStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_inliers', full_name='handsDet.ToneChange.StabilityStats.num_inliers', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inlier_fraction', full_name='handsDet.ToneChange.StabilityStats.inlier_fraction', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inlier_weight', full_name='handsDet.ToneChange.StabilityStats.inlier_weight', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=650,
  serialized_end=735,
)

_TONECHANGE = _descriptor.Descriptor(
  name='ToneChange',
  full_name='handsDet.ToneChange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gain_bias', full_name='handsDet.ToneChange.gain_bias', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='affine', full_name='handsDet.ToneChange.affine', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mixture_gain_bias', full_name='handsDet.ToneChange.mixture_gain_bias', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mixture_affine', full_name='handsDet.ToneChange.mixture_affine', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mixture_domain_sigma', full_name='handsDet.ToneChange.mixture_domain_sigma', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frac_clipped', full_name='handsDet.ToneChange.frac_clipped', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='low_percentile', full_name='handsDet.ToneChange.low_percentile', index=6,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='low_mid_percentile', full_name='handsDet.ToneChange.low_mid_percentile', index=7,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mid_percentile', full_name='handsDet.ToneChange.mid_percentile', index=8,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='high_mid_percentile', full_name='handsDet.ToneChange.high_mid_percentile', index=9,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='high_percentile', full_name='handsDet.ToneChange.high_percentile', index=10,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='log_domain', full_name='handsDet.ToneChange.log_domain', index=11,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='handsDet.ToneChange.type', index=12,
      number=14, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stability_stats', full_name='handsDet.ToneChange.stability_stats', index=13,
      number=15, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TONECHANGE_STABILITYSTATS, ],
  enum_types=[
    _TONECHANGE_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=104,
  serialized_end=767,
)


_TONEMATCHOPTIONS = _descriptor.Descriptor(
  name='ToneMatchOptions',
  full_name='handsDet.ToneMatchOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_match_percentile', full_name='handsDet.ToneMatchOptions.min_match_percentile', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.01),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_match_percentile', full_name='handsDet.ToneMatchOptions.max_match_percentile', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.99),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='match_percentile_steps', full_name='handsDet.ToneMatchOptions.match_percentile_steps', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='patch_radius', full_name='handsDet.ToneMatchOptions.patch_radius', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=18,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_frac_clipped', full_name='handsDet.ToneMatchOptions.max_frac_clipped', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.4),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='log_domain', full_name='handsDet.ToneMatchOptions.log_domain', index=5,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
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
  serialized_start=770,
  serialized_end=980,
)


_CLIPMASKOPTIONS = _descriptor.Descriptor(
  name='ClipMaskOptions',
  full_name='handsDet.ClipMaskOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_exposure', full_name='handsDet.ClipMaskOptions.min_exposure', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.02),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_exposure', full_name='handsDet.ClipMaskOptions.max_exposure', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.98),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_clipped_channels', full_name='handsDet.ClipMaskOptions.max_clipped_channels', index=2,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip_mask_diameter', full_name='handsDet.ClipMaskOptions.clip_mask_diameter', index=3,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=5,
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
  serialized_start=983,
  serialized_end=1120,
)


_TONEESTIMATIONOPTIONS_GAINBIASBOUNDS = _descriptor.Descriptor(
  name='GainBiasBounds',
  full_name='handsDet.ToneEstimationOptions.GainBiasBounds',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_inlier_fraction', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.min_inlier_fraction', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.75),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_inlier_weight', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.min_inlier_weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lower_gain', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.lower_gain', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.75),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upper_gain', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.upper_gain', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1.334),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='lower_bias', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.lower_bias', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(-0.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='upper_bias', full_name='handsDet.ToneEstimationOptions.GainBiasBounds.upper_bias', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.2),
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
  serialized_start=1712,
  serialized_end=1899,
)

_TONEESTIMATIONOPTIONS = _descriptor.Descriptor(
  name='ToneEstimationOptions',
  full_name='handsDet.ToneEstimationOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tone_match_options', full_name='handsDet.ToneEstimationOptions.tone_match_options', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip_mask_options', full_name='handsDet.ToneEstimationOptions.clip_mask_options', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stats_low_percentile', full_name='handsDet.ToneEstimationOptions.stats_low_percentile', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.05),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stats_low_mid_percentile', full_name='handsDet.ToneEstimationOptions.stats_low_mid_percentile', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stats_mid_percentile', full_name='handsDet.ToneEstimationOptions.stats_mid_percentile', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stats_high_mid_percentile', full_name='handsDet.ToneEstimationOptions.stats_high_mid_percentile', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.8),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stats_high_percentile', full_name='handsDet.ToneEstimationOptions.stats_high_percentile', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.95),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='irls_iterations', full_name='handsDet.ToneEstimationOptions.irls_iterations', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stable_gain_bias_bounds', full_name='handsDet.ToneEstimationOptions.stable_gain_bias_bounds', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='downsample_mode', full_name='handsDet.ToneEstimationOptions.downsample_mode', index=9,
      number=10, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='downsampling_size', full_name='handsDet.ToneEstimationOptions.downsampling_size', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='downsample_factor', full_name='handsDet.ToneEstimationOptions.downsample_factor', index=11,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(2),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TONEESTIMATIONOPTIONS_GAINBIASBOUNDS, ],
  enum_types=[
    _TONEESTIMATIONOPTIONS_DOWNSAMPLEMODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1123,
  serialized_end=2020,
)


_TONEMATCH = _descriptor.Descriptor(
  name='ToneMatch',
  full_name='handsDet.ToneMatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='curr_val', full_name='handsDet.ToneMatch.curr_val', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prev_val', full_name='handsDet.ToneMatch.prev_val', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=2022,
  serialized_end=2069,
)


_PATCHTONEMATCH = _descriptor.Descriptor(
  name='PatchToneMatch',
  full_name='handsDet.PatchToneMatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tone_match', full_name='handsDet.PatchToneMatch.tone_match', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='irls_weight', full_name='handsDet.PatchToneMatch.irls_weight', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
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
  serialized_start=2071,
  serialized_end=2153,
)

_TONECHANGE_STABILITYSTATS.containing_type = _TONECHANGE
_TONECHANGE.fields_by_name['gain_bias'].message_type = handsDet_dot_util_dot_tracking_dot_tone__models__pb2._GAINBIASMODEL
_TONECHANGE.fields_by_name['affine'].message_type = handsDet_dot_util_dot_tracking_dot_tone__models__pb2._AFFINETONEMODEL
_TONECHANGE.fields_by_name['mixture_gain_bias'].message_type = handsDet_dot_util_dot_tracking_dot_tone__models__pb2._MIXTUREGAINBIASMODEL
_TONECHANGE.fields_by_name['mixture_affine'].message_type = handsDet_dot_util_dot_tracking_dot_tone__models__pb2._MIXTUREAFFINETONEMODEL
_TONECHANGE.fields_by_name['type'].enum_type = _TONECHANGE_TYPE
_TONECHANGE.fields_by_name['stability_stats'].message_type = _TONECHANGE_STABILITYSTATS
_TONECHANGE_TYPE.containing_type = _TONECHANGE
_TONEESTIMATIONOPTIONS_GAINBIASBOUNDS.containing_type = _TONEESTIMATIONOPTIONS
_TONEESTIMATIONOPTIONS.fields_by_name['tone_match_options'].message_type = _TONEMATCHOPTIONS
_TONEESTIMATIONOPTIONS.fields_by_name['clip_mask_options'].message_type = _CLIPMASKOPTIONS
_TONEESTIMATIONOPTIONS.fields_by_name['stable_gain_bias_bounds'].message_type = _TONEESTIMATIONOPTIONS_GAINBIASBOUNDS
_TONEESTIMATIONOPTIONS.fields_by_name['downsample_mode'].enum_type = _TONEESTIMATIONOPTIONS_DOWNSAMPLEMODE
_TONEESTIMATIONOPTIONS_DOWNSAMPLEMODE.containing_type = _TONEESTIMATIONOPTIONS
_PATCHTONEMATCH.fields_by_name['tone_match'].message_type = _TONEMATCH
DESCRIPTOR.message_types_by_name['ToneChange'] = _TONECHANGE
DESCRIPTOR.message_types_by_name['ToneMatchOptions'] = _TONEMATCHOPTIONS
DESCRIPTOR.message_types_by_name['ClipMaskOptions'] = _CLIPMASKOPTIONS
DESCRIPTOR.message_types_by_name['ToneEstimationOptions'] = _TONEESTIMATIONOPTIONS
DESCRIPTOR.message_types_by_name['ToneMatch'] = _TONEMATCH
DESCRIPTOR.message_types_by_name['PatchToneMatch'] = _PATCHTONEMATCH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ToneChange = _reflection.GeneratedProtocolMessageType('ToneChange', (_message.Message,), dict(

  StabilityStats = _reflection.GeneratedProtocolMessageType('StabilityStats', (_message.Message,), dict(
    DESCRIPTOR = _TONECHANGE_STABILITYSTATS,
    __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
    # @@protoc_insertion_point(class_scope:handsDet.ToneChange.StabilityStats)
    ))
  ,
  DESCRIPTOR = _TONECHANGE,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ToneChange)
  ))
_sym_db.RegisterMessage(ToneChange)
_sym_db.RegisterMessage(ToneChange.StabilityStats)

ToneMatchOptions = _reflection.GeneratedProtocolMessageType('ToneMatchOptions', (_message.Message,), dict(
  DESCRIPTOR = _TONEMATCHOPTIONS,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ToneMatchOptions)
  ))
_sym_db.RegisterMessage(ToneMatchOptions)

ClipMaskOptions = _reflection.GeneratedProtocolMessageType('ClipMaskOptions', (_message.Message,), dict(
  DESCRIPTOR = _CLIPMASKOPTIONS,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ClipMaskOptions)
  ))
_sym_db.RegisterMessage(ClipMaskOptions)

ToneEstimationOptions = _reflection.GeneratedProtocolMessageType('ToneEstimationOptions', (_message.Message,), dict(

  GainBiasBounds = _reflection.GeneratedProtocolMessageType('GainBiasBounds', (_message.Message,), dict(
    DESCRIPTOR = _TONEESTIMATIONOPTIONS_GAINBIASBOUNDS,
    __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
    # @@protoc_insertion_point(class_scope:handsDet.ToneEstimationOptions.GainBiasBounds)
    ))
  ,
  DESCRIPTOR = _TONEESTIMATIONOPTIONS,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ToneEstimationOptions)
  ))
_sym_db.RegisterMessage(ToneEstimationOptions)
_sym_db.RegisterMessage(ToneEstimationOptions.GainBiasBounds)

ToneMatch = _reflection.GeneratedProtocolMessageType('ToneMatch', (_message.Message,), dict(
  DESCRIPTOR = _TONEMATCH,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.ToneMatch)
  ))
_sym_db.RegisterMessage(ToneMatch)

PatchToneMatch = _reflection.GeneratedProtocolMessageType('PatchToneMatch', (_message.Message,), dict(
  DESCRIPTOR = _PATCHTONEMATCH,
  __module__ = 'handsDet.util.tracking.tone_estimation_pb2'
  # @@protoc_insertion_point(class_scope:handsDet.PatchToneMatch)
  ))
_sym_db.RegisterMessage(PatchToneMatch)


# @@protoc_insertion_point(module_scope)
